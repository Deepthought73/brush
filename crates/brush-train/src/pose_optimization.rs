//! Joint camera-pose optimization ("approach B": autodiff wrapper, no kernel
//! changes).
//!
//! The rasterizer takes the camera as a fixed uniform, so the pose is normally
//! outside the autodiff graph. But rendering *splats-at-world-pose* with a fixed
//! camera `C` is identical to rendering *(δ-transformed splats)* with `C`, where
//! δ is a per-view rigid correction. So we apply a learnable SE(3) delta to the
//! splat means/rotations using *differentiable* Burn tensor ops before handing
//! them to the renderer. The delta then lands on the same autodiff graph as the
//! splats: the existing backward pass produces `v_transforms` (∂L/∂mean,
//! ∂L/∂quat) and Burn's chain rule turns that into ∂L/∂δ for free — no WGSL /
//! kernel work required. This mirrors nerfstudio's `CameraOptimizer`.
//!
//! The correction is parameterized as SO(3)×ℝ³ (`[num_views, 6]`: axis-angle
//! rotation tangent + translation), initialized to zero (identity). Rotation is
//! applied about the camera center so rotation and translation stay decoupled
//! and well-conditioned.

use crate::{
    adam_scaled::{AdamScaled, AdamScaledConfig},
    config::TrainConfig,
    quat_vec::quaternion_vec_multiply,
};
use brush_render::camera::Camera;
use burn::{
    lr_scheduler::{
        LrScheduler,
        exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    },
    module::{Module, Param, ParamId},
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, Gradients, Tensor, s},
};

/// Per-view learnable SE(3) camera corrections, stored as a single
/// `[num_views, 6]` tensor: columns `0..3` are the SO(3) rotation tangent
/// (axis-angle) and `3..6` are the translation.
#[derive(Module, Debug)]
pub struct PoseParams {
    deltas: Param<Tensor<2>>,
}

/// Summary of the current pose-correction magnitudes (how far the poses have
/// moved from their originals). Rotation is the axis-angle length in radians,
/// translation is the offset length in world units — kept separate because
/// they have different units. When these plateau over training, the poses have
/// converged.
#[derive(Clone, Copy, Debug)]
pub struct PoseDeltaMagnitudes {
    pub rot_mean: f32,
    pub rot_max: f32,
    pub trans_mean: f32,
    pub trans_max: f32,
}

/// Owns the pose corrections plus their optimizer and LR schedule.
pub struct PoseOptimizer {
    params: PoseParams,
    optim: OptimizerAdaptor<AdamScaled, PoseParams>,
    sched: ExponentialLrScheduler,
}

/// Numerically-safe axis-angle (`[1, 3]`) → unit quaternion (`[1, 4]`, `wxyz`).
/// The `+eps` under the sqrt keeps the gradient finite at the identity (δ = 0),
/// where all deltas start.
fn axisangle_to_quat(w: Tensor<2>) -> Tensor<2> {
    let theta = w.clone().powi_scalar(2).sum_dim(1).add_scalar(1e-12).sqrt(); // [1,1]
    let half = theta.clone().mul_scalar(0.5);
    let qw = half.clone().cos(); // [1,1]
    // sin(θ/2)/θ → ½ as θ→0, so xyz → w/2 (the correct first-order term).
    let scale = half.sin().div(theta); // [1,1]
    let xyz = w.mul(scale); // [1,3]
    Tensor::cat(vec![qw, xyz], 1) // [1,4]
}

/// Hamilton product `a ⊗ b` (both `wxyz`). Broadcasts a `[1,4]` `a` against a
/// `[N,4]` `b`, returning `[N,4]`.
fn quat_mul(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    let aw = a.clone().slice(s![.., 0..1]);
    let ax = a.clone().slice(s![.., 1..2]);
    let ay = a.clone().slice(s![.., 2..3]);
    let az = a.slice(s![.., 3..4]);
    let bw = b.clone().slice(s![.., 0..1]);
    let bx = b.clone().slice(s![.., 1..2]);
    let by = b.clone().slice(s![.., 2..3]);
    let bz = b.slice(s![.., 3..4]);

    let w = aw.clone() * bw.clone() - ax.clone() * bx.clone() - ay.clone() * by.clone()
        - az.clone() * bz.clone();
    let x =
        aw.clone() * bx.clone() + ax.clone() * bw.clone() + ay.clone() * bz.clone()
            - az.clone() * by.clone();
    let y = aw.clone() * by.clone() - ax.clone() * bz.clone() + ay.clone() * bw.clone()
        + az.clone() * bx.clone();
    let z = aw * bz + ax * by - ay * bx + az * bw;
    Tensor::cat(vec![w, x, y, z], 1)
}

impl PoseOptimizer {
    pub fn new(num_views: usize, config: &TrainConfig, device: &Device) -> Self {
        // The pose deltas persist across steps and must carry gradients, so
        // they live on the autodiff device — matching the splat transforms once
        // those are lifted for the render (the splats themselves are stored on
        // the inner device and re-lifted each step, but the poses are not).
        // Guard against double-lifting (would trip "only first-order autodiff").
        let device = if device.is_autodiff() {
            device.clone()
        } else {
            device.clone().autodiff()
        };
        let deltas = Tensor::<2>::zeros([num_views, 6], &device);
        let params = PoseParams {
            deltas: Param::initialized(ParamId::new(), deltas.require_grad()),
        };

        // Exponential decay from `lr_pose` to `lr_pose_end` over training.
        let iters = config.total_train_iters.max(1) as f64;
        let decay = (config.lr_pose_end / config.lr_pose).powf(1.0 / iters);
        let sched = ExponentialLrSchedulerConfig::new(config.lr_pose, decay)
            .init()
            .expect("Pose lr schedule must be valid.");

        Self {
            params,
            optim: AdamScaledConfig::new().with_epsilon(1e-15).init(),
            sched,
        }
    }

    /// Apply the learned correction for `view_index` to a packed `[N, 10]`
    /// transforms tensor (means(3) + quat(4, `wxyz`) + log-scales(3)). Rotation
    /// is applied about `cam_pos` so it stays decoupled from translation.
    /// Differentiable w.r.t. both the transforms and the pose deltas.
    pub fn apply(&self, transforms: Tensor<2>, view_index: usize, cam_pos: glam::Vec3) -> Tensor<2> {
        let device = transforms.device();
        let n = transforms.dims()[0];

        let delta = self
            .params
            .deltas
            .val()
            .slice(s![view_index..view_index + 1, 0..6]); // [1,6]
        let w = delta.clone().slice(s![.., 0..3]); // [1,3]
        let t = delta.slice(s![.., 3..6]); // [1,3]
        let qd = axisangle_to_quat(w); // [1,4]

        let means = transforms.clone().slice(s![.., 0..3]); // [N,3]
        let quats = transforms.clone().slice(s![.., 3..7]); // [N,4]
        let scales = transforms.slice(s![.., 7..10]); // [N,3]

        let cam = Tensor::<1>::from_floats([cam_pos.x, cam_pos.y, cam_pos.z], &device).reshape([1, 3]);
        let means_c = means - cam.clone(); // [N,3]
        let qd_n = qd.clone().repeat_dim(0, n); // [N,4]
        let means_rot = quaternion_vec_multiply(qd_n, means_c); // [N,3]
        let means_new = means_rot + cam + t; // [N,3] (broadcast)

        let quats_new = quat_mul(qd, quats); // [N,4]

        Tensor::cat(vec![means_new, quats_new, scales], 1) // [N,10]
    }

    /// Consume the pose gradients from `grads` and take an optimizer step.
    /// Returns the learning rate used.
    pub fn optimize(&mut self, grads: &mut Gradients) -> f64 {
        let lr = self.sched.step();
        let grad = GradientsParams::from_params(grads, &self.params, &[self.params.deltas.id]);
        // Module clone is cheap (Arc-backed tensor handles); the optimizer
        // consumes and returns the module like the splat optimizer does.
        self.params = self.optim.step(lr, self.params.clone(), grad);
        lr
    }

    /// Read the pose deltas back to the CPU and summarize their magnitude
    /// (mean/max rotation in radians and translation in world units). Use this
    /// to track whether the poses have converged (magnitudes stop changing).
    pub async fn delta_magnitudes(&self) -> PoseDeltaMagnitudes {
        let data: Vec<f32> = self
            .params
            .deltas
            .val()
            .inner()
            .into_data_async()
            .await
            .expect("read pose deltas")
            .into_vec()
            .expect("pose deltas are f32");

        let n = data.len() / 6;
        let mut rot_sum = 0.0f32;
        let mut rot_max = 0.0f32;
        let mut trans_sum = 0.0f32;
        let mut trans_max = 0.0f32;
        for i in 0..n {
            let o = i * 6;
            let rot = (data[o] * data[o] + data[o + 1] * data[o + 1] + data[o + 2] * data[o + 2])
                .sqrt();
            let trans = (data[o + 3] * data[o + 3]
                + data[o + 4] * data[o + 4]
                + data[o + 5] * data[o + 5])
                .sqrt();
            rot_sum += rot;
            rot_max = rot_max.max(rot);
            trans_sum += trans;
            trans_max = trans_max.max(trans);
        }
        let inv = if n > 0 { 1.0 / n as f32 } else { 0.0 };
        PoseDeltaMagnitudes {
            rot_mean: rot_sum * inv,
            rot_max,
            trans_mean: trans_sum * inv,
            trans_max,
        }
    }

    /// Read the current per-view corrections back to the CPU and apply them to
    /// the `base` cameras, returning the *optimized* camera poses (for
    /// visualization). `base` must be in the same order as the training views.
    ///
    /// `apply` transforms the world as `p_world ↦ R·(p_world − c) + c + t`
    /// (rotation about the camera center `c`, then translate). Rendering that
    /// transformed world with the fixed camera is equivalent to rendering the
    /// original world with a camera whose pose is the inverse of that rigid
    /// motion, i.e. rotation `Rᵀ·R_cam` and position `c − Rᵀ·t`.
    pub async fn corrected_cameras(&self, base: &[Camera]) -> Vec<Camera> {
        let data: Vec<f32> = self
            .params
            .deltas
            .val()
            .inner()
            .into_data_async()
            .await
            .expect("read pose deltas")
            .into_vec()
            .expect("pose deltas are f32");

        base.iter()
            .enumerate()
            .map(|(i, cam)| {
                let o = i * 6;
                if o + 6 > data.len() {
                    return *cam;
                }
                let w = glam::vec3(data[o], data[o + 1], data[o + 2]);
                let t = glam::vec3(data[o + 3], data[o + 4], data[o + 5]);
                let qd = glam::Quat::from_scaled_axis(w);
                let qd_inv = qd.conjugate(); // unit quaternion → inverse == conjugate
                let mut c = *cam;
                c.position = cam.position - qd_inv * t;
                c.rotation = (qd_inv * cam.rotation).normalize();
                c
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    // Two splats, identity orientation. [N,10]: means, quat(wxyz), log-scales.
    fn sample_transforms(device: &Device) -> Tensor<2> {
        Tensor::<2>::from_floats(
            [
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
                [0.5, 0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device,
        )
    }

    async fn max_abs_diff(a: Tensor<2>, b: Tensor<2>) -> f32 {
        (a - b)
            .abs()
            .max()
            .into_data_async()
            .await
            .expect("readback")
            .into_vec::<f32>()
            .expect("f32")[0]
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn zero_delta_is_identity() {
        // Poses live on the autodiff device, so the transforms we feed `apply`
        // must too (backends must match).
        let device: Device =
            Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let cfg = TrainConfig::default();
        let opt = PoseOptimizer::new(2, &cfg, &device);
        let transforms = sample_transforms(&device);
        // Deltas start at zero → the correction must be the identity.
        let out = opt.apply(transforms.clone(), 0, glam::vec3(4.0, -1.0, 0.5));
        assert!(max_abs_diff(out, transforms).await < 1e-5);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn pure_translation_shifts_means_only() {
        let device: Device =
            Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let cfg = TrainConfig::default();
        let mut opt = PoseOptimizer::new(1, &cfg, &device);
        // Translation-only delta: (+0.1, +0.2, -0.3) on the last three columns.
        let d = Tensor::<2>::from_floats([[0.0, 0.0, 0.0, 0.1, 0.2, -0.3]], &device);
        opt.params.deltas = Param::initialized(ParamId::new(), d.require_grad());

        let transforms = sample_transforms(&device);
        let out = opt.apply(transforms.clone(), 0, glam::vec3(0.0, 0.0, 0.0));

        let mut expected = transforms;
        let shift = Tensor::<2>::from_floats([[0.1, 0.2, -0.3]], &device);
        let shifted = expected.clone().slice(s![.., 0..3]) + shift;
        expected = expected.slice_assign(s![.., 0..3], shifted);
        assert!(max_abs_diff(out, expected).await < 1e-5);
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn gradients_reach_pose_params() {
        // The whole point of approach B: a downstream loss on the corrected
        // transforms must produce a finite, non-zero gradient on the pose delta.
        // Backward needs an autodiff-enabled device.
        let device: Device =
            Device::from(brush_cube::test_helpers::test_device().await).autodiff();
        let cfg = TrainConfig::default();
        let mut opt = PoseOptimizer::new(1, &cfg, &device);
        let d = Tensor::<2>::zeros([1, 6], &device).require_grad();
        opt.params.deltas = Param::initialized(ParamId::new(), d);

        let transforms = sample_transforms(&device);
        let out = opt.apply(transforms, 0, glam::vec3(0.5, 0.5, 0.5));
        // A pose-sensitive scalar loss (weighted sum of means).
        let loss = out.slice(s![.., 0..3]).sum();
        let grads = loss.backward();
        let g = opt
            .params
            .deltas
            .grad(&grads)
            .expect("pose delta must receive a gradient");
        let g: Vec<f32> = g.into_data_async().await.expect("readback").into_vec().expect("f32");
        assert!(g.iter().all(|v| v.is_finite()), "grad must be finite: {g:?}");
        assert!(g.iter().any(|v| v.abs() > 1e-6), "grad must be non-zero: {g:?}");
    }
}
