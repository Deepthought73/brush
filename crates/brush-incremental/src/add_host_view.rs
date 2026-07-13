use crate::config::{DensifyScaleMode, GaussianAddingMode};
use crate::{IncrementalTrainer, ViewData};
use brush_dataset::scene::{SceneBatch, sample_to_packed_data_without_copy};
use brush_render::bounding_box::BoundingBox;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::shaders::SH_C0;
use brush_render::{AlphaMode, Splats};
use brush_serde::SplatData;
use brush_train::config::TrainConfig;
use brush_train::eval::ssim_map;
use brush_train::train::{GpuBatch, SplatTrainer};
use brush_train::{knn_scales_with_context, to_init_splats};
use burn::Tensor;
use burn::module::AutodiffModule;
use burn::tensor::TensorData;
use dashmap::DashSet;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;

const TRAINER_BOUNDING_BOX: BoundingBox = BoundingBox {
    center: glam::Vec3::ZERO,
    extent: glam::Vec3::new(2.5, 1.5, 1.0),
};

impl IncrementalTrainer {
    pub async fn add_host_view(&mut self, view: &ViewData) {
        let w = view.image.width() as usize;
        let h = view.image.height() as usize;
        let mut added_depth_values = vec![false; w * h];

        let splats_before = self.splats.as_ref().map(|it| it.num_splats()).unwrap_or(0) as usize;

        match self.config.landmark_add_mode {
            GaussianAddingMode::OccupancyGrid => {
                self.add_with_occupancy_grid(view, &mut added_depth_values)
                    .await
            }
            GaussianAddingMode::StridedDepth => {
                self.add_from_strided_depth(view, &mut added_depth_values)
            }
        };

        self.train_view(view, &mut added_depth_values).await;

        let splats_after = self.splats.as_ref().unwrap().num_splats() as usize;

        self.corresponding_splats
            .insert(view.frame_id, (splats_before, splats_after));
    }

    async fn train_view(&mut self, view: &ViewData, added_depth_values: &mut [bool]) {
        let batch = self.build_scene_batch(view);

        let train_config = self.single_view_train_config();

        let mut trainer = SplatTrainer::new(&train_config, &self.device, TRAINER_BOUNDING_BOX);

        let mut gpu_batch: Option<GpuBatch> = None;

        for step in 1..=train_config.total_train_iters {
            if step.is_multiple_of(self.config.train_config.densify_every) {
                self.densify_by_ssim(view, added_depth_values).await;
                trainer = SplatTrainer::new(
                    &self.single_view_train_config(),
                    &self.device,
                    TRAINER_BOUNDING_BOX,
                );
            }

            let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(
                self.splats.as_ref().unwrap().clone(),
            );
            let gt = gpu_batch.get_or_insert_with(|| {
                GpuBatch::from_scene_batch(batch.clone(), &diff_splats.device())
            });
            let (new_diff, _stats) = trainer.step_prepared(gt, diff_splats).await;
            self.splats = Some(new_diff.valid());
        }
    }

    async fn add_with_occupancy_grid(&mut self, view: &ViewData, added: &mut [bool]) {
        let mut means = vec![];
        let mut sh_coeffs = vec![];
        let mut log_scales = vec![];

        let w = view.image.width() as usize;
        let h = view.image.height() as usize;
        let img_size = view.glam_img_size();

        let raw_img = view.image.as_rgba8().unwrap().as_raw();

        let focal = view.camera.focal(img_size);
        let factor = self.config.cov_init_scale_factor;

        let grid = self.compute_occupancy_grid().await;

        let candidates: Vec<(usize, glam::Vec3, f32, f32)> = (0..h * w)
            .into_par_iter()
            .filter_map(|idx| {
                let d = view.depth[idx];
                if d <= 0.01 {
                    return None;
                }

                let u = idx % w;
                let v = idx / w;
                let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);

                let pos_cam = view.camera.unproject(uv, d, img_size);
                let pos_world = view.camera.transform(pos_cam);

                if !grid.is_free(pos_world) {
                    return None;
                }

                let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;
                let log_s = (factor * d / focal.x).ln();
                Some((idx, pos_world, color, log_s))
            })
            .collect();

        for (idx, pos_world, color, log_s) in candidates {
            if !grid.is_free(pos_world) {
                continue;
            }
            grid.insert(pos_world);

            added[idx] = true;
            means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
            sh_coeffs.extend_from_slice(&[color, color, color]);
            log_scales.extend_from_slice(&[log_s, log_s, log_s]);
        }

        self.add_by_means(means, sh_coeffs, log_scales)
    }

    fn add_from_strided_depth(&mut self, view: &ViewData, added: &mut [bool]) {
        let w = view.image.width() as usize;
        let h = view.image.height() as usize;
        let img_size = view.glam_img_size();
        let raw_img = view.image.as_rgba8().unwrap().as_raw();
        let focal = view.camera.focal(img_size);
        let stride = self.config.gaussians_init_depth_stride;
        let factor = self.config.cov_init_scale_factor;

        let mut means = vec![];
        let mut sh_coeffs = vec![];
        let mut log_scales = vec![];

        for v in (0..h).step_by(stride) {
            for u in (0..w).step_by(stride) {
                let idx = v * w + u;

                let d = view.depth[idx];
                if d <= 0.01 {
                    continue;
                }

                let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);
                let pos_world = view
                    .camera
                    .transform(view.camera.unproject(uv, d, img_size));
                let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;
                let log_s = (factor * d / focal.x).ln();

                added[idx] = true;
                means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
                sh_coeffs.extend_from_slice(&[color, color, color]);
                log_scales.extend_from_slice(&[log_s, log_s, log_s]);
            }
        }

        self.add_by_means(means, sh_coeffs, log_scales)
    }

    async fn densify_by_ssim(&mut self, view: &ViewData, added: &mut [bool]) {
        let cfg = self.config.train_config.clone();
        let w = view.image.width() as usize;
        let h = view.image.height() as usize;
        let img_size = view.glam_img_size();

        let splats = self.splats.clone().unwrap();
        let ssim = ssim_map(
            splats,
            &view.camera,
            view.image.clone(),
            AlphaMode::Masked,
            &self.device,
        )
        .await
        .mean_dims(&[2]);

        let mean_ssim = ssim
            .clone()
            .mean()
            .into_scalar_async::<f32>()
            .await
            .expect("failed to read mean ssim")
            .clamp(-1.0, 1.0);

        let num_samples =
            (cfg.densify_max_samples as f32 * (1.0 - 0.5 * mean_ssim - 0.5)).round() as usize;
        if num_samples == 0 {
            return;
        }

        let ssim_cpu = ssim
            .into_data_async()
            .await
            .expect("failed to read ssim map")
            .into_vec::<f32>()
            .expect("ssim map should be f32");

        let weights: Vec<f32> = (0..w * h)
            .map(|idx| {
                if added[idx] || view.depth[idx] < 0.1 {
                    0.0
                } else if cfg.densify_recip_weighting {
                    1.0 / (ssim_cpu[idx].clamp(-1.0, 1.0) + 1.0 + 0.1) - 1.0 / (2.0 + 0.1)
                } else {
                    (1.0 - ssim_cpu[idx].clamp(-1.0, 1.0)) + 0.1
                }
            })
            .collect();

        let valid = weights.iter().filter(|&&wt| wt > 0.0).count();
        let n = num_samples.min(valid);
        if n == 0 {
            return;
        }

        let sampled = {
            let mut rng = rand::rng();
            rand::seq::index::sample_weighted(&mut rng, w * h, |i| weights[i], n)
                .expect("failed to sample ssim-weighted pixels")
        };

        let raw_img = view.image.as_rgba8().unwrap().as_raw();
        let mut means = Vec::with_capacity(n * 3);
        let mut sh_coeffs = Vec::with_capacity(n * 3);
        for idx in sampled.iter() {
            added[idx] = true;
            let u = idx % w;
            let v = idx / w;
            let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);
            let pos_world =
                view.camera
                    .transform(view.camera.unproject(uv, view.depth[idx], img_size));
            let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;
            means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
            sh_coeffs.extend_from_slice(&[color, color, color]);
        }

        let log_scales = match cfg.densify_scale_mode {
            DensifyScaleMode::Constant => {
                vec![cfg.densify_const_cov_scale.max(1e-6).ln(); means.len()]
            }
            DensifyScaleMode::Knn => {
                let existing = self.read_means().await;
                knn_scales_with_context(&existing, &means)
            }
        };

        self.add_by_means(means, sh_coeffs, log_scales);
    }

    fn add_by_means(&mut self, means: Vec<f32>, sh_coeffs: Vec<f32>, log_scales: Vec<f32>) {
        let sh_degree = self.config.sh_degree;
        let render_mode = self.config.render_mode;

        let n_splats = means.len() / 3;
        let new_splat = to_init_splats(
            SplatData {
                means,
                rotations: None,
                log_scales: Some(log_scales),
                sh_coeffs: Some(sh_coeffs),
                raw_opacities: Some(vec![
                    inverse_sigmoid(self.config.cov_init_opacity);
                    n_splats
                ]),
            },
            render_mode,
            &self.device,
        )
        .with_sh_degree(sh_degree);

        let splats = self.splats.take();

        self.splats = Some(match splats {
            None => new_splat,
            Some(existing) => concat_splats(&existing, &new_splat, render_mode),
        });
    }

    /* TODO
    async fn update_poses(&mut self) {
        let start = Instant::now();

        let pose_updates = self.database.collect_pose_updates();

        if pose_updates.is_empty() {
            return;
        }

        let updates: Vec<(glam::Vec3, glam::Quat, usize, usize)> = pose_updates
            .into_iter()
            .map(|(frame_id, delta_d, delta_q)| {
                let (start, end) = self.corresponding_splats.get(&frame_id).unwrap();
                (delta_d, delta_q, *start, *end)
            })
            .collect();

        let Some(splats) = self.splats.as_mut() else {
            return;
        };

        let id = splats.transforms.id;
        let device = splats.transforms.device();
        let dims = splats.transforms.dims();
        let mut data = splats
            .transforms
            .val()
            .into_data_async()
            .await
            .expect("failed to read splat transforms")
            .into_vec::<f32>()
            .expect("transforms tensor should be f32");

        for (delta_d, delta_q, start, end) in updates {
            for i in start..end {
                let base = i * 10;

                let mean = glam::Vec3::new(data[base], data[base + 1], data[base + 2]);
                let mean = delta_q * mean + delta_d;
                data[base] = mean.x;
                data[base + 1] = mean.y;
                data[base + 2] = mean.z;

                let q = glam::Quat::from_xyzw(
                    data[base + 4],
                    data[base + 5],
                    data[base + 6],
                    data[base + 3],
                );
                let q = (delta_q * q).normalize();
                data[base + 3] = q.w;
                data[base + 4] = q.x;
                data[base + 5] = q.y;
                data[base + 6] = q.z;
            }
        }

        let transforms = Tensor::from_data(TensorData::new(data, dims), &device);
        splats.transforms = Param::initialized(id, transforms.detach().require_grad());

        log::info!("Updating poses took {:?}", start.elapsed());
    }*/

    fn build_scene_batch(&self, view: &ViewData) -> SceneBatch {
        let (img_packed, has_alpha) = sample_to_packed_data_without_copy(&view.image);
        let depth_tensor = TensorData::new(
            view.depth.to_vec(),
            [view.image.height(), view.image.width()],
        );
        let view_index = self.frame_id_to_idx[&view.frame_id];
        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: AlphaMode::Masked,
            camera: view.camera,
            depth: Some(depth_tensor),
            view_index,
        }
    }

    async fn compute_occupancy_grid(&self) -> OccupancyGrid {
        let min_dist = self.config.occupancy_grid_size;
        let grid = OccupancyGrid::new(min_dist);

        if let Some(s) = &self.splats {
            let data = s
                .means()
                .into_data_async()
                .await
                .expect("failed to read gaussian means")
                .into_vec::<f32>()
                .expect("means tensor should be f32");

            data.as_chunks::<3>().0.par_iter().for_each(|it| {
                grid.insert(glam::Vec3::from_slice(it));
            });
        }

        grid
    }

    async fn read_means(&self) -> Vec<f32> {
        self.splats
            .as_ref()
            .unwrap()
            .means()
            .into_data_async()
            .await
            .expect("failed to read gaussian means")
            .into_vec::<f32>()
            .expect("means tensor should be f32")
    }

    fn single_view_train_config(&self) -> TrainConfig {
        let cfg = &self.config.train_config;
        let mut train = TrainConfig::default();

        train.total_train_iters = cfg.single_view_train_steps;
        train.render_mode = Some(self.config.render_mode);

        train.lr_mean = cfg.lr_mean;
        train.lr_mean_end = cfg.lr_mean_end;
        train.mean_noise_weight = cfg.mean_noise_weight;

        train.lr_mean = cfg.lr_mean;
        train.lr_mean_end = cfg.lr_mean_end;
        train.mean_noise_weight = cfg.mean_noise_weight;
        train.lr_coeffs_dc = cfg.lr_coeffs_dc;
        train.lr_coeffs_sh_scale = cfg.lr_coeffs_sh_scale;
        train.lr_opac = cfg.lr_opac;
        train.lr_scale = cfg.lr_scale;
        train.lr_rotation = cfg.lr_rotation;
        train.ssim_weight = cfg.ssim_weight;
        train.anti_needle_loss_weight = cfg.anti_needle_loss_weight;
        train.depth_loss_weight = cfg.depth_loss_weight;

        train
    }
}

#[derive(Clone)]
struct OccupancyGrid {
    inv_grid_size: f32,
    cells: Arc<DashSet<[i32; 3]>>,
}

impl OccupancyGrid {
    fn new(grid_size: f32) -> Self {
        Self {
            inv_grid_size: 1.0 / grid_size,
            cells: Default::default(),
        }
    }

    fn cell_of(&self, p: glam::Vec3) -> [i32; 3] {
        [
            (p.x * self.inv_grid_size).floor() as i32,
            (p.y * self.inv_grid_size).floor() as i32,
            (p.z * self.inv_grid_size).floor() as i32,
        ]
    }

    fn insert(&self, p: glam::Vec3) {
        self.cells.insert(self.cell_of(p));
    }

    fn is_free(&self, p: glam::Vec3) -> bool {
        !self.cells.contains(&self.cell_of(p))
    }
}

fn concat_splats(a: &Splats, b: &Splats, mode: SplatRenderMode) -> Splats {
    let means = Tensor::cat(vec![a.means(), b.means()], 0);
    let rotations = Tensor::cat(vec![a.rotations(), b.rotations()], 0);
    let log_scales = Tensor::cat(vec![a.log_scales(), b.log_scales()], 0);
    let sh_coeffs = Tensor::cat(vec![a.sh_coeffs.val(), b.sh_coeffs.val()], 0);
    let opacities = Tensor::cat(vec![a.raw_opacities.val(), b.raw_opacities.val()], 0);
    Splats::from_tensor_data(means, rotations, log_scales, sh_coeffs, opacities, mode)
}
