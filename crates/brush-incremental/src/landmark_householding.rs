use crate::IncrementalTrainContext;
use crate::config::DensifyScaleMode;
use brush_dataset::scene::{SceneBatch, sample_to_packed_data_without_copy};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::shaders::SH_C0;
use brush_render::{AlphaMode, Splats};
use brush_serde::SplatData;
use brush_train::config::TrainConfig;
use brush_train::eval::ssim_map;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use brush_train::{knn_scales_with_context, to_init_splats};
use burn::Tensor;
use burn::module::{AutodiffModule, Param};
use burn::tensor::TensorData;
use dashmap::DashSet;
use image::DynamicImage;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;
use std::time::Instant;

impl IncrementalTrainContext {
    async fn ensure_occupancy_grid_valid(&mut self) {
        if self.occupancy_grid.is_none() {
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
            self.occupancy_grid = Some(grid);
        }
    }

    pub async fn add_new_landmarks_by_depth(
        &mut self,
        camera: Camera,
        image: Arc<DynamicImage>,
        depth: Arc<Vec<f32>>,
    ) -> (usize, usize) {
        let mut means = vec![];
        let mut sh_coeffs = vec![];
        let mut log_scales = vec![];

        let w = image.width() as usize;
        let h = image.height() as usize;
        let img_size = glam::UVec2::new(image.width(), image.height());

        let raw_img = image.as_rgba8().unwrap().as_raw();

        let focal = camera.focal(img_size);
        let factor = self.config.cov_init_scale_factor;

        self.ensure_occupancy_grid_valid().await;
        let grid = self.occupancy_grid.as_ref().unwrap();

        let candidates: Vec<(glam::Vec3, f32, f32)> = (0..h * w)
            .into_par_iter()
            .filter_map(|idx| {
                let d = depth[idx];
                if d <= 0.01 {
                    return None;
                }

                let u = idx % w;
                let v = idx / w;
                let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);

                let pos_cam = camera.unproject(uv, d, img_size);
                let pos_world = camera.transform(pos_cam);

                if !grid.is_free(pos_world) {
                    return None;
                }

                let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;
                let log_s = (factor * d / focal.x).ln();
                Some((pos_world, color, log_s))
            })
            .collect();

        for (pos_world, color, log_s) in candidates {
            if !grid.is_free(pos_world) {
                continue;
            }
            grid.insert(pos_world);

            means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
            sh_coeffs.extend_from_slice(&[color, color, color]);
            log_scales.extend_from_slice(&[log_s, log_s, log_s]);
        }

        self.add_new_landmarks_by_means(means, Some(sh_coeffs), Some(log_scales))
    }

    /// SSIM-densification approach: seed a strided depth grid at the init cov
    /// size, then run a short training burst that repeatedly injects new
    /// Gaussians into the highest SSIM-error pixels of the new frame (the same
    /// densification as the single-view-experiment).
    pub async fn add_new_landmarks_by_ssim_densification(
        &mut self,
        camera: Camera,
        image: Arc<DynamicImage>,
        depth: Arc<Vec<f32>>,
    ) -> (usize, usize) {
        let w = image.width() as usize;
        let h = image.height() as usize;
        let mut added = vec![false; w * h];

        let (means, sh_coeffs, log_scales) =
            self.strided_depth_points(&camera, &image, &depth, &mut added);
        let (start, _) = self.add_new_landmarks_by_means(means, Some(sh_coeffs), Some(log_scales));

        let burst_config = self.burst_train_config();
        let batch = build_scene_batch(camera, &image, &depth);

        let bounds = get_splat_bounds(self.splats.clone().unwrap(), BOUND_PERCENTILE).await;
        let mut trainer = SplatTrainer::new(&burst_config, &self.device, bounds);

        let steps = self.config.densify_steps;
        let every = self.config.densify_every;

        for step in 1..=steps {
            if step.is_multiple_of(every)
                && let Some(new_trainer) = self
                    .ssim_densify_once(&camera, &image, &depth, &mut added, &burst_config)
                    .await
            {
                trainer = new_trainer;
            }

            let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(
                self.splats.as_ref().unwrap().clone(),
            );
            let (new_diff, _stats) = trainer.step(batch.clone(), diff_splats).await;
            self.splats = Some(new_diff.valid());
        }

        let end = self.splats.as_ref().unwrap().num_splats() as usize;
        (start, end)
    }

    fn strided_depth_points(
        &self,
        camera: &Camera,
        image: &DynamicImage,
        depth: &[f32],
        added: &mut [bool],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let w = image.width() as usize;
        let h = image.height() as usize;
        let img_size = glam::UVec2::new(image.width(), image.height());
        let raw_img = image.as_rgba8().unwrap().as_raw();
        let focal = camera.focal(img_size);
        let stride = self.config.gaussians_init_depth_stride.max(1);
        let factor = self.config.cov_init_scale_factor;

        let candidates: Vec<(usize, glam::Vec3, f32, f32)> = (0..h * w)
            .into_par_iter()
            .filter_map(|idx| {
                let d = depth[idx];
                if d <= 0.01 {
                    return None;
                }

                let u = idx % w;
                let v = idx / w;
                if !u.is_multiple_of(stride) || !v.is_multiple_of(stride) {
                    return None;
                }

                let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);
                let pos_world = camera.transform(camera.unproject(uv, d, img_size));
                let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;
                let log_s = (factor * d / focal.x).ln();
                Some((idx, pos_world, color, log_s))
            })
            .collect();

        let mut means = Vec::with_capacity(candidates.len() * 3);
        let mut sh_coeffs = Vec::with_capacity(candidates.len() * 3);
        let mut log_scales = Vec::with_capacity(candidates.len() * 3);
        for (idx, pos_world, color, log_s) in candidates {
            added[idx] = true;
            means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
            sh_coeffs.extend_from_slice(&[color, color, color]);
            log_scales.extend_from_slice(&[log_s, log_s, log_s]);
        }
        (means, sh_coeffs, log_scales)
    }

    async fn ssim_densify_once(
        &mut self,
        camera: &Camera,
        image: &DynamicImage,
        depth: &[f32],
        added: &mut [bool],
        burst_config: &TrainConfig,
    ) -> Option<SplatTrainer> {
        let cfg = self.config.clone();
        let w = image.width() as usize;
        let h = image.height() as usize;
        let img_size = glam::UVec2::new(image.width(), image.height());

        let splats = self.splats.clone().unwrap();
        let ssim = ssim_map(
            splats,
            camera,
            image.clone(),
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
            return None;
        }

        let ssim_cpu = ssim
            .into_data_async()
            .await
            .expect("failed to read ssim map")
            .into_vec::<f32>()
            .expect("ssim map should be f32");

        let weights: Vec<f32> = (0..w * h)
            .map(|idx| {
                if added[idx] || depth[idx] < 0.1 {
                    0.0
                } else if cfg.densify_recip_weighting {
                    1.0 / (ssim_cpu[idx].clamp(-1.0, 1.0) + 1.0 + cfg.densify_floor)
                        - 1.0 / (2.0 + cfg.densify_floor)
                } else {
                    (1.0 - ssim_cpu[idx].clamp(-1.0, 1.0)) + cfg.densify_floor
                }
            })
            .collect();

        let valid = weights.iter().filter(|&&wt| wt > 0.0).count();
        let n = num_samples.min(valid);
        if n == 0 {
            return None;
        }

        let sampled = {
            let mut rng = rand::rng();
            rand::seq::index::sample_weighted(&mut rng, w * h, |i| weights[i], n)
                .expect("failed to sample ssim-weighted pixels")
        };

        let raw_img = image.as_rgba8().unwrap().as_raw();
        let mut means = Vec::with_capacity(n * 3);
        let mut sh_coeffs = Vec::with_capacity(n * 3);
        for idx in sampled.iter() {
            added[idx] = true;
            let u = idx % w;
            let v = idx / w;
            let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);
            let pos_world = camera.transform(camera.unproject(uv, depth[idx], img_size));
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

        self.add_new_landmarks_by_means(means, Some(sh_coeffs), Some(log_scales));

        let bounds = get_splat_bounds(self.splats.clone().unwrap(), BOUND_PERCENTILE).await;
        Some(SplatTrainer::new(burst_config, &self.device, bounds))
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

    fn burst_train_config(&self) -> TrainConfig {
        let cfg = &self.config;
        let mut train = TrainConfig::default();
        train.total_train_iters = cfg.densify_steps;
        train.lr_mean = cfg.densify_lr_mean;
        train.lr_mean_end = cfg.densify_lr_mean;
        train.lr_scale = cfg.densify_lr_scale;
        train.ssim_weight = cfg.densify_ssim_weight;
        train.depth_loss_weight = cfg.densify_depth_loss;
        train.anti_needle_loss_weight = cfg.densify_anti_needle_loss;
        train
    }

    fn add_new_landmarks_by_means(
        &mut self,
        means: Vec<f32>,
        sh_coeffs: Option<Vec<f32>>,
        log_scales: Option<Vec<f32>>,
    ) -> (usize, usize) {
        let sh_degree = self.config.sh_degree;
        let render_mode = self.config.render_mode;

        let n_splats = means.len() / 3;
        let new_splat = to_init_splats(
            SplatData {
                means,
                rotations: None,
                log_scales,
                sh_coeffs,
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

        let splats_before = splats.as_ref().map(|it| it.num_splats()).unwrap_or(0) as usize;
        let splats_after = splats_before + n_splats;

        self.splats = Some(match splats {
            None => new_splat,
            Some(existing) => concat_splats(&existing, &new_splat, render_mode),
        });

        (splats_before, splats_after)
    }

    pub async fn update_poses(&mut self) {
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
    }
}

#[derive(Clone)]
pub struct OccupancyGrid {
    inv_grid_size: f32,
    cells: Arc<DashSet<[i32; 3]>>,
}

impl OccupancyGrid {
    pub fn new(grid_size: f32) -> Self {
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

fn build_scene_batch(camera: Camera, image: &DynamicImage, depth: &[f32]) -> SceneBatch {
    let (img_packed, has_alpha) = sample_to_packed_data_without_copy(image);
    let depth_tensor = TensorData::new(depth.to_vec(), [image.height(), image.width()]);
    SceneBatch {
        img_packed,
        has_alpha,
        alpha_mode: AlphaMode::Masked,
        camera,
        depth: Some(depth_tensor),
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
