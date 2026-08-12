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
use brush_train::to_init_splats;
use brush_train::train::{GpuBatch, SplatTrainer};
use burn::Tensor;
use burn::module::AutodiffModule;
use burn::tensor::TensorData;
use dashmap::DashSet;
use rand::SeedableRng;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;
use std::time::Instant;

const TRAINER_BOUNDING_BOX: BoundingBox = BoundingBox {
    center: glam::Vec3::ZERO,
    extent: glam::Vec3::new(2.5, 1.5, 1.0),
};

impl IncrementalTrainer {
    pub async fn add_host_view(&mut self, view: &mut ViewData) {
        let _guard = self.gpu_mutex.lock_arc();

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

        let burst_train_config = self.burst_train_config();

        let mut trainer =
            SplatTrainer::new(&burst_train_config, &self.device, TRAINER_BOUNDING_BOX);

        let mut gpu_batch: Option<GpuBatch> = None;

        for _ in 0..self.config.train_config.densify_at {
            let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(
                self.splats.as_ref().unwrap().clone(),
            );
            let gt = gpu_batch.get_or_insert_with(|| {
                GpuBatch::from_scene_batch(batch.clone(), &diff_splats.device())
            });
            let (new_diff, _) = trainer.step_prepared(gt, diff_splats).await;
            self.splats = Some(new_diff.valid());
        }

        if self.config.train_config.densify_max_samples > 0 {
            self.densify(view, added_depth_values).await;
        }

        trainer = SplatTrainer::new(
            &self.create_all_view_train_config(),
            &self.device,
            TRAINER_BOUNDING_BOX,
        );
        for _ in 0..self.config.train_config.single_view_train_steps {
            let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(
                self.splats.as_ref().unwrap().clone(),
            );
            let gt = gpu_batch.get_or_insert_with(|| {
                GpuBatch::from_scene_batch(batch.clone(), &diff_splats.device())
            });
            let (new_diff, _) = trainer.step_prepared(gt, diff_splats).await;
            self.splats = Some(new_diff.valid());
        }

        self.trainer = None;

        /* TODO nice for tuning the training params: renders the host view after training on it, maybe make this run optional
        let (img, _) = render_splats(
            self.splats.clone().unwrap(),
            &view.camera,
            glam::UVec2::new(view.image.width(), view.image.height()),
            Vec3::ZERO,
            None,
            TextureMode::Float,
        )
        .await;
        // Save the final rendered view to disk for inspection. Mirrors the
        // tensor -> Rgb32FImage -> rgb8 conversion used by `EvalSample::save_to_disk`.
        let render_rgb = img.slice(s![.., .., 0..3]);
        let [h, w, _] = render_rgb.dims();
        let save_result: anyhow::Result<()> = async {
            let data = render_rgb.into_data_async().await?.into_vec::<f32>()?;
            let img: image::DynamicImage = image::Rgb32FImage::from_raw(w as u32, h as u32, data)
                .expect("Rendered tensor must fit an RGB image")
                .into();
            let img = img.into_rgb8();
            let dir = std::path::Path::new("after_single_view_train_render");
            tokio::fs::create_dir_all(dir).await?;
            // Name by render timestamp (nanos since the epoch, zero-padded) so
            // lexicographic filename order matches the order views were rendered.
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = dir.join(format!("{ts:020}.png"));
            img.save(&path)?;
            Ok(())
        }
        .await;
        if let Err(e) = save_result {
            log::warn!("Failed to save rendered view {}: {e:?}", view.frame_id);
        }*/
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

        let depth = view.depth.as_ref().unwrap();
        let candidates: Vec<(usize, glam::Vec3, f32, f32)> = (0..h * w)
            .into_par_iter()
            .filter_map(|idx| {
                let d = depth[idx];
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

        if candidates.is_empty() {
            return;
        }

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

        self.add_by_means(
            means,
            sh_coeffs,
            Some(log_scales),
            self.config.cov_init_opacity,
        )
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

        let depth = view.depth.as_ref().unwrap();
        for v in (0..h).step_by(stride) {
            for u in (0..w).step_by(stride) {
                let idx = v * w + u;

                let d = depth[idx];
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
                if !self.config.init_scales_with_knn {
                    log_scales.extend_from_slice(&[log_s, log_s, log_s]);
                }
            }
        }

        if self.config.init_scales_with_knn {
            self.add_by_means(means, sh_coeffs, None, self.config.cov_init_opacity)
        } else {
            self.add_by_means(
                means,
                sh_coeffs,
                Some(log_scales),
                self.config.cov_init_opacity,
            )
        }
    }

    async fn densify(&mut self, view: &ViewData, added: &mut [bool]) {
        let start = Instant::now();

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
            .unwrap()
            .clamp(-1.0, 1.0);

        let num_samples =
            (cfg.densify_max_samples as f32 * (1.0 - 0.5 * mean_ssim - 0.5)).round() as usize;

        if num_samples == 0 {
            return;
        }

        let ssim_cpu = ssim
            .into_data_async()
            .await
            .unwrap()
            .into_vec::<f32>()
            .unwrap();

        let depth = view.depth.as_ref().unwrap();
        // Collect only the candidate pixels (weight > 0) into a compact set, so
        // the sampler never has to scan the many zero-weight (already-added or
        // no-depth) pixels.
        let mut candidate_idx: Vec<usize> = Vec::new();
        let mut candidate_wts: Vec<f32> = Vec::new();
        for idx in 0..w * h {
            if added[idx] || depth[idx] < 0.1 || ssim_cpu[idx] >= cfg.densify_ssim_threshold {
                continue;
            }
            let wt = if cfg.densify_recip_weighting {
                1.0 / (ssim_cpu[idx].clamp(-1.0, 1.0) + 1.0 + 0.1) - 1.0 / (2.0 + 0.1)
            } else {
                (1.0 - ssim_cpu[idx].clamp(-1.0, 1.0)) + 0.1
            };
            if wt > 0.0 {
                candidate_idx.push(idx);
                candidate_wts.push(wt);
            }
        }

        let valid = candidate_idx.len();
        let n = num_samples.min(valid);
        if n == 0 {
            return;
        }

        // Weighted sampling *with* replacement via an O(1)-per-draw alias table,
        // deduplicated against `added`. Since n << valid, collisions are rare so
        // the retry loop is cheap, and the result matches without-replacement
        // sampling. A fast (non-cryptographic) RNG is seeded from `self.rng` to
        // keep runs deterministic.
        let dist = rand_distr::weighted::WeightedAliasIndex::new(candidate_wts)
            .expect("failed to build ssim-weighted alias table");
        let mut rng = SmallRng::from_rng(&mut self.rng);
        let mut sampled: Vec<usize> = Vec::with_capacity(n);
        while sampled.len() < n {
            let idx = candidate_idx[dist.sample(&mut rng)];
            if !added[idx] {
                added[idx] = true;
                sampled.push(idx);
            }
        }

        let raw_img = view.image.as_rgba8().unwrap().as_raw();
        let mut means = Vec::with_capacity(n * 3);
        let mut sh_coeffs = Vec::with_capacity(n * 3);
        let depth = view.depth.as_ref().unwrap();
        for &idx in sampled.iter() {
            let u = idx % w;
            let v = idx / w;
            let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);
            let pos_world = view
                .camera
                .transform(view.camera.unproject(uv, depth[idx], img_size));
            let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;
            means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
            sh_coeffs.extend_from_slice(&[color, color, color]);
        }

        let log_scales = match cfg.densify_scale_mode {
            DensifyScaleMode::Constant => Some(vec![
                cfg.densify_const_cov_scale.max(1e-6).ln();
                means.len()
            ]),
            DensifyScaleMode::Knn => None,
        };

        self.add_by_means(means, sh_coeffs, log_scales, cfg.densify_init_opacity);
        log::info!("Adding to splats took: {:?}", start.elapsed());
    }

    fn add_by_means(
        &mut self,
        means: Vec<f32>,
        sh_coeffs: Vec<f32>,
        log_scales: Option<Vec<f32>>,
        init_opacity: f32,
    ) {
        let sh_degree = self.config.sh_degree;
        let render_mode = self.config.render_mode;

        let n_splats = means.len() / 3;
        let new_splat = to_init_splats(
            SplatData {
                means,
                rotations: None,
                log_scales,
                sh_coeffs: Some(sh_coeffs),
                raw_opacities: Some(vec![inverse_sigmoid(init_opacity); n_splats]),
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

        self.trainer = None;
    }

    fn build_scene_batch(&self, view: &ViewData) -> SceneBatch {
        let (img_packed, has_alpha) = sample_to_packed_data_without_copy(&view.image);
        let depth = view.depth.as_ref().map(|depth| {
            TensorData::new(depth.to_vec(), [view.image.height(), view.image.width()])
        });
        let view_index = self.train_frame_id_to_idx[&view.frame_id];
        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: AlphaMode::Masked,
            camera: view.camera,
            depth,
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

    fn burst_train_config(&self) -> TrainConfig {
        let cfg = &self.config.train_config;
        let mut train = TrainConfig::default();

        train.total_train_iters = cfg.densify_at;
        train.render_mode = Some(self.config.render_mode);

        train.lr_mean = cfg.single_view_lr_mean;
        train.lr_mean_end = cfg.single_view_lr_mean_end;
        train.lr_opac = cfg.single_view_lr_opac;
        train.lr_opac_end = cfg.single_view_lr_opac_end;
        train.lr_scale = cfg.single_view_lr_scale;
        train.lr_scale_end = cfg.single_view_lr_scale_end;

        train.ssim_weight = cfg.ssim_weight;
        train.anti_needle_loss_weight = cfg.anti_needle_loss_weight;
        train.depth_loss_weight = cfg.depth_loss_weight;

        train.max_cov_scale = cfg.max_cov_scale;
        train.max_cov_scale_loss_weight = cfg.max_cov_scale_loss_weight;

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
