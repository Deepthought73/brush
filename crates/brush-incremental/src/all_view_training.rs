use crate::IncrementalTrainer;
use brush_dataset::scene::{sample_to_packed_data_without_copy, SceneBatch};
use brush_train::config::TrainConfig;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use burn::module::AutodiffModule;
use std::time::Instant;
use burn::tensor::TensorData;
use brush_render::AlphaMode;

impl IncrementalTrainer {
    pub async fn train(&mut self) {
        if self.config.train_config.all_view_train_steps == 0 {
            return;
        }

        let start = Instant::now();
        let mut splats = self.splats.clone().unwrap();
        let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
        let config = self.create_all_view_train_config();
        let mut trainer = SplatTrainer::new(&config, &self.device, bounds);
        let trainer_init_dur = start.elapsed();

        let start = Instant::now();
        for _ in 0..config.total_train_iters {
            let batch = self.get_next_train_batch();

            let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(splats);
            let (new_diff, _stats) = trainer.step(batch, diff_splats).await;
            splats = new_diff.valid();
        }
        let train_dur = start.elapsed();

        log::info!("Trainer init: {trainer_init_dur:?}, Train dur: {train_dur:?}");

        if let Some(splat_sender) = &self.splat_sender {
            splat_sender.set(0, splats.clone());
        }

        self.splats = Some(splats);
    }

    fn get_next_train_batch(&mut self) -> SceneBatch {
        let frame_id = self.view_sampler.sample(&mut self.rng);
        let view = &self.train_views[&frame_id];

        let (img_packed, has_alpha) = sample_to_packed_data_without_copy(&view.image);

        let depth_tensor = TensorData::new(view.depth.clone(), [view.image.height(), view.image.width()]);

        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: AlphaMode::Masked,
            camera: view.camera,
            depth: Some(depth_tensor),
        }
    }

    fn create_all_view_train_config(&self) -> TrainConfig {
        let config = &self.config.train_config;
        let mut cfg = TrainConfig::default();
        cfg.total_train_iters = config.all_view_train_steps;
        cfg.lr_mean = config.lr_mean;
        cfg.lr_mean_end = config.lr_mean;
        cfg.ssim_weight = config.ssim_weight;
        cfg.anti_needle_loss_weight = config.anti_needle_loss_weight;
        cfg.depth_loss_weight = config.depth_loss_weight;
        cfg
    }
}
