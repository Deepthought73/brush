use crate::IncrementalTrainer;
use brush_dataset::scene::{SceneBatch, sample_to_packed_data_without_copy};
use brush_render::AlphaMode;
use brush_render::bounding_box::BoundingBox;
use brush_train::config::TrainConfig;
use brush_train::train::SplatTrainer;
use burn::module::AutodiffModule;
use burn::tensor::TensorData;
use std::time::Instant;

const TRAINER_BOUNDING_BOX: BoundingBox = BoundingBox {
    center: glam::Vec3::ZERO,
    extent: glam::Vec3::new(2.5, 1.5, 1.0),
};

impl IncrementalTrainer {
    pub async fn train(&mut self) {
        if self.config.train_config.all_view_train_steps == 0 {
            return;
        }

        let start = Instant::now();

        let config = self.create_all_view_train_config();
        let mut trainer = SplatTrainer::new(&config, &self.device, TRAINER_BOUNDING_BOX);
        trainer.enable_pose_opt(self.train_views.len(), &self.device);

        let mut splats = self.splats.clone().unwrap();

        for _ in 0..config.total_train_iters {
            let batch = self.get_next_train_batch();

            let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(splats);
            let (new_diff, _stats) = trainer.step(batch, diff_splats).await;
            splats = new_diff.valid();
        }

        let train_dur = start.elapsed();

        log::info!("Train dur: {train_dur:?}");

        // Fold the learned per-view pose corrections back into the stored CPU
        // cameras so they persist. `base` must be in training-view order, which
        // matches how `view_index` is assigned (`frame_id_to_idx`). The trainer
        // — and its deltas, which are zero-initialised each `new` — is dropped
        // at the end of this call, so the corrected cameras become the new base
        // for the next round with no double-application.
        let base: Vec<_> = self.train_views.iter().map(|v| v.camera).collect();
        if let Some(corrected) = trainer.corrected_train_cameras(&base).await {
            for (view, cam) in self.train_views.iter_mut().zip(corrected) {
                view.camera = cam;
            }
        }

        if let Some(splat_sender) = &self.splat_sender {
            splat_sender.set(0, splats.clone());
        }

        self.splats = Some(splats);
    }

    fn get_next_train_batch(&mut self) -> SceneBatch {
        let view_index = self.view_sampler.sample(&mut self.rng);
        let view = &self.train_views[view_index];

        let (img_packed, has_alpha) = sample_to_packed_data_without_copy(&view.image);

        let depth_tensor = TensorData::new(
            view.depth.clone(),
            [view.image.height(), view.image.width()],
        );

        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: AlphaMode::Masked,
            camera: view.camera,
            depth: Some(depth_tensor),
            view_index,
        }
    }

    fn create_all_view_train_config(&self) -> TrainConfig {
        let config = &self.config.train_config;
        let mut cfg = TrainConfig::default();
        cfg.total_train_iters = config.all_view_train_steps;
        cfg.lr_mean_end = config.lr_mean;
        cfg.anti_needle_loss_weight = config.anti_needle_loss_weight;
        cfg.depth_loss_weight = config.depth_loss_weight;
        cfg.pose_opt = config.pose_opt;
        cfg.lr_pose = config.lr_pose_opt;
        cfg.lr_pose_end = config.lr_pose_opt;
        cfg
    }
}
