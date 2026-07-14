use crate::IncrementalTrainer;
use brush_dataset::scene::{SceneBatch, sample_to_packed_data_without_copy};
use brush_render::AlphaMode;
use brush_render::bounding_box::BoundingBox;
use brush_train::config::TrainConfig;
use brush_train::train::SplatTrainer;
use burn::module::AutodiffModule;
use burn::tensor::TensorData;
use std::time::{Duration, Instant};

const TRAINER_BOUNDING_BOX: BoundingBox = BoundingBox {
    center: glam::Vec3::ZERO,
    extent: glam::Vec3::new(2.5, 1.5, 1.0),
};

impl IncrementalTrainer {
    pub async fn train(&mut self) {
        let _guard = self.gpu_mutex.lock_arc();

        let train_duration = Duration::from_secs_f64(self.config.train_config.all_view_train_secs);
        if train_duration.is_zero() {
            return;
        }

        let start = Instant::now();

        self.ensure_trainer();

        let mut trainer = self.trainer.take().unwrap();
        let mut splats = self.splats.take().unwrap();

        while start.elapsed() < train_duration {
            let batch = self.get_next_train_batch();

            let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(splats);
            let (new_diff, _stats) = trainer.step(batch, diff_splats).await;
            splats = new_diff.valid();
        }

        log::info!("Trained on all for: {:?}", start.elapsed());

        let base: Vec<_> = self.train_views.iter().map(|v| v.camera).collect();
        if let Some(corrected) = trainer.corrected_train_cameras(&base).await {
            for (view, cam) in self.train_views.iter_mut().zip(corrected) {
                view.camera = cam;
            }
        }

        if let Some(ui_ctx) = &self.ui_ctx {
            ui_ctx.splat_sender.set(0, splats.clone());
        }

        self.trainer = Some(trainer);
        self.splats = Some(splats);
    }

    fn get_next_train_batch(&mut self) -> SceneBatch {
        let view_index = self.view_sampler.sample(&mut self.rng);
        let view = &self.train_views[view_index];

        let (img_packed, has_alpha) = sample_to_packed_data_without_copy(&view.image);

        let depth = view
            .depth
            .as_ref()
            .map(|depth| TensorData::new(depth.clone(), [view.image.height(), view.image.width()]));

        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: AlphaMode::Masked,
            camera: view.camera,
            depth,
            view_index,
        }
    }

    fn ensure_trainer(&mut self) {
        if self.trainer.is_none() {
            let config = self.create_all_view_train_config();
            let trainer = SplatTrainer::new(&config, &self.device, TRAINER_BOUNDING_BOX);
            self.trainer = Some(trainer);
        }

        if self.config.train_config.pose_opt {
            self.trainer
                .as_mut()
                .unwrap()
                .enable_pose_opt(self.train_views.len(), &self.device);
        }
    }

    fn create_all_view_train_config(&self) -> TrainConfig {
        let config = &self.config.train_config;
        let mut cfg = TrainConfig::default();
        cfg.lr_mean = config.lr_mean;
        cfg.lr_mean_end = config.lr_mean;
        cfg.depth_loss_weight = config.depth_loss_weight;
        cfg.anti_needle_loss_weight = config.anti_needle_loss_weight;
        cfg.pose_opt = config.pose_opt;
        cfg.lr_pose = config.lr_pose_opt;
        cfg.lr_pose_end = config.lr_pose_opt;
        cfg
    }
}
