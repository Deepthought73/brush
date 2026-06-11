use crate::incremental_train_stream::incremental_database::IncrementalDatabase;
use crate::incremental_train_stream::landmark_householding::OccupancyGrid;
use crate::{
    RunningProcess,
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
    slot::SlotSender,
    wait_for_device,
};
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::{
    AlphaMode,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
};
use brush_train::eval::eval_stats;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use burn::{module::AutodiffModule, tensor::Tensor};
use image::DynamicImage;
use std::sync::Arc;
use std::time::Instant;

pub mod config;
pub mod incremental_database;
mod landmark_householding;
mod ui_interface;
mod view_sampling;

pub type FrameId = u64;

pub struct ImageData {
    pub frame_id: FrameId,
    pub image: DynamicImage,
    pub depth: Vec<u16>,
}

pub struct PoseData {
    pub frame_id: FrameId,
    pub translation: glam::Vec3,
    pub quat: glam::Quat,
}

pub fn create_incremental_training_process(
    database: IncrementalDatabase,
    config: TrainStreamConfig,
) -> RunningProcess {
    let (splat_tx, splat_view) = crate::slot::channel();

    let stream = try_fn_stream(|emitter| async move {
        let mut train_ctx = IncrementalTrainContext::new(database, splat_tx, emitter, config).await;
        train_ctx.init_ui().await;
        train_ctx.run_train_loop().await
    });

    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
    }
}

pub struct IncrementalTrainContext {
    database: IncrementalDatabase,

    trainer: Option<SplatTrainer>,
    training_iteration: u32,
    training_start: Instant,
    splats: Option<Splats>,
    config: TrainStreamConfig,

    occupancy_grid: Option<OccupancyGrid>,

    device: burn::tensor::Device,

    // communication with ui
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
    splat_sender: SlotSender<Splats>,
    splat_sender_initialized: bool,
    up_axis: Option<glam::Vec3>,
    up_axis_factor_count: f32,
}

impl IncrementalTrainContext {
    async fn new(
        database: IncrementalDatabase,
        splat_sender: SlotSender<Splats>,
        emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
        config: TrainStreamConfig,
    ) -> Self {
        let device: burn::tensor::Device = wait_for_device().await.clone().into();
        device.seed(config.process_config.seed);

        Self {
            database,
            splat_sender,
            splats: None,
            trainer: None,
            training_iteration: 0,
            training_start: Instant::now(),
            emitter,
            config,
            occupancy_grid: None,
            device,
            up_axis: None,
            splat_sender_initialized: false,
            up_axis_factor_count: 0.0,
        }
    }

    async fn run_train_loop(&mut self) -> anyhow::Result<()> {
        log::info!("Start training thread");

        let mut last_gaussian_added = Instant::now();

        loop {
            if last_gaussian_added.elapsed().as_secs_f64()
                > self
                    .config
                    .incremental_train_config
                    .add_gaussians_every_secs
            {
                let unregistered_frames = self.database.get_unregistered_frames();

                if unregistered_frames.is_empty() {
                    self.refine().await;
                } else {
                    self.extend_gaussians(unregistered_frames).await;
                }
                last_gaussian_added = Instant::now();

                self.update_ui_dataset().await;
            }

            if self.trainer.is_some() && self.splats.is_some() && !self.database.is_empty() {
                self.train_step().await;

                let refine_every = self.config.train_config.refine_every;
                if self.training_iteration > 0
                    && self.training_iteration.is_multiple_of(refine_every)
                {
                    self.refine().await;
                }
            }

            if self.training_iteration.is_multiple_of(100) {
                self.update_train_status_ui().await;
                self.update_splat_in_ui().await;
            }

            if self
                .training_iteration
                .is_multiple_of(self.config.process_config.eval_every)
            {
                self.eval_step().await?;
            }

            brush_async::yield_now().await;
        }
    }

    async fn train_step(&mut self) {
        self.training_iteration += 1;
        self.occupancy_grid = None;

        let batch = self.database.get_next_train_batch();

        let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(
            self.splats.as_ref().unwrap().clone(),
        );
        let (new_diff, _stats) = self
            .trainer
            .as_mut()
            .unwrap()
            .step(batch, diff_splats)
            .await;
        self.splats = Some(new_diff.valid());

        self.splat_sender
            .set(0, self.splats.as_ref().unwrap().clone());
    }

    async fn refine(&mut self) {
        if self.splats.is_none() {
            return;
        }

        let current = self.splats.take().unwrap();
        let before_num = current.num_splats();
        let (new_splats, refine_stats) = self
            .trainer
            .as_mut()
            .unwrap()
            .refine(self.training_iteration, current)
            .await;
        let after_num = new_splats.num_splats();
        log::info!("Refinement: {before_num} -> {after_num}");
        self.emitter
            .emit(ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                cur_splat_count: refine_stats.total_splats,
                iter: self.training_iteration,
            }))
            .await;
        self.splats = Some(new_splats);
    }

    async fn eval_step(&self) -> anyhow::Result<()> {
        if let Some(splats) = self.splats.clone() {
            let mut psnr_sum = 0.;
            let mut ssim_sum = 0.;
            let eval_views = self.database.eval_views();
            for (_, camera, image) in &eval_views {
                let eval_result = eval_stats(
                    splats.clone(),
                    camera,
                    (**image).clone(),
                    AlphaMode::Masked,
                    &self.device,
                )
                .await?;

                psnr_sum += eval_result.psnr.clone().into_scalar_async::<f32>().await?;
                ssim_sum += eval_result.ssim.clone().into_scalar_async::<f32>().await?;
            }
            let psnr = psnr_sum / eval_views.len() as f32;
            let ssim = ssim_sum / eval_views.len() as f32;
            self.emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::EvalResult {
                    iter: self.training_iteration,
                    avg_psnr: psnr,
                    avg_ssim: ssim,
                }))
                .await;

            log::info!(
                "Train time: {:.2}, ITER: {}, PSNR: {}, SSIM: {}",
                self.training_start.elapsed().as_secs_f64(),
                self.training_iteration,
                psnr,
                ssim
            );
        }

        Ok(())
    }

    async fn extend_gaussians(&mut self, frames: Vec<(Camera, Arc<DynamicImage>, Arc<Vec<u16>>)>) {
        if frames.is_empty() {
            return;
        }

        log::info!("Add new Gaussians with new {} views", frames.len());

        self.update_last_images(frames.last().unwrap().1.clone())
            .await;

        if self.database.total_view_count() < 50 {
            self.update_up_axis(frames.iter().map(|it| it.0));
        }

        for (camera, image, depth) in frames {
            let start = Instant::now();
            self.add_new_landmarks_by_depth(camera, image, depth).await;
            let elapsed = start.elapsed();
            log::info!("Adding new landmarks took: {elapsed:?}");
        }

        if let Some(s) = &self.splats {
            let bounds = get_splat_bounds(s.clone(), BOUND_PERCENTILE).await;
            self.trainer = Some(SplatTrainer::new(
                &self.config.train_config,
                &self.device,
                bounds,
            ));
        }
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
