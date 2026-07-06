use crate::config::IncrementalTrainConfig;
use anyhow::Context;
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::slot::SlotSender;
use brush_process::{RunningProcess, slot, wait_for_device};
use brush_render::{AlphaMode, camera::Camera, gaussian_splats::Splats};
use brush_train::eval::eval_stats;
use image::DynamicImage;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

pub mod config;
mod landmark_householding;
mod ui_interface;

pub type FrameId = u64;

pub struct ViewData {
    pub frame_id: FrameId,
    pub camera: Camera,
    pub image: DynamicImage,
    pub depth: Vec<f32>,
    pub is_eval: bool,
}

impl ViewData {
    fn glam_img_size(&self) -> glam::UVec2 {
        glam::UVec2::new(self.image.width(), self.image.height())
    }
}

pub fn create_incremental_training_process(
    view_receiver: mpsc::Receiver<ViewData>,
    done_sender: mpsc::Sender<()>,
    config: IncrementalTrainConfig,
) -> RunningProcess {
    let (splat_tx, splat_view) = slot::channel();

    let stream = try_fn_stream(|emitter| async move {
        let mut train_ctx = IncrementalTrainer::new(
            view_receiver,
            done_sender,
            Some(splat_tx),
            Some(emitter),
            config,
        )
        .await;
        train_ctx.init_ui().await;
        train_ctx.run_train_loop().await
    });

    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
    }
}

pub fn run_incremental_training_headless(
    runtime: &Runtime,
    view_receiver: mpsc::Receiver<ViewData>,
    done_sender: mpsc::Sender<()>,
    config: IncrementalTrainConfig,
) {
    runtime.spawn(async move {
        brush_process::burn_init_setup().await;

        let mut train_ctx =
            IncrementalTrainer::new(view_receiver, done_sender, None, None, config).await;

        train_ctx.run_train_loop().await
    });
}

pub struct IncrementalTrainer {
    view_receiver: mpsc::Receiver<ViewData>,
    done_sender: mpsc::Sender<()>,

    train_views: Vec<ViewData>,
    eval_views: Vec<ViewData>,

    training_start: Option<Instant>,
    splats: Option<Splats>,
    config: IncrementalTrainConfig,

    corresponding_splats: HashMap<FrameId, (usize, usize)>,

    device: burn::tensor::Device,

    // communication with ui
    emitter: Option<TryStreamEmitter<ProcessMessage, anyhow::Error>>,
    splat_sender: Option<SlotSender<Splats>>,
    splat_sender_initialized: bool,
    up_axis: Option<glam::Vec3>,
    up_axis_factor_count: f32,
}

impl IncrementalTrainer {
    async fn new(
        view_receiver: mpsc::Receiver<ViewData>,
        done_sender: mpsc::Sender<()>,
        splat_sender: Option<SlotSender<Splats>>,
        emitter: Option<TryStreamEmitter<ProcessMessage, anyhow::Error>>,
        config: IncrementalTrainConfig,
    ) -> Self {
        let device: burn::tensor::Device = wait_for_device().await.clone().into();
        device.seed(config.seed);

        Self {
            view_receiver,
            done_sender,
            train_views: vec![],
            eval_views: vec![],
            splat_sender,
            splats: None,
            training_start: None,
            emitter,
            config,
            corresponding_splats: Default::default(),
            device,
            up_axis: None,
            splat_sender_initialized: false,
            up_axis_factor_count: 0.0,
        }
    }

    async fn run_train_loop(&mut self) -> anyhow::Result<()> {
        log::info!("Start training thread");

        let mut eval_count = 0.0;

        loop {
            let view_data = self.view_receiver.recv().await;

            let view_data = if let Some(view_data) = view_data {
                view_data
            } else {
                break;
            };

            if self.training_start.is_none() {
                self.training_start = Some(Instant::now());
            }

            let training_secs = self.training_secs();

            // TODO self.update_poses().await;

            if self.emitter.is_some() && self.train_views.len() + self.eval_views.len() < 50 {
                self.update_up_axis(&view_data.camera);
            }

            if view_data.is_eval {
                self.eval_views.push(view_data);
            } else {
                self.add_gaussians_from_view(&view_data).await;
                self.train_views.push(view_data);
            }

            if let Some(eval_every) = self.config.eval_every_sec
                && training_secs >= eval_every * eval_count
            {
                self.eval(false).await?;

                if self.config.export_on_eval {
                    self.export_checkpoint().await?;
                }

                eval_count += 1.0;
            }

            if self.done_sender.send(()).await.is_err() {
                break;
            }

            self.update_ui_dataset().await;
            self.update_train_status_ui().await;
            self.update_splat_in_ui().await;

            brush_async::yield_now().await;
        }

        log::info!("Finish training thread");

        self.eval(true).await?;

        Ok(())
    }

    async fn eval(&self, eval_all_views: bool) -> anyhow::Result<()> {
        if let Some(splats) = self.splats.clone() {
            let mut psnr_sum = 0.;
            let mut ssim_sum = 0.;

            let extra_views: &[ViewData] = if eval_all_views {
                &self.train_views
            } else {
                &[]
            };
            let num_views = self.eval_views.len() + extra_views.len();

            if num_views == 0 {
                return Ok(());
            }

            for view in self.eval_views.iter().chain(extra_views.iter()) {
                let eval_result = eval_stats(
                    splats.clone(),
                    &view.camera,
                    view.image.clone(),
                    AlphaMode::Masked,
                    &self.device,
                )
                .await?;

                psnr_sum += eval_result.psnr.clone().into_scalar_async::<f32>().await?;
                ssim_sum += eval_result.ssim.clone().into_scalar_async::<f32>().await?;
            }
            let psnr = psnr_sum / num_views as f32;
            let ssim = ssim_sum / num_views as f32;

            if let Some(emitter) = &self.emitter {
                emitter
                    .emit(ProcessMessage::TrainMessage(TrainMessage::EvalResult {
                        iter: 0,
                        avg_psnr: psnr,
                        avg_ssim: ssim,
                    }))
                    .await;
            }

            log::info!(
                "Train time: {:.2}, PSNR: {}, SSIM: {}",
                self.training_start.unwrap().elapsed().as_secs_f64(),
                psnr,
                ssim
            );
        }

        Ok(())
    }

    async fn export_checkpoint(&self) -> Result<(), anyhow::Error> {
        if let Some(splats) = self.splats.clone() {
            let secs = self.training_start.unwrap().elapsed().as_millis();
            let num_splats = splats.num_splats();

            let export_path = PathBuf::from(&self.config.export_path);
            tokio::fs::create_dir_all(&export_path)
                .await
                .with_context(|| format!("Creating export directory {}", export_path.display()))?;
            let splat_data = brush_serde::splat_to_ply(splats, self.up_axis)
                .await
                .context("Serializing splat data")?;
            tokio::fs::write(
                export_path.join(format!("{}_{}.ply", secs, num_splats)),
                splat_data,
            )
            .await
            .context(format!("Failed to export ply {export_path:?}"))?;
        }
        Ok(())
    }

    fn training_secs(&self) -> f64 {
        self.training_start
            .map_or(0.0, |it| it.elapsed().as_secs_f64())
    }
}
