use crate::IncrementalTrainMessage::{ExternalPoseUpdate, NewView};
use crate::config::IncrementalProcessConfig;
use crate::view_sampling::{ViewSampler, create_view_sampler};
use IncrementalTrainMessage::ContinueTrain;
use anyhow::Context;
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::slot::SlotSender;
use brush_process::{RunningProcess, slot, wait_for_device};
use brush_render::{AlphaMode, camera::Camera, gaussian_splats::Splats};
use brush_train::eval::eval_stats;
use image::DynamicImage;
use parking_lot::Mutex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

mod add_host_view;
mod all_view_training;
pub mod config;
mod pose_update;
mod ui_interface;
mod view_sampling;

pub type FrameId = i64;

pub enum IncrementalTrainMessage {
    NewView(ViewData),
    ExternalPoseUpdate(Vec<(FrameId, glam::Vec3, glam::Quat)>),
    ContinueTrain,
}

pub struct ViewData {
    pub frame_id: FrameId,
    pub camera: Camera,
    pub image: DynamicImage,
    pub depth: Option<Vec<f32>>,
    pub is_eval: bool,
    pub is_host_frame: bool,
}

impl ViewData {
    fn glam_img_size(&self) -> glam::UVec2 {
        glam::UVec2::new(self.image.width(), self.image.height())
    }
}

pub struct IncrementalTrainerCreationContext {
    pub message_receiver: mpsc::UnboundedReceiver<IncrementalTrainMessage>,
    pub gpu_mutex: Arc<Mutex<()>>,
    pub config: IncrementalProcessConfig,
}

pub fn create_incremental_training_process(
    cc: IncrementalTrainerCreationContext,
) -> RunningProcess {
    let (splat_tx, splat_view) = slot::channel();

    let stream = try_fn_stream(|emitter| async move {
        let mut trainer = IncrementalTrainer::new(
            cc.message_receiver,
            cc.gpu_mutex,
            Some(splat_tx),
            Some(emitter),
            cc.config,
        )
        .await;
        trainer.init_ui().await;
        trainer.run().await
    });

    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
    }
}

pub async fn run_incremental_training_headless(
    cc: IncrementalTrainerCreationContext,
) -> anyhow::Result<()> {
    brush_process::burn_init_setup().await;

    let mut trainer =
        IncrementalTrainer::new(cc.message_receiver, cc.gpu_mutex, None, None, cc.config).await;

    trainer.run().await?;

    Ok(())
}

pub struct IncrementalTrainer {
    message_receiver: mpsc::UnboundedReceiver<IncrementalTrainMessage>,
    gpu_mutex: Arc<Mutex<()>>,

    train_frame_id_to_idx: HashMap<FrameId, usize>,
    eval_frame_id_to_idx: HashMap<FrameId, usize>,
    train_views: Vec<ViewData>,
    eval_views: Vec<ViewData>,

    view_sampler: Box<dyn ViewSampler>,
    rng: StdRng,

    training_start: Option<Instant>,
    splats: Option<Splats>,
    config: IncrementalProcessConfig,

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
        message_receiver: mpsc::UnboundedReceiver<IncrementalTrainMessage>,
        gpu_mutex: Arc<Mutex<()>>,
        splat_sender: Option<SlotSender<Splats>>,
        emitter: Option<TryStreamEmitter<ProcessMessage, anyhow::Error>>,
        config: IncrementalProcessConfig,
    ) -> Self {
        let device: burn::tensor::Device = wait_for_device().await.clone().into();
        device.seed(config.seed);

        let view_sampler = create_view_sampler(&config.train_config.view_sampling_strategy);

        let rng = StdRng::from_seed([config.seed as u8; 32]);

        Self {
            message_receiver,
            gpu_mutex,
            train_frame_id_to_idx: Default::default(),
            eval_frame_id_to_idx: Default::default(),
            train_views: Default::default(),
            eval_views: Default::default(),
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
            view_sampler,
            rng,
        }
    }

    async fn run(&mut self) -> anyhow::Result<()> {
        log::info!("Start training thread");

        let mut eval_count = 0.0;

        loop {
            match self.message_receiver.recv().await {
                Some(message) => match message {
                    NewView(view_data) => {
                        self.update_up_axis(&view_data.camera);
                        {
                            let _guard = self.gpu_mutex.lock_arc();
                            self.add_view(view_data).await;
                        }
                    }
                    ExternalPoseUpdate(new_poses) => {
                        self.update_poses(new_poses).await;
                        continue;
                    }
                    ContinueTrain => {}
                },
                None => break,
            }

            let training_secs = self.training_secs();

            {
                let _guard = self.gpu_mutex.lock_arc();
                self.train().await;
            }

            if let Some(eval_every) = self.config.eval_every_sec
                && training_secs >= eval_every * eval_count
            {
                self.eval(self.config.eval_train_views).await?;

                if self.config.export_on_eval {
                    self.export_checkpoint().await?;
                }

                eval_count += 1.0;
            }

            self.update_ui_dataset().await;
            self.update_train_status_ui().await;
            self.update_splat_in_ui().await;

            brush_async::yield_now().await;
        }

        log::info!("Finish training thread");

        Ok(())
    }

    async fn add_view(&mut self, view_data: ViewData) {
        if view_data.is_eval {
            self.eval_frame_id_to_idx
                .insert(view_data.frame_id, self.eval_views.len());
            self.eval_views.push(view_data);
        } else {
            self.view_sampler.added_new_view(self.train_views.len());
            self.train_frame_id_to_idx
                .insert(view_data.frame_id, self.train_views.len());
            if view_data.is_host_frame {
                self.add_host_view(&view_data).await;
            }
            self.train_views.push(view_data);
        }
    }

    async fn eval(&self, eval_train: bool) -> anyhow::Result<()> {
        if let Some(splats) = self.splats.clone() {
            let mut psnr_sum = 0.;
            let mut ssim_sum = 0.;

            let views = if eval_train {
                self.train_views.iter()
            } else {
                self.eval_views.iter()
            };
            let num_views = views.len();

            if num_views == 0 {
                return Ok(());
            }

            for view in views {
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
                "Train time: {:.2}, Eval views: {num_views}, PSNR: {}, SSIM: {}",
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

    fn training_secs(&mut self) -> f64 {
        let training_start = self.training_start.get_or_insert(Instant::now());
        training_start.elapsed().as_secs_f64()
    }
}
