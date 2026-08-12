use crate::IncrementalTrainMessage::*;
use crate::config::IncrementalProcessConfig;
use crate::ui_interface::UpdateUiContext;
use crate::view_sampling::{ViewSampler, create_view_sampler};
use async_fn_stream::try_fn_stream;
use brush_process::{RunningProcess, slot, wait_for_device};
use brush_render::{
    AlphaMode, TextureMode, camera::Camera, gaussian_splats::Splats, render_splats,
};
use brush_train::eval::{eval_stats, ssim_map};
use brush_train::train::SplatTrainer;
use image::DynamicImage;
use parking_lot::Mutex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::num_traits::Zero;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::{mpsc, oneshot};

mod add_anchor_view;
mod all_view_training;
pub mod config;
mod export;
mod pose_update;
mod ui_interface;
mod view_sampling;

pub type FrameId = i64;

pub enum IncrementalTrainMessage {
    ComputeUnreconstructedAreaAndSSIM {
        camera: Camera,
        img_resolution: glam::UVec2,
        gt_img: DynamicImage,
        result_sender: oneshot::Sender<(f32, f32)>,
    },
    NewView(ViewData),
    NewKeyFrames(Vec<KeyFrame>),
    ExternalPoseUpdate(Vec<(FrameId, glam::Vec3, glam::Quat)>),
    Eval(oneshot::Sender<(f32, f32)>),
    Stop,
}

pub struct KeyFrame {
    pub frame_id: FrameId,
    pub camera: Camera,
    pub left_rectified: DynamicImage,
    pub right_rectified: DynamicImage,
}

pub struct ViewData {
    pub frame_id: FrameId,
    pub camera: Camera,
    pub image: DynamicImage,
    pub depth: Option<Vec<f32>>,
    pub is_anchor: bool,
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
    pub r_unrectified_rectified: glam::Quat,
}

pub fn create_incremental_training_process(
    cc: IncrementalTrainerCreationContext,
) -> RunningProcess {
    let (splat_sender, splat_view) = slot::channel();

    let stream = try_fn_stream(|emitter| async move {
        let mut trainer = IncrementalTrainer::new(
            cc.message_receiver,
            cc.gpu_mutex,
            cc.config,
            cc.r_unrectified_rectified,
            Some(UpdateUiContext::new(emitter, splat_sender)),
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

    let mut trainer = IncrementalTrainer::new(
        cc.message_receiver,
        cc.gpu_mutex,
        cc.config,
        cc.r_unrectified_rectified,
        None,
    )
    .await;

    trainer.run().await?;

    Ok(())
}

pub struct IncrementalTrainer {
    message_receiver: mpsc::UnboundedReceiver<IncrementalTrainMessage>,
    gpu_mutex: Arc<Mutex<()>>,

    train_frame_id_to_idx: HashMap<FrameId, usize>,
    train_views: Vec<ViewData>,

    trainer: Option<SplatTrainer>,
    view_sampler: Box<dyn ViewSampler>,
    rng: StdRng,

    training_start: Option<Instant>,
    splats: Option<Splats>,
    config: IncrementalProcessConfig,
    anchor_count: usize,
    export_count: f64,
    r_unrectified_rectified: glam::Quat,

    corresponding_splats: HashMap<FrameId, (usize, usize)>,

    device: burn::tensor::Device,

    up_axis: Option<glam::Vec3>,
    up_axis_set: bool,

    ui_ctx: Option<UpdateUiContext>,
}

impl IncrementalTrainer {
    async fn new(
        message_receiver: mpsc::UnboundedReceiver<IncrementalTrainMessage>,
        gpu_mutex: Arc<Mutex<()>>,
        config: IncrementalProcessConfig,
        r_unrectified_rectified: glam::Quat,
        ui_ctx: Option<UpdateUiContext>,
    ) -> Self {
        let device: burn::tensor::Device = wait_for_device().await.clone().into();
        device.seed(config.seed);

        let view_sampler = create_view_sampler(&config.train_config.view_sampling_strategy);

        let rng = StdRng::from_seed([config.seed as u8; 32]);

        Self {
            message_receiver,
            gpu_mutex,
            train_frame_id_to_idx: Default::default(),
            train_views: Default::default(),
            splats: None,
            training_start: None,
            config,
            corresponding_splats: Default::default(),
            device,
            view_sampler,
            rng,
            trainer: None,
            up_axis: None,
            up_axis_set: false,
            ui_ctx,
            anchor_count: 0,
            export_count: 1.0,
            r_unrectified_rectified,
        }
    }

    async fn run(&mut self) -> anyhow::Result<()> {
        log::info!("Start training thread");

        loop {
            match self.receive_message().await {
                Ok(message) => match message {
                    ComputeUnreconstructedAreaAndSSIM {
                        camera,
                        img_resolution,
                        gt_img,
                        result_sender,
                    } => {
                        let unreconstructed_area = self
                            .compute_unreconstructed_area(&camera, img_resolution)
                            .await;
                        let ssim = self.compute_ssim(&camera, gt_img).await;
                        result_sender.send((unreconstructed_area, ssim)).unwrap();
                    }
                    NewView(view_data) => {
                        self.update_up_axis(&view_data.camera);
                        self.add_view(view_data).await;
                    }
                    NewKeyFrames(_key_frames) => {
                        /*for kf in key_frames {
                            let unreconstructed_area = self
                                .compute_unreconstructed_area(&kf.camera, img_resolution)
                                .await;
                            let ssim = self.compute_ssim(&kf.camera, gt_img).await;

                        }*/
                    }
                    ExternalPoseUpdate(new_poses) => {
                        self.update_poses(new_poses).await;
                    }
                    Eval(result_sender) => {
                        let (psnr, ssim) = self.eval().await?;
                        result_sender.send((psnr, ssim)).unwrap();
                    }
                    Stop => break,
                },
                Err(TryRecvError::Empty) => self.train().await,
                Err(TryRecvError::Disconnected) => break,
            }

            self.update_splat_in_ui().await;
            self.update_ui_dataset().await;
            self.update_train_status_ui().await;

            self.export_all().await?;

            brush_async::yield_now().await;
        }

        log::info!("Finish training thread");

        Ok(())
    }

    async fn receive_message(&mut self) -> Result<IncrementalTrainMessage, TryRecvError> {
        let ret = if self.splats.is_none() || self.config.train_config.all_view_train_secs.is_zero()
        {
            self.message_receiver
                .recv()
                .await
                .ok_or_else(|| TryRecvError::Disconnected)
        } else {
            self.message_receiver.try_recv()
        };

        // Training time starts on first message
        if self.training_start.is_none() {
            self.training_start = Some(Instant::now());
        }

        ret
    }

    async fn add_view(&mut self, mut view_data: ViewData) {
        self.view_sampler.added_new_view(self.train_views.len());
        self.train_frame_id_to_idx
            .insert(view_data.frame_id, self.train_views.len());
        if view_data.is_anchor {
            self.anchor_count += 1;
            let start = Instant::now();
            self.add_anchor(&mut view_data).await;
            log::info!("Adding anchor view took: {:?}", start.elapsed());
        }
        self.train_views.push(view_data);
    }

    async fn compute_unreconstructed_area(
        &mut self,
        camera: &Camera,
        img_resolution: glam::UVec2,
    ) -> f32 {
        if self.splats.is_none() {
            return 1.0;
        }

        let (img, _) = render_splats(
            self.splats.clone().unwrap(),
            camera,
            img_resolution,
            glam::Vec3::ZERO,
            None,
            TextureMode::Packed,
        )
        .await;

        let floats = img
            .into_data_async()
            .await
            .unwrap()
            .into_vec::<f32>()
            .unwrap();
        let packed: &[u32] = bytemuck::cast_slice(&floats);

        let empty = packed.iter().filter(|&&p| p >> 24 <= 5).count();
        empty as f32 / packed.len() as f32
    }

    async fn compute_ssim(&mut self, camera: &Camera, gt_img: DynamicImage) -> f32 {
        if self.splats.is_none() {
            return -1.0;
        }

        ssim_map(
            self.splats.clone().unwrap(),
            camera,
            gt_img,
            AlphaMode::Masked,
            &self.device,
        )
        .await
        .mean()
        .into_scalar_async::<f32>()
        .await
        .unwrap()
    }

    async fn eval(&mut self) -> anyhow::Result<(f32, f32)> {
        if let Some(splats) = self.splats.clone() {
            let mut psnr_sum = 0.;
            let mut ssim_sum = 0.;

            let views = self.train_views.iter();
            let num_views = views.len();

            if num_views == 0 {
                return Ok((0., 0.));
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

            self.update_eval_ui(psnr, ssim).await;

            log::info!(
                "Train time: {:.2}, Eval views: {num_views}, PSNR: {}, SSIM: {}",
                self.training_duration().as_secs_f64(),
                psnr,
                ssim
            );

            Ok((psnr, ssim))
        } else {
            Ok((0., 0.))
        }
    }

    fn update_up_axis(&mut self, camera: &Camera) {
        if !self.up_axis_set {
            self.up_axis_set = true;
            let rot = glam::Mat3::from_quat(camera.rotation);
            self.up_axis = Some(rot.y_axis);
        }
    }

    fn training_duration(&self) -> Duration {
        self.training_start.unwrap().elapsed()
    }
}
