use crate::incremental_train_stream::view_sampling::{create_view_sampler, RandomViewSampler, ViewSampler};
use crate::{
    RunningProcess,
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
    slot::SlotSender,
    wait_for_device,
};
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_dataset::scene::{SceneBatch, sample_to_packed_data, view_to_sample_image};
use brush_render::shaders::SH_C0;
use brush_render::{
    AlphaMode,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
};
use brush_serde::SplatData;
use brush_train::eval::eval_stats;
use brush_train::{
    to_init_splats,
    train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
};
use burn::{module::AutodiffModule, tensor::Tensor};
use image::DynamicImage;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Arc;
use std::time::Instant;

pub mod config;
mod ui_interface;
mod view_sampling;

pub struct FrameData {
    pub poses: Vec<(i64, glam::Vec3, glam::Quat)>,
    pub landmarks_packed: Vec<f32>,
    pub image: DynamicImage,
    pub depth_data: Vec<u16>,
    /// Packed row-major grayscale u8 pixels, size = width * height.
    pub image_frame_id: i64,
}

pub struct ImageWithCamera {
    pub frame_id: i64,
    pub image: Arc<DynamicImage>,
    pub depth_data: Vec<u16>,
    pub camera: Camera,
}

pub fn create_incremental_training_process(
    training_data_receiver: tokio::sync::mpsc::UnboundedReceiver<NewTrainingData>,
    config: TrainStreamConfig,
) -> RunningProcess {
    let (splat_tx, splat_view) = crate::slot::channel();

    let stream = try_fn_stream(|emitter| async move {
        let mut train_ctx =
            IncrementalTrainContext::new(training_data_receiver, splat_tx, emitter, config).await;
        train_ctx.init_ui().await;
        train_ctx.run_train_loop().await
    });

    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
    }
}

pub struct IncrementalTrainContext {
    training_data_receiver: tokio::sync::mpsc::UnboundedReceiver<NewTrainingData>,
    trainer: Option<SplatTrainer>,
    training_iteration: u32,
    training_start: Instant,
    view_sampler: Box<dyn ViewSampler>,
    total_views: usize,
    training_views: Vec<ImageWithCamera>,
    eval_views: Vec<ImageWithCamera>,
    splats: Option<Splats>,
    config: TrainStreamConfig,

    device: burn::tensor::Device,
    rng: StdRng,

    // communication with ui
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
    splat_sender: SlotSender<Splats>,
    splat_sender_initialized: bool,
    up_axis: Option<glam::Vec3>,
    up_axis_factor_count: f32,
}

impl IncrementalTrainContext {
    async fn new(
        training_data_receiver: tokio::sync::mpsc::UnboundedReceiver<NewTrainingData>,
        splat_sender: SlotSender<Splats>,
        emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
        config: TrainStreamConfig,
    ) -> Self {
        let device: burn::tensor::Device = wait_for_device().await.clone().into();
        let rng = StdRng::from_seed([config.process_config.seed as u8; 32]);
        device.seed(config.process_config.seed);

        let view_sampler = create_view_sampler(&config.incremental_train_config.view_sampling_strategy);

        Self {
            training_data_receiver,
            splat_sender,
            training_views: vec![],
            eval_views: vec![],
            splats: None,
            trainer: None,
            training_iteration: 0,
            training_start: Instant::now(),
            emitter,
            config,
            device,
            rng,
            up_axis: None,
            splat_sender_initialized: false,
            up_axis_factor_count: 0.0,
            total_views: 0,
            view_sampler,
        }
    }

    async fn run_train_loop(&mut self) -> anyhow::Result<()> {
        log::info!("Start training thread");

        loop {
            self.receive_new_training_data().await;

            if self.trainer.is_some() && self.splats.is_some() && !self.training_views.is_empty() {
                self.train_step().await;
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

        let batch = self.get_next_train_batch();

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

        let refine_every = self.config.train_config.refine_every;
        if self.training_iteration > 0 && self.training_iteration % refine_every == 0 {
            let current = self.splats.take().unwrap();
            let before_num = current.num_splats();
            let (new_splats, refine_stats) = self
                .trainer
                .as_mut()
                .unwrap()
                .refine(self.training_iteration, current)
                .await;
            let after_num = new_splats.num_splats();
            log::info!("Refinement: {} -> {}", before_num, after_num);
            self.emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                    cur_splat_count: refine_stats.total_splats,
                    iter: self.training_iteration,
                }))
                .await;
            self.splats = Some(new_splats);
        }
    }

    async fn eval_step(&mut self) -> anyhow::Result<()> {
        if let Some(splats) = self.splats.clone() {
            let mut psnr_sum = 0.;
            let mut ssim_sum = 0.;
            for view in self.eval_views.iter() {
                let eval_result = eval_stats(
                    splats.clone(),
                    &view.camera,
                    (*view.image).clone(),
                    AlphaMode::Masked,
                    &self.device,
                )
                .await?;

                psnr_sum += eval_result.psnr.clone().into_scalar_async::<f32>().await?;
                ssim_sum += eval_result.ssim.clone().into_scalar_async::<f32>().await?;
            }
            let psnr = psnr_sum / self.eval_views.len() as f32;
            let ssim = ssim_sum / self.eval_views.len() as f32;
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

    fn get_next_train_batch(&mut self) -> SceneBatch {
        let idx = self.view_sampler.sample(&mut self.rng);
        let view = &self.training_views[idx];
        let alpha_mode = AlphaMode::Masked;
        let img = view_to_sample_image(view.image.as_ref().clone(), alpha_mode);
        let (img_packed, has_alpha) = sample_to_packed_data(img);
        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode,
            camera: view.camera,
        }
    }

    async fn receive_new_training_data(&mut self) {
        while let Ok(new_data) = self.training_data_receiver.try_recv() {
            log::info!("Received new {} new views", new_data.views.len());

            self.update_last_images(new_data.views.last()).await;

            if self.total_views < 50 {
                self.update_up_axis(&new_data);
            }

            for view in new_data.views {
                if let Some(n) = self.config.load_config.eval_split_every
                    && self.total_views.is_multiple_of(n)
                {
                    self.eval_views.push(view);
                } else {
                    self.add_new_landmarks_by_depth(&view);
                    self.training_views.push(view);
                    self.view_sampler.added_new_item();
                }
                self.total_views += 1;
            }

            self.update_ui_dataset().await;

            if let Some(ref s) = self.splats {
                let bounds = get_splat_bounds(s.clone(), BOUND_PERCENTILE).await;
                self.trainer = Some(SplatTrainer::new(
                    &self.config.train_config,
                    &self.device,
                    bounds,
                ));
            }
        }
    }

    fn add_new_landmarks_by_depth(&mut self, view: &ImageWithCamera) {
        let mut means = vec![];
        let mut sh_coeffs = vec![];
        let mut log_scales = vec![];

        let w = view.image.width() as usize;
        let h = view.image.height() as usize;
        let img_size = glam::UVec2::new(view.image.width(), view.image.height());

        let raw_img = view.image.as_rgba8().unwrap().as_raw();

        let stride = 10;
        let focal = view.camera.focal(img_size);

        for v in (0..h).step_by(stride) {
            for u in (0..w).step_by(stride) {
                let idx = v * w + u;
                let d = view.depth_data[idx];
                if d == 0 {
                    continue;
                }

                // TODO depth is in mm, maybe preprocess somewhere else, if the unit changes
                // TODO or: provide unit of depth in config
                let d = d as f32 / 1000.;

                let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;

                let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);

                let pos_cam = view.camera.unproject(uv, d, img_size);
                let pos_world = view.camera.transform(pos_cam);
                means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
                sh_coeffs.extend_from_slice(&[color, color, color]);

                // TODO: make this less magic
                let scale = 4. * d / focal.x;
                let log_s = scale.ln();
                log_scales.extend_from_slice(&[log_s, log_s, log_s]);
            }
        }

        self.add_new_landmarks_by_means(means, Some(sh_coeffs), Some(log_scales));
    }

    fn add_new_landmarks_by_means(
        &mut self,
        means: Vec<f32>,
        sh_coeffs: Option<Vec<f32>>,
        log_scales: Option<Vec<f32>>,
    ) {
        let sh_degree = self.config.model_config.sh_degree;
        let render_mode = self
            .config
            .train_config
            .render_mode
            .unwrap_or(SplatRenderMode::Default);

        let new_splat = to_init_splats(
            SplatData {
                means,
                rotations: None,
                log_scales,
                sh_coeffs,
                raw_opacities: None,
            },
            render_mode,
            &self.device,
        )
        .with_sh_degree(sh_degree);

        self.splats = Some(match self.splats.take() {
            None => new_splat,
            Some(existing) => concat_splats(existing, new_splat, render_mode),
        });
    }
}

pub struct NewTrainingData {
    pub views: Vec<ImageWithCamera>,
    // [x, y, z, x, y, z, ... ]
    pub new_landmarks_packed: Vec<f32>,
}

impl NewTrainingData {
    pub fn build_from_frame_data(
        poses_with_image: Vec<(i64, glam::Vec3, glam::Quat, DynamicImage, Vec<u16>)>,
        new_landmarks_packed: Vec<f32>,
        unit_camera: Camera,
    ) -> Self {
        let mut views = vec![];

        for (frame_id, translation, quat, image, depth_data) in poses_with_image {
            let mut camera = unit_camera.clone();
            camera.position = translation;
            camera.rotation = quat;

            views.push(ImageWithCamera {
                frame_id,
                image: Arc::new(image),
                depth_data,
                camera,
            });
        }

        Self {
            views,
            new_landmarks_packed,
        }
    }
}

fn concat_splats(a: Splats, b: Splats, mode: SplatRenderMode) -> Splats {
    let means = Tensor::cat(vec![a.means(), b.means()], 0);
    let rotations = Tensor::cat(vec![a.rotations(), b.rotations()], 0);
    let log_scales = Tensor::cat(vec![a.log_scales(), b.log_scales()], 0);
    let sh_coeffs = Tensor::cat(vec![a.sh_coeffs.val(), b.sh_coeffs.val()], 0);
    let opacities = Tensor::cat(vec![a.raw_opacities.val(), b.raw_opacities.val()], 0);
    Splats::from_tensor_data(means, rotations, log_scales, sh_coeffs, opacities, mode)
}
