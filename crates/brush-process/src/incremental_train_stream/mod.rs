use crate::{
    Emitter, RunningProcess,
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
    slot::SlotSender,
    wait_for_device,
};
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_dataset::scene::LoadImage;
use brush_dataset::{
    Dataset,
    scene::{SceneBatch, SceneView, sample_to_packed_data, view_to_sample_image},
};
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
use brush_vfs::BrushVfs;
use burn::{module::AutodiffModule, tensor::Tensor};
use image::{DynamicImage, ImageFormat, load};
use rand::rngs::StdRng;
use rand::{SeedableRng, seq::IndexedRandom};
use std::io::Cursor;
use std::time::Instant;
use std::{collections::HashMap, path::PathBuf, sync::Arc};

pub struct FrameData {
    pub poses: Vec<(i64, glam::Vec3, glam::Quat)>,
    pub landmarks_packed: Vec<f32>,
    pub image_data: Vec<u16>,
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
    mask_path: PathBuf,
) -> RunningProcess {
    let (splat_tx, splat_view) = crate::slot::channel();

    let stream = try_fn_stream(|emitter| async move {
        let mut train_ctx = IncrementalTrainContext::new(
            training_data_receiver,
            splat_tx,
            emitter,
            config,
            mask_path,
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

pub struct IncrementalTrainContext {
    training_data_receiver: tokio::sync::mpsc::UnboundedReceiver<NewTrainingData>,
    trainer: Option<SplatTrainer>,
    training_iteration: u32,
    training_start: Instant,
    total_views: usize,
    training_views: Vec<ImageWithCamera>,
    eval_views: Vec<ImageWithCamera>,
    splats: Option<Splats>,
    config: TrainStreamConfig,
    mask_image_raw: Vec<u8>,
    mask_image_png_bytes: Arc<Vec<u8>>,

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
        mask_path: PathBuf,
    ) -> Self {
        let device: burn::tensor::Device = wait_for_device().await.clone().into();
        let rng = StdRng::from_seed([config.process_config.seed as u8; 32]);
        device.seed(config.process_config.seed);

        let mask_img = image::open(&mask_path).unwrap().to_luma8();
        let mut mask_image_png_bytes: Vec<u8> = Vec::new();
        mask_img
            .write_to(
                &mut Cursor::new(&mut mask_image_png_bytes),
                ImageFormat::Png,
            )
            .unwrap();
        let mask_image_png_bytes = Arc::new(mask_image_png_bytes);
        let mask_image_raw = mask_img.into_raw();

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
            mask_image_raw,
            mask_image_png_bytes,
            device,
            rng,
            up_axis: None,
            splat_sender_initialized: false,
            up_axis_factor_count: 0.0,
            total_views: 0,
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

    async fn update_train_status_ui(&mut self) {
        let (num_splats, sh) = self
            .splats
            .as_ref()
            .map(|it| (it.num_splats(), it.sh_degree()))
            .unwrap_or((0, 0));
        self.emitter
            .emit(ProcessMessage::SplatsUpdated {
                up_axis: None,
                frame: 0,
                total_frames: 1,
                num_splats,
                sh_degree: sh,
            })
            .await;
        self.emitter
            .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                iter: self.training_iteration,
                total_elapsed: self.training_start.elapsed(),
                lod_progress: None,
            }))
            .await;
    }

    fn get_next_train_batch(&mut self) -> SceneBatch {
        let view = self
            .training_views
            .choose(&mut self.rng)
            .expect("views non-empty");
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
            log::info!(
                "Adding new training data: {} landmarks, {} poses",
                new_data.new_landmarks_packed.len(),
                new_data.views.len()
            );

            if let Some(last_image) = new_data.views.last() {
                self.emitter
                    .emit(ProcessMessage::TrainMessage(TrainMessage::NewImage {
                        image: last_image.image.clone(),
                    }))
                    .await;
            }

            self.update_up_axis(&new_data);

            if new_data.new_landmarks_packed.len() > 0 {
                self.add_new_landmarks(new_data.new_landmarks_packed);
            }

            for view in new_data.views {
                if self.total_views.is_multiple_of(20) {
                    self.eval_views.push(view);
                } else {
                    self.training_views.push(view);
                }
                self.total_views += 1;
            }

            self.update_ui_dataset().await;

            if let Some(ref s) = self.splats {
                log::info!(
                    "[brush-incremental] resetting trainer bounds ({} splats)",
                    s.num_splats()
                );
                let bounds = get_splat_bounds(s.clone(), BOUND_PERCENTILE).await;
                self.trainer = Some(SplatTrainer::new(
                    &self.config.train_config,
                    &self.device,
                    bounds,
                ));
            }
        }
    }

    fn add_new_landmarks(&mut self, new_landmarks_packed: Vec<f32>) {
        let sh_degree = self.config.model_config.sh_degree;
        let render_mode = self
            .config
            .train_config
            .render_mode
            .unwrap_or(SplatRenderMode::Default);

        let means: Vec<f32> = new_landmarks_packed;
        let new_splat = to_init_splats(
            SplatData {
                means,
                rotations: None,
                log_scales: None,
                sh_coeffs: None,
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

    fn update_up_axis(&mut self, new_data: &NewTrainingData) {
        for train_view in new_data.views.iter() {
            let rot = glam::Mat3::from_quat(train_view.camera.rotation);
            if self.up_axis.is_none() {
                self.up_axis = Some(rot.y_axis)
            } else if let Some(up_axis) = self.up_axis.as_mut() {
                *up_axis *= self.up_axis_factor_count;
                *up_axis += rot.y_axis;
                *up_axis = up_axis.normalize();
            }
            self.up_axis_factor_count += 1.;
        }
    }

    async fn update_ui_dataset(&self) {
        let views = self
            .training_views
            .iter()
            .map(|view| {
                let img_path = PathBuf::from(&format!("{}.png", view.frame_id));
                /*let mask_path = PathBuf::from("mask.png");

                let mut png_bytes: Vec<u8> = Vec::new();
                view.image
                    .write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
                    .unwrap();

                let vfs = BrushVfs::from_entries(HashMap::from([
                    (img_path.clone(), Arc::new(png_bytes)),
                    (mask_path.clone(), self.mask_image_png_bytes.clone()),
                ]));
                let load_image = LoadImage::new(
                    Arc::new(vfs),
                    img_path,
                    Some(mask_path),
                    u32::MAX,
                    Some(AlphaMode::Masked),
                );*/
                SceneView {
                    image: LoadImage::new(
                        Arc::new(BrushVfs::empty()),
                        img_path,
                        None,
                        u32::MAX,
                        None,
                    ),
                    camera: view.camera,
                }
            })
            .collect();

        self.emitter
            .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
                dataset: Dataset::from_views(views, vec![]),
            }))
            .await;
    }

    async fn update_splat_in_ui(&mut self) {
        if let Some(splats) = &self.splats {
            self.splat_sender.set(0, splats.clone());
            if !self.splat_sender_initialized {
                self.splat_sender_initialized = true;
                self.emitter.emit(ProcessMessage::DoneLoading).await;
            }
            self.emitter
                .emit(ProcessMessage::SplatsUpdated {
                    up_axis: self.up_axis,
                    frame: 0,
                    total_frames: 1,
                    num_splats: splats.num_splats(),
                    sh_degree: splats.sh_degree(),
                })
                .await;
        }
    }

    async fn init_ui(&mut self) {
        self.emitter.emit(ProcessMessage::NewProcess).await;
        self.emitter
            .emit(ProcessMessage::StartLoading {
                name: "incremental".to_owned(),
                source: brush_vfs::DataSource::Path("incremental".to_owned()),
                training: true,
                base_path: None,
            })
            .await;
        self.emitter
            .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
                config: Box::new(self.config.clone()),
            }))
            .await;
    }
}

pub struct NewTrainingData {
    pub views: Vec<ImageWithCamera>,
    // [x, y, z, x, y, z, ... ]
    pub new_landmarks_packed: Vec<f32>,
}

impl NewTrainingData {
    pub fn build_from_frame_data(
        poses_with_image: Vec<(i64, glam::Vec3, glam::Quat, Vec<u16>, Vec<u16>)>,
        new_landmarks_packed: Vec<f32>,
        unit_camera: Camera,
        img_size: glam::UVec2,
        mask_path: PathBuf,
    ) -> Self {
        let mut views = vec![];

        // TODO load mask only once
        let mask_disk = image::open(&mask_path).expect("failed to open mask image");
        if mask_disk.width() != img_size.x || mask_disk.height() != img_size.y {
            panic!("mask image dimensions do not match");
        }
        let mask_luma = mask_disk.to_luma8();

        let mask_raw: Vec<u8> = mask_luma.into_raw();

        for (frame_id, translation, quat, image_data, depth_data) in poses_with_image {
            let gray = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(
                img_size.x, img_size.y, image_data,
            )
            .unwrap();

            let pixel_count = (img_size.x * img_size.y) as usize;
            let mut rgba_bytes = Vec::with_capacity(pixel_count * 4);
            for (g, m) in gray.as_raw().iter().zip(mask_raw.iter()) {
                let g = (*g >> 8) as u8;
                rgba_bytes.extend_from_slice(&[g, g, g, *m]);
            }
            let train_rgba = image::RgbaImage::from_raw(img_size.x, img_size.y, rgba_bytes)
                .expect("rgba buffer size mismatch");
            let train_img = DynamicImage::ImageRgba8(train_rgba);

            let mut camera = unit_camera.clone();
            camera.position = translation;
            camera.rotation = quat;

            views.push(ImageWithCamera {
                frame_id,
                image: Arc::new(train_img),
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
