use crate::{
    Emitter, RunningProcess,
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
    slot::SlotSender,
    wait_for_device,
};
use async_fn_stream::try_fn_stream;
use brush_dataset::{
    Dataset,
    scene::{SceneBatch, SceneView, sample_to_packed_data, view_to_sample_image},
};
use brush_render::kernels::camera_model::pinhole::PinholeParams;
use brush_render::{
    AlphaMode,
    camera::{Camera, focal_to_fov},
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::camera_model::CameraModel,
};
use brush_serde::SplatData;
use brush_train::{
    to_init_splats,
    train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
};
use brush_vfs::BrushVfs;
use burn::{module::AutodiffModule, tensor::Tensor};
use image::DynamicImage;
use rand::{SeedableRng, seq::IndexedRandom};
use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::Arc,
};
use web_time::{Duration, Instant};

pub struct FrameData {
    pub t_ns: i64,
    pub translation: glam::Vec3,
    pub quat: glam::Quat,
    pub landmarks: Vec<glam::Vec3>,
    /// Packed row-major grayscale u8 pixels, size = width * height.
    pub image_data: Vec<u8>,
}

pub struct TrainingView {
    pub camera: Camera,
    pub image: Arc<DynamicImage>,
}

pub fn create_incremental_training_process(
    training_data_receiver: tokio::sync::mpsc::UnboundedReceiver<NewTrainingData>,
    train_stream_config: TrainStreamConfig,
) -> RunningProcess {
    let (splat_tx, splat_view) = crate::slot::channel();
    let stream = try_fn_stream(|emitter| async move {
        incremental_train_stream(
            training_data_receiver,
            train_stream_config,
            &emitter,
            splat_tx,
        )
        .await
    });
    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
    }
}

pub async fn incremental_train_stream(
    mut training_data_receiver: tokio::sync::mpsc::UnboundedReceiver<NewTrainingData>,
    train_stream_config: TrainStreamConfig,
    emitter: &Emitter,
    slot: SlotSender<Splats>,
) -> anyhow::Result<()> {
    log::info!("[brush-incremental] starting");

    emitter.emit(ProcessMessage::NewProcess).await;
    emitter
        .emit(ProcessMessage::StartLoading {
            name: "incremental".to_owned(),
            source: brush_vfs::DataSource::Path("incremental".to_owned()),
            training: true,
            base_path: None,
        })
        .await;
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
            config: Box::new(train_stream_config.clone()),
        }))
        .await;
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
            dataset: Dataset::empty(),
        }))
        .await;

    let wgpu_device = wait_for_device().await;
    let device: burn::tensor::Device = wgpu_device.clone().into();
    let seed = train_stream_config.process_config.seed;
    device.seed(seed);
    let mut rng = rand::rngs::StdRng::from_seed([seed as u8; 32]);

    let render_mode = train_stream_config
        .train_config
        .render_mode
        .unwrap_or(SplatRenderMode::Default);
    let sh_degree = train_stream_config.model_config.sh_degree;

    let mut views: Vec<TrainingView> = vec![];
    let mut scene_views = vec![];
    let mut splats: Option<Splats> = None;
    let mut trainer: Option<SplatTrainer> = None;
    let mut iter: u32 = 0;
    let mut train_duration = Duration::from_secs(0);
    let mut slot_initialized = false;

    loop {
        while let Ok(new_data) = training_data_receiver.try_recv() {
            add_new_landmarks(
                new_data.new_landmarks,
                &mut splats,
                &device,
                render_mode,
                sh_degree,
            );

            scene_views.extend(new_data.scene_views);
            views.extend(new_data.training_views);

            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
                    dataset: Dataset::from_views(scene_views.clone(), vec![]),
                }))
                .await;

            if let Some(ref s) = splats {
                log::info!(
                    "[brush-incremental] resetting trainer bounds ({} splats)",
                    s.num_splats()
                );
                let bounds = get_splat_bounds(s.clone(), BOUND_PERCENTILE).await;
                trainer = Some(SplatTrainer::new(
                    &train_stream_config.train_config,
                    &device,
                    bounds,
                ));
                slot.set(0, s.clone());
                if !slot_initialized {
                    slot_initialized = true;
                    emitter.emit(ProcessMessage::DoneLoading).await;
                }
                emitter
                    .emit(ProcessMessage::SplatsUpdated {
                        up_axis: None,
                        frame: 0,
                        total_frames: 1,
                        num_splats: s.num_splats(),
                        sh_degree: s.sh_degree(),
                    })
                    .await;
            }
        }

        // Nothing to train on yet.
        if trainer.is_none() || splats.is_none() {
            brush_async::yield_now().await;
            continue;
        }

        // Run one training step.
        {
            let step_start = Instant::now();

            // TODO try importance sampling depending on how often an image was trained on
            let batch = {
                let view = views.choose(&mut rng).expect("views non-empty");
                let alpha_mode = AlphaMode::Transparent;
                let img = view_to_sample_image(view.image.as_ref().clone(), alpha_mode);
                let (img_packed, has_alpha) = sample_to_packed_data(img);
                SceneBatch {
                    img_packed,
                    has_alpha,
                    alpha_mode,
                    camera: view.camera,
                }
            };

            let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(
                splats.as_ref().unwrap().clone(),
            );
            let (new_diff, _stats) = trainer.as_mut().unwrap().step(batch, diff_splats).await;
            splats = Some(new_diff.valid());

            train_duration += step_start.elapsed();
            iter += 1;

            slot.set(0, splats.as_ref().unwrap().clone());

            let refine_every = train_stream_config.train_config.refine_every;
            if iter > 0 && iter % refine_every == 0 {
                let current = splats.take().unwrap();
                let (new_splats, refine_stats) =
                    trainer.as_mut().unwrap().refine(iter, current).await;
                slot.set(0, new_splats.clone());
                emitter
                    .emit(ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                        cur_splat_count: refine_stats.total_splats,
                        iter,
                    }))
                    .await;
                splats = Some(new_splats);
            }

            const UPDATE_EVERY: u32 = 5;
            if iter % UPDATE_EVERY == 0 {
                let num_splats = splats.as_ref().unwrap().num_splats();
                let sh = splats.as_ref().unwrap().sh_degree();
                emitter
                    .emit(ProcessMessage::SplatsUpdated {
                        up_axis: None,
                        frame: 0,
                        total_frames: 1,
                        num_splats,
                        sh_degree: sh,
                    })
                    .await;
                emitter
                    .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                        iter,
                        total_elapsed: train_duration,
                        lod_progress: None,
                    }))
                    .await;
            }
        }

        brush_async::yield_now().await;
    }
}

fn add_new_landmarks(
    new_landmarks: Vec<glam::Vec3>,
    splats: &mut Option<Splats>,
    device: &burn::tensor::Device,
    render_mode: SplatRenderMode,
    sh_degree: u32,
) {
    let means: Vec<f32> = new_landmarks.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    let new_splat = to_init_splats(
        SplatData {
            means,
            rotations: None,
            log_scales: None,
            sh_coeffs: None,
            raw_opacities: None,
        },
        render_mode,
        device,
    )
    .with_sh_degree(sh_degree);

    *splats = Some(match splats.take() {
        None => new_splat,
        Some(existing) => concat_splats(existing, new_splat, render_mode),
    });
}

pub struct NewTrainingData {
    pub training_views: Vec<TrainingView>,
    pub scene_views: Vec<SceneView>,
    pub new_landmarks: Vec<glam::Vec3>,
}

impl NewTrainingData {
    pub fn build_from_frame_data(
        fds: Vec<FrameData>,
        unit_camera: Camera,
        img_size: glam::UVec2,
    ) -> Self {
        let mut training_views = vec![];
        let mut scene_views = vec![];
        let mut new_landmarks = vec![];

        for frame in fds {
            let gray =
                image::GrayImage::from_raw(img_size.x, img_size.y, frame.image_data).unwrap();
            let img = DynamicImage::ImageLuma8(gray);

            let mut png_bytes: Vec<u8> = Vec::new();
            img.write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

            let img_path = PathBuf::from(&format!("{}.png", frame.t_ns));
            let vfs =
                BrushVfs::from_entries(HashMap::from([(img_path.clone(), Arc::new(png_bytes))]));
            let load_image = brush_dataset::scene::LoadImage::new(
                Arc::new(vfs),
                img_path,
                None, // TODO use mask
                u32::MAX,
                Some(AlphaMode::Transparent),
            );

            let mut camera = unit_camera.clone();
            camera.position = frame.translation;
            camera.rotation = frame.quat;

            training_views.push(TrainingView {
                camera,
                image: Arc::new(img),
            });
            scene_views.push(SceneView {
                image: load_image,
                camera,
            });

            if !frame.landmarks.is_empty() {
                new_landmarks.extend(frame.landmarks);
            }
        }

        Self {
            training_views,
            scene_views,
            new_landmarks,
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
