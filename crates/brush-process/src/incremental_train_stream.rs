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
use brush_render::{
    AlphaMode,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
};
use brush_serde::SplatData;
use brush_train::{
    to_init_splats,
    train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
};
use brush_vfs::BrushVfs;
use burn::{module::AutodiffModule, tensor::Tensor};
use image::{DynamicImage, ImageFormat};
use rand::{SeedableRng, seq::IndexedRandom};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use web_time::{Duration, Instant};

pub struct FrameData {
    pub poses: Vec<(i64, glam::Vec3, glam::Quat)>,
    pub landmarks: Vec<glam::Vec3>,
    /// Packed row-major grayscale u8 pixels, size = width * height.
    pub image_frame_id: i64,
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

    let mut up_axis = None;

    loop {
        while let Ok(new_data) = training_data_receiver.try_recv() {
            log::info!(
                "Adding new training data: {} landmarks, {} poses",
                new_data.new_landmarks.len(),
                new_data.training_views.len()
            );

            if up_axis.is_none() {
                let rot = glam::Mat3::from_quat(new_data.training_views[0].camera.rotation);
                up_axis = Some(rot.y_axis)
            }

            if new_data.new_landmarks.len() > 0 {
                add_new_landmarks(
                    new_data.new_landmarks,
                    &mut splats,
                    &device,
                    render_mode,
                    sh_degree,
                );
            }

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
                        up_axis,
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
                let alpha_mode = AlphaMode::Masked;
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
                let before_num = current.num_splats();
                let (new_splats, refine_stats) =
                    trainer.as_mut().unwrap().refine(iter, current).await;
                let after_num = new_splats.num_splats();
                log::info!("Refinement: {} -> {}", before_num, after_num);
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
        poses_with_image: Vec<(i64, glam::Vec3, glam::Quat, Vec<u8>)>,
        new_landmarks: Vec<glam::Vec3>,
        unit_camera: Camera,
        img_size: glam::UVec2,
        mask_path: PathBuf,
    ) -> Self {
        let mut training_views = vec![];
        let mut scene_views = vec![];

        // TODO load mask only once
        let mask_disk = image::open(&mask_path).expect("failed to open mask image");
        if mask_disk.width() != img_size.x || mask_disk.height() != img_size.y {
            panic!("mask image dimensions do not match");
        }
        let mask_luma = mask_disk.to_luma8();

        let mut mask_png_bytes: Vec<u8> = Vec::new();
        DynamicImage::ImageLuma8(mask_luma.clone())
            .write_to(
                &mut std::io::Cursor::new(&mut mask_png_bytes),
                ImageFormat::Png,
            )
            .unwrap();
        let mask_png_arc = Arc::new(mask_png_bytes);
        let mask_raw: Vec<u8> = mask_luma.into_raw();

        for (frame_id, translation, quat, image_data) in poses_with_image {
            let gray = image::GrayImage::from_raw(img_size.x, img_size.y, image_data).unwrap();

            // Trainer input: RGBA where RGB replicates the grayscale value and
            // A holds the mask, so the AlphaMode::Masked path actually zeros
            // out the loss outside the fisheye circle.
            let pixel_count = (img_size.x * img_size.y) as usize;
            let mut rgba_bytes = Vec::with_capacity(pixel_count * 4);
            for (g, m) in gray.as_raw().iter().zip(mask_raw.iter()) {
                rgba_bytes.extend_from_slice(&[*g, *g, *g, *m]);
            }
            let train_rgba = image::RgbaImage::from_raw(img_size.x, img_size.y, rgba_bytes)
                .expect("rgba buffer size mismatch");
            let train_img = DynamicImage::ImageRgba8(train_rgba);

            // scene_views path keeps a separate grayscale PNG + shared mask PNG
            // in the VFS; LoadImage will fold the mask into alpha on demand.
            let mut png_bytes: Vec<u8> = Vec::new();
            DynamicImage::ImageLuma8(gray)
                .write_to(&mut std::io::Cursor::new(&mut png_bytes), ImageFormat::Png)
                .unwrap();

            let img_path = PathBuf::from(&format!("{}.png", frame_id));
            let vfs = BrushVfs::from_entries(HashMap::from([
                (img_path.clone(), Arc::new(png_bytes)),
                (mask_path.clone(), mask_png_arc.clone()),
            ]));
            let load_image = brush_dataset::scene::LoadImage::new(
                Arc::new(vfs),
                img_path,
                Some(mask_path.clone()),
                u32::MAX,
                Some(AlphaMode::Masked),
            );

            let mut camera = unit_camera.clone();
            camera.position = translation;
            camera.rotation = quat;

            training_views.push(TrainingView {
                camera,
                image: Arc::new(train_img),
            });
            scene_views.push(SceneView {
                image: load_image,
                camera,
            });
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
