use brush_app::ui::app::App;
use brush_process::config::TrainStreamConfig;
use brush_process::incremental_train_stream::{
    FrameData as ProcessFrameData, FrameData, NewTrainingData, create_incremental_training_process,
};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use image::DynamicImage;
use std::collections::HashMap;
use std::fs::File;
use std::{fs, mem};
use std::path::PathBuf;
use std::process::exit;
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::time::Instant;
use crate::ffi::CameraModelId;

#[cxx::bridge]
mod ffi {
    #[namespace = "brush_cxx_bridge"]
    #[derive(Debug)]
    enum CameraModelId {
        Pinhole,
        KannalaBrandt4,
    }

    #[namespace = "brush_cxx_bridge"]
    struct StampedPose {
        frame_id: i64,
        translation: [f32; 3],
        quat: [f32; 4],
    }

    #[namespace = "brush_cxx_bridge"]
    struct FrameData {
        new_poses: Vec<StampedPose>,
        new_landmarks_packed: Vec<f32>,
        image_frame_id: i64,
        image_ptr: *const u16,
        depth_ptr: *const u16,
    }

    #[namespace = "brush_cxx_bridge"]
    extern "Rust" {
        type BrushBridge;

        fn create_brush_bridge(
            config_path: String,
            camera_params: Vec<f64>,
            camera_model_id: CameraModelId,
            img_width: u32,
            img_height: u32,
            mask_path: String,
        ) -> Box<BrushBridge>;
        fn send_fd(&self, data: FrameData);
        fn run_ui(&mut self) -> Result<()>;
    }
}

struct BrushBridge {
    config: TrainStreamConfig,

    sender: UnboundedSender<FrameData>,
    receiver: Option<UnboundedReceiver<FrameData>>,

    camera_params: Vec<f64>,
    camera_model: CameraModel,
    img_width: u32,
    img_height: u32,

    mask_raw: Option<Vec<u8>>,
}

fn create_brush_bridge(
    config_path: String,
    camera_params: Vec<f64>,
    camera_model_id: CameraModelId,
    img_width: u32,
    img_height: u32,
    mask_path: String,
) -> Box<BrushBridge> {
    let (sender, receiver) = unbounded_channel::<FrameData>();

    let config = if config_path.is_empty() {
        TrainStreamConfig::default()
    } else {
        let config_path = PathBuf::from(config_path);
        if fs::exists(&config_path).unwrap_or(false) {
            serde_json::from_reader(File::open(&config_path).expect("Error reading config")).unwrap()
        } else {
            serde_json::to_writer(File::create(&config_path).unwrap(), &TrainStreamConfig::default()).unwrap();
            TrainStreamConfig::default()
        }
    };

    let mask_raw = if mask_path.is_empty() {
        None
    } else {
        let mask_img = image::open(&mask_path).expect("failed to open mask image");
        if mask_img.width() != img_width || mask_img.height() != img_height {
            println!("mask image dimensions do not match");
            exit(1);
        }
        Some(mask_img.to_luma8().into_raw())
    };

    let camera_model = match camera_model_id {
        CameraModelId::Pinhole => CameraModel::Pinhole,
        CameraModelId::KannalaBrandt4 => CameraModel::KannalaBrandt4(KannalaBrandt4Params {
            k1: camera_params[4] as f32,
            k2: camera_params[5] as f32,
            k3: camera_params[6] as f32,
            k4: camera_params[7] as f32,
        }),
        _ => panic!("invalid camera model id"),
    };

    BrushBridge {
        config,
        sender,
        receiver: Some(receiver),
        camera_params,
        camera_model,
        img_width,
        img_height,
        mask_raw,
    }
        .into()
}

impl BrushBridge {
    fn send_fd(&self, data: ffi::FrameData) {
        let poses = data
            .new_poses
            .into_iter()
            .map(|pose| {
                let quat =
                    glam::Quat::from_xyzw(pose.quat[0], pose.quat[1], pose.quat[2], pose.quat[3])
                        .normalize();
                let translation = glam::Vec3::new(
                    pose.translation[0],
                    pose.translation[1],
                    pose.translation[2],
                );
                (pose.frame_id, translation, quat)
            })
            .collect();

        // TODO try grayscale only training (probably requires heavy modifications)
        let pixel_count = (self.img_width * self.img_height) as usize;
        let mut rgba_bytes = Vec::with_capacity(pixel_count * 4);

        if data.image_ptr.is_null() {
            println!("WTF, image ptr is null");
            exit(1);
        }

        let image_slice = unsafe { std::slice::from_raw_parts(data.image_ptr, pixel_count) };
        if let Some(mask) = &self.mask_raw {
            for i in 0..pixel_count {
                let g = (image_slice[i] >> 8) as u8;
                let m = mask[i];
                rgba_bytes.extend_from_slice(&[g, g, g, m]);
            }
        } else {
            for i in 0..pixel_count {
                let g = (image_slice[i] >> 8) as u8;
                rgba_bytes.extend_from_slice(&[g, g, g, 255]);
            }
        }

        let image = DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(self.img_width, self.img_height, rgba_bytes).unwrap(),
        );

        // TODO try to pass depth data as shared_ptr to avoid copy
        let depth_slice = unsafe { std::slice::from_raw_parts(data.depth_ptr, pixel_count) };
        let depth_data = depth_slice.to_vec();

        let frame = ProcessFrameData {
            poses,
            landmarks_packed: data.new_landmarks_packed,
            image_frame_id: data.image_frame_id,
            image,
            depth_data,
        };

        self.sender.send(frame).unwrap();
    }

    fn run_ui(&mut self) -> anyhow::Result<()> {
        let fx = self.camera_params[0];
        let fy = self.camera_params[1];
        let cx = self.camera_params[2];
        let cy = self.camera_params[3];
        let unit_camera = Camera::new(
            glam::Vec3::ZERO,
            glam::Quat::IDENTITY,
            focal_to_fov(fx, self.img_width, &self.camera_model),
            focal_to_fov(fy, self.img_height, &self.camera_model),
            glam::vec2(
                cx as f32 / self.img_width as f32,
                cy as f32 / self.img_height as f32,
            ),
            self.camera_model,
        );

        let flush_every = Duration::from_secs(1);

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to initialize tokio runtime");

        let (training_data_sender, training_data_receiver) = unbounded_channel();
        runtime.spawn(buffer_frame_data(
            mem::take(&mut self.receiver).unwrap(),
            training_data_sender,
            unit_camera,
            flush_every,
        ));

        let process = create_incremental_training_process(training_data_receiver, self.config.clone());

        runtime.block_on(async move {
            let logger = env_logger::Builder::from_default_env()
                .target(env_logger::Target::Stdout)
                .build();
            let max = logger.filter();
            brush_app::ui::log_panel::install_global_logger(Box::new(logger), max);

            let native_options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_inner_size(egui::Vec2::new(1450.0, 1200.0))
                    .with_active(true),
                wgpu_options: brush_app::ui::create_egui_options(),
                persist_window: true,
                ..Default::default()
            };

            eframe::run_native(
                "Incremental Brush",
                native_options,
                Box::new(move |cc| Ok(Box::new(App::new(cc, Some(process))))),
            )?;

            Result::<(), anyhow::Error>::Ok(())
        })?;

        Ok(())
    }
}

async fn buffer_frame_data(
    mut fd_receiver: UnboundedReceiver<FrameData>,
    training_data_sender: UnboundedSender<NewTrainingData>,
    unit_camera: Camera,
    flush_every: Duration,
) {
    let mut landmarks_packed = vec![];
    let mut poses = vec![];
    let mut images = HashMap::new();
    let mut last_flush = Instant::now();

    loop {
        let fd = fd_receiver.recv().await.unwrap();
        landmarks_packed.extend(fd.landmarks_packed);
        poses.extend(fd.poses);
        images.insert(fd.image_frame_id, (fd.image, fd.depth_data));

        if !poses.is_empty() && last_flush.elapsed() >= flush_every {
            let mut poses_with_image = vec![];
            for (frame_id, translation, quat) in mem::take(&mut poses) {
                if let Some((img, depth_data)) = images.remove(&frame_id) {
                    poses_with_image.push((frame_id, translation, quat, img, depth_data));
                }
            }

            last_flush = Instant::now();
            let td = NewTrainingData::build_from_frame_data(
                poses_with_image,
                mem::take(&mut landmarks_packed),
                unit_camera,
            );
            training_data_sender.send(td).unwrap();
        }
    }
}
