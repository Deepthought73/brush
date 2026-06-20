use crate::ffi::CameraModelId;
use brush_app::ui::app::App;
use brush_process::config::TrainStreamConfig;
use brush_process::incremental_train_stream::incremental_database::IncrementalDatabase;
use brush_process::incremental_train_stream::{
    ImageData, PoseData, create_incremental_training_process,
};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use image::DynamicImage;
use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc;
use std::{fs, mem};

#[cxx::bridge]
mod ffi {
    #[namespace = "brush_cxx_bridge"]
    #[derive(Debug)]
    enum CameraModelId {
        Pinhole,
        KannalaBrandt4,
    }

    #[namespace = "brush_cxx_bridge"]
    extern "Rust" {
        type BrushBridge;

        fn create_brush_bridge(
            config_path: String,
            camera_params: &[f64],
            camera_model_id: CameraModelId,
            img_width: u32,
            img_height: u32,
            mask_path: &str,
        ) -> Box<BrushBridge>;
        unsafe fn send_image(&self, frame_id: u64, image_ptr: *const u16, depth_ptr: *const f32);
        fn new_pose(&self, frame_id: u64, translation: [f32; 3], quat: [f32; 4]);
        fn update_pose(&self, frame_id: u64, translation: [f32; 3], quat: [f32; 4]);
        fn run_ui(&mut self) -> Result<()>;
    }
}

struct BrushBridge {
    config: TrainStreamConfig,

    image_sender: mpsc::Sender<ImageData>,
    pose_sender: mpsc::Sender<PoseData>,
    database: Option<IncrementalDatabase>,

    img_width: u32,
    img_height: u32,

    mask_raw: Option<Vec<u8>>,
}

fn create_brush_bridge(
    config_path: String,
    camera_params: &[f64],
    camera_model_id: CameraModelId,
    img_width: u32,
    img_height: u32,
    mask_path: &str,
) -> Box<BrushBridge> {
    let (image_sender, image_receiver) = mpsc::channel::<ImageData>();
    let (pose_sender, pose_receiver) = mpsc::channel::<PoseData>();

    let config = get_config(config_path);

    let mask_raw = if mask_path.is_empty() {
        None
    } else {
        let mask_img = image::open(mask_path).expect("failed to open mask image");
        assert!(
            !(mask_img.width() != img_width || mask_img.height() != img_height),
            "mask image dimensions do not match"
        );
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
    let fx = camera_params[0];
    let fy = camera_params[1];
    let cx = camera_params[2];
    let cy = camera_params[3];
    let unit_camera = Camera::new(
        glam::Vec3::ZERO,
        glam::Quat::IDENTITY,
        focal_to_fov(fx, img_width, &camera_model),
        focal_to_fov(fy, img_height, &camera_model),
        glam::vec2(cx as f32 / img_width as f32, cy as f32 / img_height as f32),
        camera_model,
    );

    let database = IncrementalDatabase::new(image_receiver, pose_receiver, unit_camera, &config);

    BrushBridge {
        config,
        image_sender,
        pose_sender,
        database: Some(database),
        img_width,
        img_height,
        mask_raw,
    }
    .into()
}

impl BrushBridge {
    unsafe fn send_image(&self, frame_id: u64, image_ptr: *const u16, depth_ptr: *const f32) {
        let pixel_count = (self.img_width * self.img_height) as usize;
        let mut rgba_bytes = Vec::with_capacity(pixel_count * 4);

        let image_slice = unsafe { std::slice::from_raw_parts(image_ptr, pixel_count) };
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
        let depth_slice = unsafe { std::slice::from_raw_parts(depth_ptr, pixel_count) };
        let depth = depth_slice.to_vec();

        self.image_sender
            .send(ImageData {
                frame_id,
                image,
                depth,
            })
            .unwrap();
    }

    fn new_pose(&self, frame_id: u64, translation: [f32; 3], quat: [f32; 4]) {
        let quat = glam::Quat::from_xyzw(quat[0], quat[1], quat[2], quat[3]).normalize();
        let translation = glam::Vec3::new(translation[0], translation[1], translation[2]);
        self.pose_sender
            .send(PoseData {
                frame_id,
                translation,
                quat,
            })
            .unwrap();
    }

    fn update_pose(&self, frame_id: u64, translation: [f32; 3], quat: [f32; 4]) {
        self.new_pose(frame_id, translation, quat);
    }

    fn run_ui(&mut self) -> anyhow::Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to initialize tokio runtime");

        let process = create_incremental_training_process(
            mem::take(&mut self.database).unwrap(),
            self.config.clone(),
        );

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

fn get_config(config_path: String) -> TrainStreamConfig {
    if config_path.is_empty() {
        TrainStreamConfig::default()
    } else {
        let config_path = PathBuf::from(config_path);
        if fs::exists(&config_path).unwrap_or(false) {
            serde_json::from_reader(File::open(&config_path).expect("Error reading config"))
                .unwrap()
        } else {
            serde_json::to_writer(
                File::create(&config_path).unwrap(),
                &TrainStreamConfig::default(),
            )
            .unwrap();
            TrainStreamConfig::default()
        }
    }
}
