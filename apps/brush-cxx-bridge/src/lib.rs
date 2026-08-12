use crate::ffi::{DepthProviderBridge, EvalResult, ImageBridge, RectificationBridge, StampedPose};
use anyhow::{Context, ensure};
use brush_app::ui::app::App;
use brush_incremental::IncrementalTrainMessage::*;
use brush_incremental::config::IncrementalProcessConfig;
use brush_incremental::{
    IncrementalTrainMessage, IncrementalTrainerCreationContext, ViewData,
    create_incremental_training_process, run_incremental_training_headless,
};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use cxx::SharedPtr;
use image::DynamicImage;
use parking_lot::Mutex;
use std::fs;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot};

mod gpu_mutex;
use gpu_mutex::{GpuMutex, GpuMutexGuard, new_gpu_mutex};

#[cxx::bridge(namespace = "brush_cxx_bridge")]
mod ffi {
    struct StampedPose {
        frame_id: i64,
        t: [f32; 3],
        q: [f32; 4],
    }

    struct EvalResult {
        psnr: f32,
        ssim: f32,
    }

    extern "Rust" {
        type BrushBridge;

        fn new_brush_bridge(
            config_path: String,
            camera_params: &[f64],
            img_width: u32,
            img_height: u32,
            mask_path: &str,
            gpu_mutex: Box<GpuMutex>,
            r_unrectified_rectified: [f32; 4],
            rectification: SharedPtr<RectificationBridge>,
            depth_provider: SharedPtr<DepthProviderBridge>,
        ) -> Result<Box<BrushBridge>>;

        fn add_raw_frame(
            &mut self,
            frame_id: i64,
            left: SharedPtr<ImageBridge>,
            right: SharedPtr<ImageBridge>,
            translation: [f32; 3],
            quat: [f32; 4],
        );

        fn eval(&self) -> EvalResult;

        fn update_poses(&mut self, new_poses: Vec<StampedPose>);

        fn run(&mut self) -> Result<()>;

        fn run_headless(&mut self) -> Result<()>;

        fn stop(&mut self);
    }

    extern "Rust" {
        type GpuMutex;
        type GpuMutexGuard;

        fn new_gpu_mutex() -> Box<GpuMutex>;
        fn lock(self: &GpuMutex) -> Box<GpuMutexGuard>;
        fn clone(self: &GpuMutex) -> Box<GpuMutex>;
    }

    unsafe extern "C++" {
        include!("basalt/rt_splatter/image_bridge.h");

        type ImageBridge;

        fn width(self: &ImageBridge) -> u32;
        fn height(self: &ImageBridge) -> u32;
        fn data(self: &ImageBridge) -> &[u16];
    }

    unsafe extern "C++" {
        include!("basalt/rt_splatter/rectification_bridge.h");

        type RectificationBridge;

        fn rectify_left(self: &RectificationBridge, raw: &ImageBridge) -> SharedPtr<ImageBridge>;
        fn rectify_right(self: &RectificationBridge, raw: &ImageBridge) -> SharedPtr<ImageBridge>;
    }

    unsafe extern "C++" {
        include!("basalt/rt_splatter/depth_provider_bridge.h");

        type DepthProviderBridge;

        fn get_depth(
            self: &DepthProviderBridge,
            left_rectified: &ImageBridge,
            right_rectified: &ImageBridge,
        ) -> Vec<f32>;
    }
}

struct BrushBridge {
    cc: Option<IncrementalTrainerCreationContext>,

    message_sender: mpsc::UnboundedSender<IncrementalTrainMessage>,

    unit_camera: Camera,
    img_width: u32,
    img_height: u32,

    mask_raw: Option<Vec<u8>>,

    runtime: Runtime,

    gpu_mutex: Arc<Mutex<()>>,
    rectification: SharedPtr<RectificationBridge>,
    depth_provider: SharedPtr<DepthProviderBridge>,

    unreconstructed_area_threshold: f32,
    max_ssim_new_host: f32,
}

fn new_brush_bridge(
    config_path: String,
    camera_params: &[f64],
    img_width: u32,
    img_height: u32,
    mask_path: &str,
    gpu_mutex: Box<GpuMutex>,
    r_unrectified_rectified: [f32; 4],
    rectification: SharedPtr<RectificationBridge>,
    depth_provider: SharedPtr<DepthProviderBridge>,
) -> anyhow::Result<Box<BrushBridge>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize tokio runtime");
    let (message_sender, message_receiver) = mpsc::unbounded_channel::<IncrementalTrainMessage>();
    let config = get_config(config_path)?;
    let mask_raw = load_mask(mask_path, img_width, img_height)?;
    let unit_camera = build_unit_camera(camera_params, img_width, img_height);
    let gpu_mutex_arc = gpu_mutex.arc();
    let unreconstructed_area_threshold = config.train_config.unreconstructed_area_threshold;
    let max_ssim_new_host = config.train_config.max_ssim_new_anchor;

    Ok(BrushBridge {
        cc: Some(IncrementalTrainerCreationContext {
            message_receiver,
            gpu_mutex: gpu_mutex_arc.clone(),
            config,
            r_unrectified_rectified: glam::Quat::from_array(r_unrectified_rectified),
        }),
        message_sender,
        unit_camera,
        img_width,
        img_height,
        mask_raw,
        runtime,
        gpu_mutex: gpu_mutex_arc,
        rectification,
        depth_provider,
        unreconstructed_area_threshold,
        max_ssim_new_host,
    }
    .into())
}

impl BrushBridge {
    fn add_raw_frame(
        &mut self,
        frame_id: i64,
        left: SharedPtr<ImageBridge>,
        right: SharedPtr<ImageBridge>,
        translation: [f32; 3],
        quat: [f32; 4],
    ) {
        let left_rect = self.rectification.rectify_left(&left);
        let right_rect = self.rectification.rectify_right(&right);

        let gt_img = self.copy_into_rgba_image(&left_rect);

        let (result_sender, result_receiver) = oneshot::channel();
        self.message_sender
            .send(ComputeUnreconstructedAreaAndSSIM {
                camera: self.build_camera(translation, quat),
                img_resolution: glam::UVec2::new(self.img_width, self.img_height),
                gt_img: gt_img.clone(),
                result_sender,
            })
            .unwrap();
        let (unreconstructed_area, ssim) = self.runtime.block_on(result_receiver).unwrap();

        let is_anchor = unreconstructed_area > self.unreconstructed_area_threshold
            || ssim < self.max_ssim_new_host;

        let depth = if is_anchor {
            let _guard = self.gpu_mutex.lock_arc();
            Some(self.depth_provider.get_depth(&left_rect, &right_rect))
        } else {
            None
        };

        self.message_sender
            .send(NewView(ViewData {
                frame_id,
                camera: self.build_camera(translation, quat),
                image: gt_img,
                depth,
                is_anchor,
            }))
            .unwrap();
    }

    fn eval(&self) -> EvalResult {
        let (tx, rx) = oneshot::channel();
        self.message_sender.send(Eval(tx)).unwrap();
        let (psnr, ssim) = self.runtime.block_on(rx).unwrap();
        EvalResult { psnr, ssim }
    }

    fn update_poses(&mut self, new_poses: Vec<StampedPose>) {
        let new_poses = new_poses
            .into_iter()
            .map(|sp| {
                (
                    sp.frame_id,
                    glam::Vec3::new(sp.t[0], sp.t[1], sp.t[2]),
                    glam::Quat::from_xyzw(sp.q[0], sp.q[1], sp.q[2], sp.q[3]).normalize(),
                )
            })
            .collect::<Vec<_>>();
        self.message_sender
            .send(ExternalPoseUpdate(new_poses))
            .unwrap();
    }

    fn run(&mut self) -> anyhow::Result<()> {
        let process = create_incremental_training_process(self.cc.take().unwrap());

        self.runtime.block_on(async {
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
            )
        })?;

        Ok(())
    }

    fn run_headless(&mut self) -> anyhow::Result<()> {
        env_logger::Builder::from_default_env()
            .target(env_logger::Target::Stdout)
            .try_init()?;

        self.runtime
            .spawn(run_incremental_training_headless(self.cc.take().unwrap()));

        Ok(())
    }

    fn stop(&mut self) {
        self.message_sender.send(Stop).unwrap();
    }

    fn copy_into_rgba_image(&self, image: &ImageBridge) -> DynamicImage {
        let pixel_count = (self.img_width * self.img_height) as usize;
        let mut rgba_bytes = Vec::with_capacity(pixel_count * 4);

        let image_slice = image.data();
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

        DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(self.img_width, self.img_height, rgba_bytes).unwrap(),
        )
    }

    fn build_camera(&self, translation: [f32; 3], quat: [f32; 4]) -> Camera {
        let quat = glam::Quat::from_xyzw(quat[0], quat[1], quat[2], quat[3]).normalize();
        let translation = glam::Vec3::new(translation[0], translation[1], translation[2]);
        let mut camera = self.unit_camera.clone();
        camera.position = translation;
        camera.rotation = quat;
        camera
    }
}

fn load_mask(mask_path: &str, img_width: u32, img_height: u32) -> anyhow::Result<Option<Vec<u8>>> {
    if mask_path.is_empty() {
        return Ok(None);
    }

    let mask_img =
        image::open(mask_path).with_context(|| format!("failed to open mask image {mask_path}"))?;

    ensure!(
        mask_img.width() == img_width && mask_img.height() == img_height,
        "mask resolution {}x{} doesn't match camera resolution {img_width}x{img_height}",
        mask_img.width(),
        mask_img.height(),
    );

    Ok(Some(mask_img.to_luma8().into_raw()))
}

fn build_unit_camera(camera_params: &[f64], img_width: u32, img_height: u32) -> Camera {
    let camera_model = CameraModel::Pinhole;
    let fx = camera_params[0];
    let fy = camera_params[1];
    let cx = camera_params[2];
    let cy = camera_params[3];

    Camera::new(
        glam::Vec3::ZERO,
        glam::Quat::IDENTITY,
        focal_to_fov(fx, img_width, &camera_model),
        focal_to_fov(fy, img_height, &camera_model),
        glam::vec2(cx as f32 / img_width as f32, cy as f32 / img_height as f32),
        camera_model,
    )
}

fn get_config(config_path: String) -> anyhow::Result<IncrementalProcessConfig> {
    Ok(if config_path.is_empty() {
        IncrementalProcessConfig::default()
    } else {
        let config_path = PathBuf::from(config_path);
        if fs::exists(&config_path).unwrap_or(false) {
            serde_json::from_reader(File::open(&config_path).expect("Error reading config"))?
        } else {
            serde_json::to_writer(
                File::create(&config_path)?,
                &IncrementalProcessConfig::default(),
            )?;
            IncrementalProcessConfig::default()
        }
    })
}
