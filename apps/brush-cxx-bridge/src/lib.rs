use crate::ffi::{CameraModelId, StampedPose};
use anyhow::{Context, ensure};
use brush_app::ui::app::App;
use brush_incremental::IncrementalTrainMessage::{ContinueTrain, ExternalPoseUpdate, NewView};
use brush_incremental::config::IncrementalProcessConfig;
use brush_incremental::{
    IncrementalTrainMessage, IncrementalTrainerCreationContext, ViewData,
    create_incremental_training_process, run_incremental_training_headless,
};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use image::DynamicImage;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;
use std::{fs, mem};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

mod gpu_mutex;
use gpu_mutex::{GpuMutex, GpuMutexGuard, new_gpu_mutex};

#[cxx::bridge(namespace = "brush_cxx_bridge")]
mod ffi {
    #[derive(Debug)]
    enum CameraModelId {
        Pinhole,
        KannalaBrandt4,
    }

    struct StampedPose {
        frame_id: i64,
        t: [f32; 3],
        q: [f32; 4],
    }

    extern "Rust" {
        type BrushBridge;

        fn new_brush_bridge(
            config_path: String,
            camera_params: &[f64],
            camera_model_id: CameraModelId,
            img_width: u32,
            img_height: u32,
            mask_path: &str,
            gpu_mutex: Box<GpuMutex>,
        ) -> Result<Box<BrushBridge>>;

        unsafe fn add_view_to_splat(
            &mut self,
            frame_id: i64,
            image_ptr: *const u16,
            depth_ptr: *const f32,
            translation: [f32; 3],
            quat: [f32; 4],
            is_eval: bool,
            is_host_frame: bool,
        );

        fn continue_train(&mut self);

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
}

struct BrushBridge {
    cc: Option<IncrementalTrainerCreationContext>,

    message_sender: mpsc::UnboundedSender<IncrementalTrainMessage>,

    unit_camera: Camera,
    img_width: u32,
    img_height: u32,

    mask_raw: Option<Vec<u8>>,

    runtime: Runtime,
}

fn new_brush_bridge(
    config_path: String,
    camera_params: &[f64],
    camera_model_id: CameraModelId,
    img_width: u32,
    img_height: u32,
    mask_path: &str,
    gpu_mutex: Box<GpuMutex>,
) -> anyhow::Result<Box<BrushBridge>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize tokio runtime");
    let (message_sender, message_receiver) = mpsc::unbounded_channel::<IncrementalTrainMessage>();
    let config = get_config(config_path);
    let mask_raw = load_mask(mask_path, img_width, img_height)?;
    let unit_camera = build_unit_camera(camera_params, camera_model_id, img_width, img_height);

    Ok(BrushBridge {
        cc: Some(IncrementalTrainerCreationContext {
            message_receiver,
            gpu_mutex: gpu_mutex.arc(),
            config,
        }),
        message_sender,
        unit_camera,
        img_width,
        img_height,
        mask_raw,
        runtime,
    }
    .into())
}

impl BrushBridge {
    fn add_view_to_splat(
        &mut self,
        frame_id: i64,
        image_ptr: *const u16,
        depth_ptr: *const f32,
        translation: [f32; 3],
        quat: [f32; 4],
        is_eval: bool,
        is_host_frame: bool,
    ) {
        let image = unsafe { self.copy_into_rgba_image(image_ptr) };

        let depth = if depth_ptr.is_null() {
            None
        } else {
            // TODO try to pass depth data as shared_ptr to avoid copy
            let pixel_count = (self.img_width * self.img_height) as usize;
            let depth_slice = unsafe { std::slice::from_raw_parts(depth_ptr, pixel_count) };
            Some(depth_slice.to_vec())
        };

        let quat = glam::Quat::from_xyzw(quat[0], quat[1], quat[2], quat[3]).normalize();
        let translation = glam::Vec3::new(translation[0], translation[1], translation[2]);

        let mut camera = self.unit_camera.clone();
        camera.position = translation;
        camera.rotation = quat;

        let start = Instant::now();
        self.message_sender
            .send(NewView(ViewData {
                frame_id,
                camera,
                image,
                depth,
                is_eval,
                is_host_frame,
            }))
            .unwrap();

        let add_splats_dur = start.elapsed();

        if !is_eval {
            log::info!("Adding view took: {add_splats_dur:?}, is_host_frame = {is_host_frame:?}");
        }
    }

    fn continue_train(&mut self) {
        let error = self.message_sender.send(ContinueTrain).is_err();
        if error {
            return;
        }
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
        let error = self
            .message_sender
            .send(ExternalPoseUpdate(new_poses))
            .is_err();
        if error {
            return;
        }
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
            .block_on(run_incremental_training_headless(self.cc.take().unwrap()))
    }

    fn stop(&mut self) {
        drop(mem::replace(
            &mut self.message_sender,
            mpsc::unbounded_channel().0,
        ));
    }

    unsafe fn copy_into_rgba_image(&self, image_ptr: *const u16) -> DynamicImage {
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

        DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(self.img_width, self.img_height, rgba_bytes).unwrap(),
        )
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

fn build_unit_camera(
    camera_params: &[f64],
    camera_model_id: CameraModelId,
    img_width: u32,
    img_height: u32,
) -> Camera {
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

    Camera::new(
        glam::Vec3::ZERO,
        glam::Quat::IDENTITY,
        focal_to_fov(fx, img_width, &camera_model),
        focal_to_fov(fy, img_height, &camera_model),
        glam::vec2(cx as f32 / img_width as f32, cy as f32 / img_height as f32),
        camera_model,
    )
}

fn get_config(config_path: String) -> IncrementalProcessConfig {
    if config_path.is_empty() {
        IncrementalProcessConfig::default()
    } else {
        let config_path = PathBuf::from(config_path);
        if fs::exists(&config_path).unwrap_or(false) {
            serde_json::from_reader(File::open(&config_path).expect("Error reading config"))
                .unwrap()
        } else {
            serde_json::to_writer(
                File::create(&config_path).unwrap(),
                &IncrementalProcessConfig::default(),
            )
            .unwrap();
            IncrementalProcessConfig::default()
        }
    }
}
