use crate::ffi::CameraModelId;
use brush_app::ui::app::App;
use brush_process::config::TrainStreamConfig;
use brush_process::incremental_train_stream::{
    FrameData as ProcessFrameData, NewTrainingData, create_incremental_training_process,
};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use std::collections::HashMap;
use std::mem;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::time::Instant;

#[cxx::bridge]
mod ffi {
    #[namespace = "brush_cxx_bridge"]
    #[derive(Debug)]
    enum CameraModelId {
        Pinhole,
        KannalaBrandt4,
    }

    #[namespace = "brush_cxx_bridge"]
    struct Point {
        x: f64,
        y: f64,
        z: f64,
    }

    #[namespace = "brush_cxx_bridge"]
    struct StampedPose {
        frame_id: i64,
        translation: [f64; 3],
        quat: [f64; 4],
    }

    #[namespace = "brush_cxx_bridge"]
    struct FrameData {
        new_poses: Vec<StampedPose>,
        new_landmarks: Vec<Point>,
        image_frame_id: i64,
        image_data: Vec<u8>,
    }

    #[namespace = "brush_cxx_bridge"]
    extern "Rust" {
        type FrameDataSender;
        type FrameDataReceiver;
        type FrameDataChannel;

        fn create_frame_data_channel() -> Box<FrameDataChannel>;
        fn fd_channel_take_sender(channel: &mut Box<FrameDataChannel>) -> Box<FrameDataSender>;
        fn fd_channel_take_receiver(channel: &mut Box<FrameDataChannel>) -> Box<FrameDataReceiver>;
        fn send_fd(sender: &Box<FrameDataSender>, data: FrameData);
        fn run_brush_ui(
            kfd_receiver: Box<FrameDataReceiver>,
            camera_params: Vec<f64>,
            camera_model_id: CameraModelId,
            img_width: u32,
            img_height: u32,
            mask_path: String,
        ) -> Result<()>;
    }
}

struct FrameDataChannel {
    sender: Option<Box<FrameDataSender>>,
    receiver: Option<Box<FrameDataReceiver>>,
}

struct FrameDataSender(tokio::sync::mpsc::UnboundedSender<ProcessFrameData>);
struct FrameDataReceiver(tokio::sync::mpsc::UnboundedReceiver<ProcessFrameData>);

fn create_frame_data_channel() -> Box<FrameDataChannel> {
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel::<ProcessFrameData>();
    FrameDataChannel {
        sender: Some(FrameDataSender(sender).into()),
        receiver: Some(FrameDataReceiver(receiver).into()),
    }
    .into()
}

fn fd_channel_take_sender(channel: &mut Box<FrameDataChannel>) -> Box<FrameDataSender> {
    channel.sender.take().unwrap()
}

fn fd_channel_take_receiver(channel: &mut Box<FrameDataChannel>) -> Box<FrameDataReceiver> {
    channel.receiver.take().unwrap()
}

fn send_fd(sender: &Box<FrameDataSender>, data: ffi::FrameData) {
    let poses = data
        .new_poses
        .into_iter()
        .map(|pose| {
            let quat = glam::Quat::from_xyzw(
                pose.quat[0] as f32,
                pose.quat[1] as f32,
                pose.quat[2] as f32,
                pose.quat[3] as f32,
            )
            .normalize();
            let translation = glam::Vec3::new(
                pose.translation[0] as f32,
                pose.translation[1] as f32,
                pose.translation[2] as f32,
            );
            (pose.frame_id, translation, quat)
        })
        .collect();

    let landmarks: Vec<glam::Vec3> = data
        .new_landmarks
        .iter()
        .map(|pt| glam::Vec3::new(pt.x as f32, pt.y as f32, pt.z as f32))
        .collect();

    let frame = ProcessFrameData {
        poses,
        landmarks,
        image_frame_id: data.image_frame_id,
        image_data: data.image_data,
    };

    sender.0.send(frame).unwrap();
}

async fn buffer_frame_data(
    fd_receiver: Box<FrameDataReceiver>,
    training_data_sender: UnboundedSender<NewTrainingData>,
    unit_camera: Camera,
    img_size: glam::UVec2,
    flush_every: Duration,
    mask_path: PathBuf,
) {
    let mut landmarks = vec![];
    let mut poses = vec![];
    let mut images = HashMap::new();
    let mut last_flush = Instant::now();
    let FrameDataReceiver(mut fd_receiver) = *fd_receiver;

    loop {
        let fd = fd_receiver.recv().await.unwrap();
        landmarks.extend(fd.landmarks);
        poses.extend(fd.poses);
        images.insert(fd.image_frame_id, fd.image_data);

        if !poses.is_empty() && last_flush.elapsed() >= flush_every {
            let mut poses_with_image = vec![];
            let mut poses_without_image = 0;
            for (frame_id, translation, quat) in mem::take(&mut poses) {
                if let Some(img) = images.remove(&frame_id) {
                    poses_with_image.push((frame_id, translation, quat, img));
                } else {
                    poses_without_image += 1;
                }
            }
            println!("left over poses without image: {}", poses_without_image);

            last_flush = Instant::now();
            let td = NewTrainingData::build_from_frame_data(
                poses_with_image,
                mem::take(&mut landmarks),
                unit_camera,
                img_size,
                mask_path.clone(),
            );
            training_data_sender.send(td).unwrap();
        }
    }
}

fn run_brush_ui(
    fd_receiver: Box<FrameDataReceiver>,
    camera_params: Vec<f64>,
    camera_model_id: CameraModelId,
    img_width: u32,
    img_height: u32,
    mask_path: String,
) -> anyhow::Result<()> {
    let mask_path = PathBuf::from(mask_path);
    let mut config = TrainStreamConfig::default();
    //config.train_config.max_splats = 100000;
    config.train_config.refine_every = 500;

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

    let img_size = glam::UVec2::new(img_width, img_height);

    let flush_every = Duration::from_secs(1);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize tokio runtime");

    let (training_data_sender, training_data_receiver) = unbounded_channel();
    runtime.spawn(buffer_frame_data(
        fd_receiver,
        training_data_sender,
        unit_camera,
        img_size,
        flush_every,
        mask_path,
    ));

    let process = create_incremental_training_process(training_data_receiver, config);

    runtime.block_on(async move {
        let logger = env_logger::Builder::from_default_env()
            .target(env_logger::Target::Stdout)
            .build();
        let max = logger.filter();
        brush_app::ui::log_panel::install_global_logger(Box::new(logger), max);

        let icon = eframe::icon_data::from_png_bytes(
            &include_bytes!("../../brush-app/assets/icon-256.png")[..],
        )
        .expect("Failed to load icon");

        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size(egui::Vec2::new(1450.0, 1200.0))
                .with_active(true)
                .with_icon(std::sync::Arc::new(icon)),
            wgpu_options: brush_app::ui::create_egui_options(),
            persist_window: true,
            ..Default::default()
        };

        let title = if cfg!(debug_assertions) {
            "Brush  -  Debug"
        } else {
            "Brush"
        };

        eframe::run_native(
            title,
            native_options,
            Box::new(move |cc| Ok(Box::new(App::new(cc, Some(process))))),
        )?;

        Result::<(), anyhow::Error>::Ok(())
    })?;

    Ok(())
}
