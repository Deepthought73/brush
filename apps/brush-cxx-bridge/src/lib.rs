use crate::ffi::CameraModelId;
use brush_app::ui::app::App;
use brush_process::incremental_train_stream::{
    FrameData as ProcessFrameData, NewTrainingData, create_incremental_training_process,
};
use brush_render::camera::{focal_to_fov, Camera};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use std::mem;
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::time::Instant;

#[cxx::bridge]
mod ffi {
    enum CameraModelId {
        Pinhole,
        KannalaBrandt4,
    }

    #[namespace = "brush_cxx_bridge"]
    #[derive(Debug)]
    struct StampedPose {
        t_ns: i64,
        /// Quaternion in [w, x, y, z] order (Sophus/Eigen convention).
        quat: [f64; 4],
        translation: [f64; 3],
    }

    #[namespace = "brush_cxx_bridge"]
    struct Point {
        x: f64,
        y: f64,
        z: f64,
    }

    /// Per-frame data sent from C++.
    ///
    /// `pose` is the camera-to-world transform for the primary camera.
    /// Intrinsics (fx, fy, cx, cy) are in pixel units; width/height are the
    /// image dimensions for this camera.
    /// `image_data` is a packed row-major grayscale u8 buffer of size width*height.
    #[namespace = "brush_cxx_bridge"]
    struct FrameData {
        pose: StampedPose,
        new_landmarks: Vec<Point>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: u32,
        height: u32,
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

/// Convert an FFI `FrameData` to the process-internal representation and send it.
///
/// The C++ side uses [w, x, y, z] quaternion order (Sophus convention); glam
/// `from_xyzw` expects [x, y, z, w], so the components are reordered here.
fn send_fd(sender: &Box<FrameDataSender>, data: ffi::FrameData) {
    let p = &data.pose;
    // C++ layout: quat = [w, x, y, z]
    let quat = glam::Quat::from_xyzw(
        p.quat[1] as f32,
        p.quat[2] as f32,
        p.quat[3] as f32,
        p.quat[0] as f32,
    )
    .normalize();
    let translation = glam::Vec3::new(
        p.translation[0] as f32,
        p.translation[1] as f32,
        p.translation[2] as f32,
    );

    let landmarks: Vec<glam::Vec3> = data
        .new_landmarks
        .iter()
        .map(|pt| glam::Vec3::new(pt.x as f32, pt.y as f32, pt.z as f32))
        .collect();

    let frame = ProcessFrameData {
        t_ns: p.t_ns,
        translation,
        quat,
        landmarks,
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
) {
    let mut buffer = vec![];
    let mut last_flush = Instant::now();
    let FrameDataReceiver(mut fd_receiver) = *fd_receiver;

    loop {
        let fd = fd_receiver.recv().await.unwrap();
        buffer.push(fd);

        if !buffer.is_empty() && last_flush.elapsed() >= flush_every {
            last_flush = Instant::now();
            let td = NewTrainingData::build_from_frame_data(
                mem::take(&mut buffer),
                unit_camera,
                img_size,
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
) -> anyhow::Result<()> {
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
        focal_to_fov(fx, img_width, &CameraModel::default()),
        focal_to_fov(fy, img_height, &CameraModel::default()),
        glam::vec2(
            cx as f32 / img_width as f32,
            cy as f32 / img_height as f32,
        ),
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
    ));

    let process = create_incremental_training_process(
        training_data_receiver,
        brush_process::config::TrainStreamConfig::default(),
    );

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
