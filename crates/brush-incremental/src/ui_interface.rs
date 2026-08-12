use crate::{IncrementalTrainer, ViewData};
use async_fn_stream::TryStreamEmitter;
use brush_dataset::Dataset;
use brush_dataset::load_image::LoadImage;
use brush_dataset::scene::SceneView;
use brush_process::config::TrainStreamConfig;
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::slot::SlotSender;
use brush_render::Splats;
use brush_vfs::BrushVfs;
use std::path::PathBuf;
use std::sync::Arc;

pub struct UpdateUiContext {
    pub emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
    pub splat_sender: SlotSender<Splats>,
    pub splat_sender_initialized: bool,
}

impl UpdateUiContext {
    pub fn new(
        emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
        splat_sender: SlotSender<Splats>,
    ) -> Self {
        Self {
            emitter,
            splat_sender,
            splat_sender_initialized: false,
        }
    }
}

impl IncrementalTrainer {
    pub async fn update_splat_in_ui(&mut self) {
        if let Some(splats) = &self.splats
            && let Some(ctx) = &mut self.ui_ctx
        {
            ctx.splat_sender.set(0, splats.clone());
            if !ctx.splat_sender_initialized {
                ctx.splat_sender_initialized = true;
                ctx.emitter.emit(ProcessMessage::DoneLoading).await;
            }
            ctx.emitter
                .emit(ProcessMessage::SplatsUpdated {
                    up_axis: self.up_axis,
                    frame: 0,
                    total_frames: 1,
                    num_splats: splats.num_splats(),
                    sh_degree: splats.sh_degree(),
                })
                .await;
            self.up_axis = None;
        }
    }

    pub async fn init_ui(&self) {
        if let Some(ctx) = &self.ui_ctx {
            ctx.emitter.emit(ProcessMessage::NewProcess).await;
            ctx.emitter
                .emit(ProcessMessage::StartLoading {
                    name: "incremental".to_owned(),
                    source: brush_vfs::DataSource::Path("incremental".to_owned()),
                    training: true,
                    base_path: None,
                })
                .await;
            ctx.emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
                    config: Box::new(TrainStreamConfig::default()),
                }))
                .await;
        }
    }

    pub async fn update_ui_dataset(&self) {
        if let Some(ctx) = &self.ui_ctx {
            let train_views = collect_scene_views(self.train_views.iter());

            ctx.emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
                    dataset: Dataset::from_views(train_views, vec![]),
                }))
                .await;
        }
    }

    pub async fn update_train_status_ui(&self) {
        if let Some(ctx) = &self.ui_ctx {
            let (num_splats, sh) = self
                .splats
                .as_ref()
                .map_or((0, 0), |it| (it.num_splats(), it.sh_degree()));
            ctx.emitter
                .emit(ProcessMessage::SplatsUpdated {
                    up_axis: None,
                    frame: 0,
                    total_frames: 1,
                    num_splats,
                    sh_degree: sh,
                })
                .await;
            ctx.emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                    iter: 0,
                    total_elapsed: self.training_duration(),
                    lod_progress: None,
                }))
                .await;
        }
    }

    pub async fn update_eval_ui(&self, avg_psnr: f32, avg_ssim: f32) {
        if let Some(ctx) = &self.ui_ctx {
            ctx.emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::EvalResult {
                    iter: 0,
                    avg_psnr,
                    avg_ssim,
                }))
                .await;
        }
    }
}

fn collect_scene_views<'a>(views: impl Iterator<Item = &'a ViewData>) -> Vec<SceneView> {
    views
        .map(|view| {
            let img_path = PathBuf::from(&format!("{}.png", view.frame_id));
            SceneView {
                image: LoadImage::new(Arc::new(BrushVfs::empty()), img_path, None, u32::MAX, None),
                camera: view.camera,
            }
        })
        .collect()
}
