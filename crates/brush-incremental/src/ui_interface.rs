use crate::{IncrementalTrainer, ViewData};
use brush_dataset::Dataset;
use brush_dataset::load_image::LoadImage;
use brush_dataset::scene::SceneView;
use brush_process::config::TrainStreamConfig;
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_render::camera::Camera;
use brush_vfs::BrushVfs;
use std::path::PathBuf;
use std::sync::Arc;

impl IncrementalTrainer {
    pub async fn update_splat_in_ui(&mut self) {
        if let Some(splats) = &self.splats
            && let Some(splat_sender) = &self.splat_sender
            && let Some(emitter) = &self.emitter
        {
            splat_sender.set(0, splats.clone());
            if !self.splat_sender_initialized {
                self.splat_sender_initialized = true;
                emitter.emit(ProcessMessage::DoneLoading).await;
            }
            emitter
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

    pub async fn init_ui(&mut self) {
        if let Some(emitter) = &self.emitter {
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
                    config: Box::new(TrainStreamConfig::default()),
                }))
                .await;
        }
    }

    pub fn update_up_axis(&mut self, camera: &Camera) {
        let rot = glam::Mat3::from_quat(camera.rotation);
        if self.up_axis.is_none() {
            self.up_axis = Some(rot.y_axis);
        } else if let Some(up_axis) = self.up_axis.as_mut() {
            *up_axis *= self.up_axis_factor_count;
            *up_axis += rot.y_axis;
            *up_axis = up_axis.normalize();
        }
        self.up_axis_factor_count += 1.;
    }

    pub async fn update_ui_dataset(&self) {
        if let Some(emitter) = &self.emitter {
            let train_views = collect_scene_views(&self.train_views);
            let eval_views = collect_scene_views(&self.eval_views);

            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
                    dataset: Dataset::from_views(train_views, eval_views),
                }))
                .await;
        }
    }

    pub async fn update_train_status_ui(&mut self) {
        if let Some(emitter) = &self.emitter {
            let (num_splats, sh) = self
                .splats
                .as_ref()
                .map_or((0, 0), |it| (it.num_splats(), it.sh_degree()));
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
                    iter: 0,
                    total_elapsed: self.training_start.unwrap().elapsed(),
                    lod_progress: None,
                }))
                .await;
        }
    }
}

fn collect_scene_views(views: &[ViewData]) -> Vec<SceneView> {
    views
        .iter()
        .map(|view| {
            let img_path = PathBuf::from(&format!("{}.png", view.frame_id));
            SceneView {
                image: LoadImage::new(Arc::new(BrushVfs::empty()), img_path, None, u32::MAX, None),
                camera: view.camera,
            }
        })
        .collect()
}
