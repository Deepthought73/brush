use crate::incremental_train_stream::{FrameId, IncrementalTrainContext};
use crate::message::{ProcessMessage, TrainMessage};
use brush_dataset::Dataset;
use brush_dataset::load_image::LoadImage;
use brush_dataset::scene::SceneView;
use brush_render::camera::Camera;
use brush_vfs::BrushVfs;
use image::DynamicImage;
use std::path::PathBuf;
use std::sync::Arc;

impl IncrementalTrainContext {
    pub async fn update_splat_in_ui(&mut self) {
        if let Some(splats) = &self.splats {
            self.splat_sender.set(0, splats.clone());
            if !self.splat_sender_initialized {
                self.splat_sender_initialized = true;
                self.emitter.emit(ProcessMessage::DoneLoading).await;
            }
            self.emitter
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
        self.emitter.emit(ProcessMessage::NewProcess).await;
        self.emitter
            .emit(ProcessMessage::StartLoading {
                name: "incremental".to_owned(),
                source: brush_vfs::DataSource::Path("incremental".to_owned()),
                training: true,
                base_path: None,
            })
            .await;
        self.emitter
            .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
                config: Box::new(self.config.clone()),
            }))
            .await;
    }

    pub fn update_up_axis(&mut self, new_cameras: impl Iterator<Item = Camera>) {
        for camera in new_cameras {
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
    }

    pub async fn update_ui_dataset(&self) {
        let train_views = collect_scene_views(self.database.train_poses());
        let eval_views = collect_scene_views(self.database.eval_poses());

        self.emitter
            .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
                dataset: Dataset::from_views(train_views, eval_views),
            }))
            .await;
    }

    pub async fn update_train_status_ui(&mut self) {
        let (num_splats, sh) = self
            .splats
            .as_ref()
            .map_or((0, 0), |it| (it.num_splats(), it.sh_degree()));
        self.emitter
            .emit(ProcessMessage::SplatsUpdated {
                up_axis: None,
                frame: 0,
                total_frames: 1,
                num_splats,
                sh_degree: sh,
            })
            .await;
        self.emitter
            .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                iter: self.training_iteration,
                total_elapsed: self.training_start.unwrap().elapsed(),
                lod_progress: None,
            }))
            .await;
    }

    pub async fn update_last_images(&self, image: Arc<DynamicImage>) {
        self.emitter
            .emit(ProcessMessage::TrainMessage(TrainMessage::NewImage {
                image,
            }))
            .await;
    }
}

fn collect_scene_views(iter: dashmap::iter::Iter<'_, FrameId, Camera>) -> Vec<SceneView> {
    iter.map(|view| {
        let img_path = PathBuf::from(&format!("{}.png", view.key()));
        SceneView {
            image: LoadImage::new(Arc::new(BrushVfs::empty()), img_path, None, u32::MAX, None),
            camera: *view.value(),
        }
    })
    .collect()
}
