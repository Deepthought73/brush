use crate::config::TrainStreamConfig;
use crate::incremental_train_stream::view_sampling::{ViewSampler, create_view_sampler};
use crate::incremental_train_stream::{FrameId, ImageData, PoseData};
use brush_dataset::scene::{SceneBatch, sample_to_packed_data_witout_copy};
use brush_render::AlphaMode;
use brush_render::camera::Camera;
use dashmap::{DashMap, DashSet};
use image::DynamicImage;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, mpsc};
use std::{mem, thread};

#[derive(Clone, Default)]
struct Inner {
    train_poses: Arc<DashMap<FrameId, Camera>>,
    eval_poses: Arc<DashMap<FrameId, Camera>>,
    image_data: Arc<DashMap<FrameId, Arc<DynamicImage>>>,
    depth_data: Arc<DashMap<FrameId, Arc<Vec<f32>>>>,
    total_poses: Arc<AtomicUsize>,
    unregistered_frame_ids: Arc<RwLock<Arc<DashSet<FrameId>>>>,
}

pub struct IncrementalDatabase {
    inner: Inner,
    view_sampler: Box<dyn ViewSampler>,
    rng: StdRng,
}

impl IncrementalDatabase {
    pub fn new(
        image_receiver: mpsc::Receiver<ImageData>,
        pose_receiver: mpsc::Receiver<PoseData>,
        unit_camera: Camera,
        config: &TrainStreamConfig,
    ) -> Self {
        let inner = Inner::default();

        let view_sampler =
            create_view_sampler(&config.incremental_train_config.view_sampling_strategy);

        spawn_image_receiver(image_receiver, inner.clone());
        spawn_pose_receiver(pose_receiver, unit_camera, config, inner.clone());

        let rng = StdRng::from_seed([config.process_config.seed as u8; 32]);

        Self {
            inner,
            view_sampler,
            rng,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.train_poses.is_empty()
    }

    pub fn get_next_train_batch(&mut self) -> SceneBatch {
        let frame_id = self.view_sampler.sample(&mut self.rng);
        let camera = *self.inner.train_poses.get(&frame_id).unwrap();
        let image = self.inner.image_data.get(&frame_id).unwrap().clone();

        let (img_packed, has_alpha) = sample_to_packed_data_witout_copy(&image);

        SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: AlphaMode::Masked,
            camera,
            depth: todo!()
        }
    }

    pub fn train_poses(&self) -> dashmap::iter::Iter<'_, FrameId, Camera> {
        self.inner.train_poses.iter()
    }

    pub fn eval_poses(&self) -> dashmap::iter::Iter<'_, FrameId, Camera> {
        self.inner.eval_poses.iter()
    }

    pub fn eval_views(&self) -> Vec<(FrameId, Camera, Arc<DynamicImage>)> {
        self.inner
            .eval_poses
            .iter()
            .filter_map(|it| {
                if let Some(image) = self.inner.image_data.get(it.key()) {
                    Some((*it.key(), *it.value(), image.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn total_view_count(&self) -> usize {
        self.inner.total_poses.load(Ordering::Relaxed)
    }

    pub fn get_unregistered_frames(&mut self) -> Vec<(Camera, Arc<DynamicImage>, Arc<Vec<f32>>)> {
        let unregistered_frame_ids = {
            let mut guard = self.inner.unregistered_frame_ids.write().unwrap();
            mem::take(&mut *guard)
        };

        let guard = self.inner.unregistered_frame_ids.read().unwrap();
        unregistered_frame_ids
            .iter()
            .filter_map(|frame_id| {
                let frame_id = *frame_id;

                if let Some(image) = self.inner.image_data.get(&frame_id)
                    && let Some(depth) = self.inner.depth_data.get(&frame_id)
                {
                    let camera = *self.inner.train_poses.get(&frame_id).unwrap();
                    self.view_sampler.added_new_item(frame_id);
                    Some((camera, image.clone(), depth.clone()))
                } else {
                    guard.insert(frame_id);
                    None
                }
            })
            .collect()
    }
}

fn spawn_image_receiver(
    image_receiver: mpsc::Receiver<ImageData>,
    inner: Inner,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while let Ok(data) = image_receiver.recv() {
            inner.image_data.insert(data.frame_id, data.image.into());
            inner.depth_data.insert(data.frame_id, data.depth.into());
        }
    })
}

fn spawn_pose_receiver(
    pose_receiver: mpsc::Receiver<PoseData>,
    unit_camera: Camera,
    config: &TrainStreamConfig,
    inner: Inner,
) -> thread::JoinHandle<()> {
    let Inner {
        train_poses,
        eval_poses,
        total_poses,
        unregistered_frame_ids,
        ..
    } = inner;

    let eval_every = config.load_config.eval_split_every;

    thread::spawn(move || {
        while let Ok(data) = pose_receiver.recv() {
            let mut camera = unit_camera;
            camera.position = data.translation;
            camera.rotation = data.quat;

            if train_poses.contains_key(&data.frame_id) {
                train_poses.insert(data.frame_id, camera);
            } else if eval_poses.contains_key(&data.frame_id) {
                eval_poses.insert(data.frame_id, camera);
            } else {
                total_poses.fetch_add(1, Ordering::Relaxed);

                if let Some(n) = eval_every
                    && total_poses.load(Ordering::Relaxed).is_multiple_of(n)
                {
                    eval_poses.insert(data.frame_id, camera);
                } else {
                    train_poses.insert(data.frame_id, camera);
                    unregistered_frame_ids.read().unwrap().insert(data.frame_id);
                }
            }
        }
    })
}
