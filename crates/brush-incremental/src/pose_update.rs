use crate::{FrameId, IncrementalTrainer};
use burn::Tensor;
use burn::backend::TensorData;
use burn::module::Param;

impl IncrementalTrainer {
    pub async fn update_poses(&mut self, new_updates: Vec<(FrameId, glam::Vec3, glam::Quat)>) {
        let mut updates = vec![];
        for (frame_id, position, rotation) in new_updates.iter() {
            if let Some(idx) = self.train_frame_id_to_idx.get(frame_id) {
                if let Some((start_idx, end_idx)) = self.corresponding_splats.get(frame_id) {
                    let prev_camera = self.train_views[*idx].camera;
                    let delta_q = rotation * prev_camera.rotation.inverse();
                    let delta_t = position - delta_q * prev_camera.position;

                    updates.push((delta_t, delta_q, *start_idx, *end_idx));
                }

                self.train_views[*idx].camera.position = *position;
                self.train_views[*idx].camera.rotation = *rotation;
            }
        }

        if updates.is_empty() {
            return;
        }

        let Some(splats) = self.splats.as_mut() else {
            return;
        };

        let id = splats.transforms.id;
        let device = splats.transforms.device();
        let dims = splats.transforms.dims();
        let mut data = splats
            .transforms
            .val()
            .into_data_async()
            .await
            .expect("failed to read splat transforms")
            .into_vec::<f32>()
            .expect("transforms tensor should be f32");

        for (delta_d, delta_q, start, end) in updates {
            for i in start..end {
                let base = i * 10;

                let mean = glam::Vec3::new(data[base], data[base + 1], data[base + 2]);
                let mean = delta_q * mean + delta_d;
                data[base] = mean.x;
                data[base + 1] = mean.y;
                data[base + 2] = mean.z;

                let q = glam::Quat::from_xyzw(
                    data[base + 4],
                    data[base + 5],
                    data[base + 6],
                    data[base + 3],
                );
                let q = (delta_q * q).normalize();
                data[base + 3] = q.w;
                data[base + 4] = q.x;
                data[base + 5] = q.y;
                data[base + 6] = q.z;
            }
        }

        let transforms = Tensor::from_data(TensorData::new(data, dims), &device);
        splats.transforms = Param::initialized(id, transforms.detach().require_grad());
    }
}
