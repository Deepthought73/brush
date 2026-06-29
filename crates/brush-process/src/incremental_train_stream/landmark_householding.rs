use crate::incremental_train_stream::IncrementalTrainContext;
use brush_render::Splats;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, inverse_sigmoid};
use brush_render::shaders::SH_C0;
use brush_serde::SplatData;
use brush_train::to_init_splats;
use burn::Tensor;
use burn::module::Param;
use burn::tensor::TensorData;
use dashmap::DashSet;
use image::DynamicImage;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;
use std::time::Instant;

impl IncrementalTrainContext {
    async fn ensure_occupancy_grid_valid(&mut self) {
        if self.occupancy_grid.is_none() {
            let min_dist = self.config.incremental_train_config.depth_landmark_min_dist;
            let grid = OccupancyGrid::new(min_dist);
            if let Some(s) = &self.splats {
                let data = s
                    .means()
                    .into_data_async()
                    .await
                    .expect("failed to read gaussian means")
                    .into_vec::<f32>()
                    .expect("means tensor should be f32");

                data.as_chunks::<3>().0.par_iter().for_each(|it| {
                    grid.insert(glam::Vec3::from_slice(it));
                });
            }
            self.occupancy_grid = Some(grid);
        }
    }

    pub async fn add_new_landmarks_by_depth(
        &mut self,
        camera: Camera,
        image: Arc<DynamicImage>,
        depth: Arc<Vec<f32>>,
    ) -> (usize, usize) {
        let mut means = vec![];
        let mut sh_coeffs = vec![];
        let mut log_scales = vec![];

        let w = image.width() as usize;
        let h = image.height() as usize;
        let img_size = glam::UVec2::new(image.width(), image.height());

        let raw_img = image.as_rgba8().unwrap().as_raw();

        let focal = camera.focal(img_size);
        let factor = self
            .config
            .incremental_train_config
            .depth_landmark_scale_factor;

        self.ensure_occupancy_grid_valid().await;
        let grid = self.occupancy_grid.as_ref().unwrap();

        let candidates: Vec<(glam::Vec3, f32, f32)> = (0..h * w)
            .into_par_iter()
            .filter_map(|idx| {
                let d = depth[idx];
                if d <= 0.01 {
                    return None;
                }

                let u = idx % w;
                let v = idx / w;
                let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);

                let pos_cam = camera.unproject(uv, d, img_size);
                let pos_world = camera.transform(pos_cam);

                if !grid.is_free(pos_world) {
                    return None;
                }

                let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;
                let log_s = (factor * d / focal.x).ln();
                Some((pos_world, color, log_s))
            })
            .collect();

        for (pos_world, color, log_s) in candidates {
            if !grid.is_free(pos_world) {
                continue;
            }
            grid.insert(pos_world);

            means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
            sh_coeffs.extend_from_slice(&[color, color, color]);
            log_scales.extend_from_slice(&[log_s, log_s, log_s]);
        }

        self.add_new_landmarks_by_means(means, Some(sh_coeffs), Some(log_scales))
    }

    fn add_new_landmarks_by_means(
        &mut self,
        means: Vec<f32>,
        sh_coeffs: Option<Vec<f32>>,
        log_scales: Option<Vec<f32>>,
    ) -> (usize, usize) {
        let sh_degree = self.config.model_config.sh_degree;
        let render_mode = self
            .config
            .train_config
            .render_mode
            .unwrap_or(SplatRenderMode::Default);

        let n_splats = means.len() / 3;
        let new_splat = to_init_splats(
            SplatData {
                means,
                rotations: None,
                log_scales,
                sh_coeffs,
                raw_opacities: Some(vec![
                    inverse_sigmoid(
                        self.config.incremental_train_config.initial_opacity
                    );
                    n_splats
                ]),
            },
            render_mode,
            &self.device,
        )
        .with_sh_degree(sh_degree);

        let splats = self.splats.take();

        let splats_before = splats.as_ref().map(|it| it.num_splats()).unwrap_or(0) as usize;
        let splats_after = splats_before + n_splats;

        self.splats = Some(match splats {
            None => new_splat,
            Some(existing) => concat_splats(&existing, &new_splat, render_mode),
        });

        (splats_before, splats_after)
    }

    pub async fn update_poses(&mut self) {
        let start = Instant::now();

        let pose_updates = self.database.collect_pose_updates();

        if pose_updates.is_empty() {
            return;
        }

        let updates: Vec<(glam::Vec3, glam::Quat, usize, usize)> = pose_updates
            .into_iter()
            .map(|(frame_id, delta_d, delta_q)| {
                let (start, end) = self.corresponding_splats.get(&frame_id).unwrap();
                (delta_d, delta_q, *start, *end)
            })
            .collect();

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

        log::info!("Updating poses took {:?}", start.elapsed());
    }
}

#[derive(Clone)]
pub struct OccupancyGrid {
    inv_grid_size: f32,
    cells: Arc<DashSet<[i32; 3]>>,
}

impl OccupancyGrid {
    pub fn new(grid_size: f32) -> Self {
        Self {
            inv_grid_size: 1.0 / grid_size,
            cells: Default::default(),
        }
    }

    fn cell_of(&self, p: glam::Vec3) -> [i32; 3] {
        [
            (p.x * self.inv_grid_size).floor() as i32,
            (p.y * self.inv_grid_size).floor() as i32,
            (p.z * self.inv_grid_size).floor() as i32,
        ]
    }

    fn insert(&self, p: glam::Vec3) {
        self.cells.insert(self.cell_of(p));
    }

    fn is_free(&self, p: glam::Vec3) -> bool {
        !self.cells.contains(&self.cell_of(p))
    }
}

fn concat_splats(a: &Splats, b: &Splats, mode: SplatRenderMode) -> Splats {
    let means = Tensor::cat(vec![a.means(), b.means()], 0);
    let rotations = Tensor::cat(vec![a.rotations(), b.rotations()], 0);
    let log_scales = Tensor::cat(vec![a.log_scales(), b.log_scales()], 0);
    let sh_coeffs = Tensor::cat(vec![a.sh_coeffs.val(), b.sh_coeffs.val()], 0);
    let opacities = Tensor::cat(vec![a.raw_opacities.val(), b.raw_opacities.val()], 0);
    Splats::from_tensor_data(means, rotations, log_scales, sh_coeffs, opacities, mode)
}
