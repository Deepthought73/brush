use crate::incremental_train_stream::{IncrementalTrainContext, concat_splats};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::shaders::SH_C0;
use brush_serde::SplatData;
use brush_train::to_init_splats;
use image::DynamicImage;
use std::collections::HashSet;
use std::sync::Arc;

impl IncrementalTrainContext {
    async fn ensure_occupancy_grid_valid(&mut self) {
        if self.occupancy_grid.is_none() {
            let min_dist = self.config.incremental_train_config.depth_landmark_min_dist;
            let mut grid = OccupancyGrid::new(min_dist);
            if let Some(s) = &self.splats {
                let data = s
                    .means()
                    .into_data_async()
                    .await
                    .expect("failed to read gaussian means")
                    .into_vec::<f32>()
                    .expect("means tensor should be f32");
                for c in data.as_chunks::<3>().0 {
                    grid.insert(glam::Vec3::from_slice(c));
                }
            }
            self.occupancy_grid = Some(grid);
        }
    }

    pub async fn add_new_landmarks_by_depth(
        &mut self,
        camera: Camera,
        image: Arc<DynamicImage>,
        depth: Arc<Vec<u16>>,
    ) {
        let mut means = vec![];
        let mut sh_coeffs = vec![];
        let mut log_scales = vec![];

        let w = image.width() as usize;
        let h = image.height() as usize;
        let img_size = glam::UVec2::new(image.width(), image.height());

        let raw_img = image.as_rgba8().unwrap().as_raw();

        let stride = 1;
        let focal = camera.focal(img_size);

        self.ensure_occupancy_grid_valid().await;
        let grid = self.occupancy_grid.as_mut().unwrap();

        for v in (0..h).step_by(stride) {
            for u in (0..w).step_by(stride) {
                let idx = v * w + u;
                let d = depth[idx];
                if d == 0 {
                    continue;
                }

                // TODO depth is in mm, maybe preprocess somewhere else, if the unit changes
                // TODO or: provide unit of depth in config
                let d = d as f32 / 1000.;

                let uv = glam::Vec2::new(u as f32 + 0.5, v as f32 + 0.5);

                let pos_cam = camera.unproject(uv, d, img_size);
                let pos_world = camera.transform(pos_cam);

                // Skip points too close to an existing/just-added gaussian.
                // Inserting accepted points makes the new batch self-dedup too.
                if !grid.is_free(pos_world) {
                    continue;
                }
                grid.insert(pos_world);

                let color = (raw_img[idx * 4] as f32 / 255.0 - 0.5) / SH_C0;

                means.extend_from_slice(&[pos_world.x, pos_world.y, pos_world.z]);
                sh_coeffs.extend_from_slice(&[color, color, color]);

                let factor = self
                    .config
                    .incremental_train_config
                    .depth_landmark_scale_factor;
                let scale = factor * d / focal.x;
                let log_s = scale.ln();
                log_scales.extend_from_slice(&[log_s, log_s, log_s]);
            }
        }

        self.add_new_landmarks_by_means(means, Some(sh_coeffs), Some(log_scales));
    }

    fn add_new_landmarks_by_means(
        &mut self,
        means: Vec<f32>,
        sh_coeffs: Option<Vec<f32>>,
        log_scales: Option<Vec<f32>>,
    ) {
        let sh_degree = self.config.model_config.sh_degree;
        let render_mode = self
            .config
            .train_config
            .render_mode
            .unwrap_or(SplatRenderMode::Default);

        let new_splat = to_init_splats(
            SplatData {
                means,
                rotations: None,
                log_scales,
                sh_coeffs,
                raw_opacities: None,
            },
            render_mode,
            &self.device,
        )
        .with_sh_degree(sh_degree);

        self.splats = Some(match self.splats.take() {
            None => new_splat,
            Some(existing) => concat_splats(&existing, &new_splat, render_mode),
        });
    }
}

pub struct OccupancyGrid {
    inv_grid_size: f32,
    cells: HashSet<[i32; 3]>,
}

impl OccupancyGrid {
    pub fn new(grid_size: f32) -> Self {
        Self {
            inv_grid_size: 1.0 / grid_size,
            cells: HashSet::new(),
        }
    }

    fn cell_of(&self, p: glam::Vec3) -> [i32; 3] {
        [
            (p.x * self.inv_grid_size).floor() as i32,
            (p.y * self.inv_grid_size).floor() as i32,
            (p.z * self.inv_grid_size).floor() as i32,
        ]
    }

    fn insert(&mut self, p: glam::Vec3) {
        self.cells.insert(self.cell_of(p));
    }

    fn is_free(&self, p: glam::Vec3) -> bool {
        let [cx, cy, cz] = self.cell_of(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if self.cells.contains(&[cx + dx, cy + dy, cz + dz]) {
                        return false;
                    }
                }
            }
        }
        true
    }
}
