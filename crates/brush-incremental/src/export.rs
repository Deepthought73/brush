use crate::IncrementalTrainer;
use anyhow::Context;
use serde::Serialize;
use std::fmt::Write as _;
use std::fs::File;
use std::path::PathBuf;
use std::time::{Instant, SystemTime};

#[derive(Serialize)]
struct MetaInfo {
    timestamp_ns: u128,
    train_view_count: usize,
    host_view_count: usize,
    num_splats: u32,
}

impl IncrementalTrainer {
    pub async fn export_all(&mut self) -> Result<(), anyhow::Error> {
        let training_secs = self.training_start.unwrap().elapsed().as_secs_f64();

        if let Some(export_every) = self.config.export_every_secs
            && training_secs > self.export_count * export_every
        {
            let now = SystemTime::now();
            let timestamp = now.duration_since(SystemTime::UNIX_EPOCH)?.as_nanos();
            let file_stem = timestamp.to_string();

            let start = Instant::now();
            self.export_ply(&file_stem).await?;
            self.export_poses(&file_stem)?;
            self.export_meta_info(&file_stem, timestamp)?;
            log::info!("Export took: {:?}", start.elapsed());
            self.export_count += 1.0;
        }

        Ok(())
    }

    async fn export_ply(&mut self, file_stem: &str) -> Result<(), anyhow::Error> {
        if let Some(splats) = self.splats.clone() {
            let export_path = PathBuf::from(&self.config.export_ply_path);
            tokio::fs::create_dir_all(&export_path)
                .await
                .with_context(|| format!("Creating export directory {}", export_path.display()))?;
            let splat_data = brush_serde::splat_to_ply(splats, self.up_axis)
                .await
                .context("Serializing splat data")?;
            tokio::fs::write(export_path.join(format!("{file_stem}.ply")), splat_data)
                .await
                .context(format!("Failed to export ply {export_path:?}"))?;
        }
        Ok(())
    }

    /// Export the training-view camera poses in TUM trajectory format
    /// (what brush-eval expects).
    ///
    /// One pose per line, space-separated, 8 fields:
    /// `timestamp tx ty tz qx qy qz qw`
    ///
    /// - `timestamp` — seconds. Derived from `frame_id` (ns) / 1e9.
    /// - `tx ty tz` — camera position in world coordinates (translation of the
    ///   camera-to-world transform).
    /// - `qx qy qz qw` — camera-to-world orientation as a normalized quaternion,
    ///   order x, y, z, w (w last).
    ///
    /// Brush stores `Camera::position`/`Camera::rotation` as the camera-to-world
    /// transform (T_wc) in the optical (+X right, +Y down, +Z forward) frame,
    /// which matches TUM directly — no inversion or axis flip needed.
    fn export_poses(&self, file_stem: &str) -> Result<(), anyhow::Error> {
        let export_path = PathBuf::from(&self.config.export_poses_path);
        std::fs::create_dir_all(&export_path)
            .with_context(|| format!("Creating export directory {}", export_path.display()))?;

        // Sort by timestamp so the trajectory is monotonically ordered.
        let mut views: Vec<_> = self.train_views.iter().collect();
        views.sort_by_key(|view| view.frame_id);

        let mut out = String::new();
        for view in views {
            let ts_seconds = view.frame_id;
            let t = view.camera.position;
            let q = (view.camera.rotation * self.r_unrectified_rectified.inverse()).normalize();
            writeln!(
                out,
                "{ts_seconds} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7} {:.7}",
                t.x, t.y, t.z, q.x, q.y, q.z, q.w
            )
            .expect("writing to a String cannot fail");
        }

        let file_path = export_path.join(format!("{file_stem}.tum"));
        std::fs::write(&file_path, out)
            .with_context(|| format!("Failed to export poses {file_path:?}"))?;

        Ok(())
    }

    fn export_meta_info(&self, file_stem: &str, timestamp_ns: u128) -> Result<(), anyhow::Error> {
        let export_path = PathBuf::from(&self.config.export_meta_info_path);
        std::fs::create_dir_all(&export_path)
            .with_context(|| format!("Creating export directory {}", export_path.display()))?;

        let num_splats = self.splats.clone().unwrap().num_splats();
        let meta_info = MetaInfo {
            timestamp_ns,
            train_view_count: self.train_views.len(),
            host_view_count: self.host_view_count,
            num_splats,
        };

        let file_path = export_path.join(format!("{file_stem}.json"));
        let out = File::create(file_path)?;
        serde_json::to_writer_pretty(out, &meta_info)?;

        Ok(())
    }
}
