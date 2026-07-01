use brush_render::gaussian_splats::SplatRenderMode;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

/// How new Gaussians are added for freshly registered frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LandmarkAddMode {
    /// Project every depth pixel and keep the ones whose cell in a coarse
    /// occupancy grid is still free.
    OccupancyGrid,
    /// Add a strided depth grid, then run a short training burst that densifies
    /// SSIM-error regions (as in the single-view-experiment).
    SsimDensification,
}

/// How the scale of a Gaussian injected during SSIM densification is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DensifyScaleMode {
    /// One fixed world-space size for every injected Gaussian.
    Constant,
    /// Density-based size from the KNN trick, respecting existing Gaussians.
    Knn,
}

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct IncrementalTrainConfig {
    #[arg(long, default_value = "42")]
    pub seed: u64,

    #[arg(long)]
    pub eval_every_sec: Option<f64>,

    #[arg(long, default_value = "true")]
    pub export_on_eval: bool,

    #[arg(long, default_value = "./plys")]
    pub export_path: String,

    #[arg(long, default_value = "1")]
    pub sh_degree: u32,

    #[arg(long, default_value = "default")]
    pub render_mode: SplatRenderMode,

    #[arg(long, default_value = "8")]
    pub eval_split_every: usize,

    /// View sampling strategy
    #[arg(long, default_value = "random")]
    pub view_sampling_strategy: String,

    /// Which method to use for adding new Gaussians for registered frames.
    #[arg(long, value_enum, default_value = "occupancy-grid")]
    pub landmark_add_mode: LandmarkAddMode,

    /// Min distance to other Gaussians for a landmark to be added
    #[arg(long, default_value = "0.04")]
    pub occupancy_grid_size: f32,

    /// Scale factor multiplied to newly added landmarks
    #[arg(long, default_value = "2.0")]
    pub cov_init_scale_factor: f32,

    /// Opacity given to Gaussians on initialization
    #[arg(long, default_value = "0.9")]
    pub cov_init_opacity: f32,

    /// How often new Gaussians should be added
    #[arg(long, default_value = "1.0")]
    pub add_gaussians_every_secs: f64,

    /// How often (in seconds) the splats and training status are pushed to the UI.
    #[arg(long, default_value = "1.0")]
    pub update_ui_every_sec: f64,

    #[arg(long, default_value = "32")]
    pub gaussians_init_depth_stride: usize,

    /// SSIM densification: number of training-burst steps run after the strided
    /// depth init when a new frame is added.
    #[arg(long, default_value = "200")]
    pub densify_steps: u32,

    /// SSIM densification: inject SSIM-weighted Gaussians every this many burst
    /// steps.
    #[arg(long, default_value = "10", value_parser = clap::value_parser!(u32).range(1..))]
    pub densify_every: u32,

    /// SSIM densification: maximum pixels injected per densification, weighted by
    /// SSIM error.
    #[arg(long, default_value = "5000")]
    pub densify_max_samples: usize,

    /// SSIM densification: baseline weight added to every pixel's SSIM error so
    /// well-reconstructed regions still have a small chance of being sampled.
    #[arg(long, default_value = "0.1")]
    pub densify_floor: f32,

    /// SSIM densification: use reciprocal SSIM weighting instead of `1 - ssim`.
    #[arg(long, default_value = "false")]
    pub densify_recip_weighting: bool,

    /// SSIM densification: learning rate for the mean parameters during the burst.
    #[arg(long, default_value = "1e-4")]
    pub densify_lr_mean: f64,

    /// SSIM densification: learning rate for the scale parameters during the burst.
    #[arg(long, default_value = "5e-2")]
    pub densify_lr_scale: f64,

    #[arg(long, default_value = "0.2")]
    pub densify_ssim_weight: f32,

    #[arg(long, default_value = "0.05")]
    pub densify_depth_loss: f32,

    #[arg(long, default_value = "0.05")]
    pub densify_anti_needle_loss: f32,

    /// SSIM densification: how the scale of injected Gaussians is chosen.
    #[arg(long, value_enum, default_value = "knn")]
    pub densify_scale_mode: DensifyScaleMode,

    /// SSIM densification: constant world-space cov scale for injected Gaussians
    /// when `--densify-scale-mode constant` (distinct from `--cov-init-scale-factor`).
    #[arg(long, default_value = "0.5")]
    pub densify_const_cov_scale: f32,
}

impl Default for IncrementalTrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}
