use brush_render::gaussian_splats::SplatRenderMode;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

/// How new Gaussians are added for freshly registered frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GaussianAddingMode {
    OccupancyGrid,
    StridedDepth,
}

/// How the scale of a Gaussian injected during SSIM densification is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum DensifyScaleMode {
    #[default]
    Constant,
    Knn,
}

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct IncrementalProcessConfig {
    #[arg(long, default_value = "42")]
    pub seed: u64,

    #[arg(long)]
    pub eval_every_sec: Option<f64>,

    #[arg(long)]
    pub eval_train_views: bool,

    #[arg(long, default_value = "true")]
    pub export_on_eval: bool,

    #[arg(long, default_value = "./plys")]
    pub export_path: String,

    #[arg(long, default_value = "1")]
    pub sh_degree: u32,

    #[arg(long, default_value = "default")]
    pub render_mode: SplatRenderMode,

    /// Which method to use for adding new Gaussians for registered frames.
    #[arg(long, value_enum, default_value = "strided-depth")]
    pub landmark_add_mode: GaussianAddingMode,

    /// Min distance to other Gaussians for a landmark to be added
    #[arg(long, default_value = "0.04")]
    pub occupancy_grid_size: f32,

    /// Scale factor multiplied to newly added landmarks
    #[arg(long, default_value = "2.0")]
    pub cov_init_scale_factor: f32,

    /// Opacity given to Gaussians on initialization
    #[arg(long, default_value = "0.9")]
    pub cov_init_opacity: f32,

    #[arg(long, default_value = "32")]
    pub gaussians_init_depth_stride: usize,

    #[arg(skip)]
    #[serde(default)]
    pub train_config: IncrementalTrainConfig,

}

#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub struct IncrementalTrainConfig {
    pub view_sampling_strategy: String,
    pub all_view_train_steps: u32,

    pub densify_every: u32,
    pub densify_max_samples: usize,
    pub densify_recip_weighting: bool,
    pub densify_scale_mode: DensifyScaleMode,
    pub densify_const_cov_scale: f32,
    pub single_view_train_steps: u32,
    pub lr_mean: f64,
    pub lr_mean_end: f64,
    pub mean_noise_weight: f32,
    pub lr_coeffs_dc: f64,
    pub lr_coeffs_sh_scale: f32,
    pub lr_opac: f64,
    pub lr_scale: f64,
    pub lr_rotation: f64,
    pub ssim_weight: f32,
    pub anti_needle_loss_weight: f32,
    pub depth_loss_weight: f32,

    pub pose_opt: bool,
    pub lr_pose_opt: f64,
}

impl Default for IncrementalProcessConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}
