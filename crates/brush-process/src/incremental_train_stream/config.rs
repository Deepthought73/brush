use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct IncrementalTrainConfig {
    #[arg(
        long,
        help_heading = "View sampling strategy",
        default_value = "random"
    )]
    pub view_sampling_strategy: String,

    #[arg(
        long,
        help_heading = "Min distance to other Gaussians for a landmark to be added",
        default_value = "0.01"
    )]
    pub depth_landmark_min_dist: f32,

    #[arg(
        long,
        help_heading = "Scale factor multiplied to newly added landmarks",
        default_value = "1.0"
    )]
    pub depth_landmark_scale_factor: f32,

    #[arg(
        long,
        help_heading = "How often new Gaussians should be added",
        default_value = "1.0"
    )]
    pub add_gaussians_every_secs: f64,
}
