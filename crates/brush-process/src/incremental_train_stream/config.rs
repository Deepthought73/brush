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

    #[arg(
        long,
        help_heading = "How many training iterations between hard opacity resets (0 disables)",
        default_value = "3000"
    )]
    pub opacity_reset_every: u32,

    #[arg(
        long,
        help_heading = "Opacity value every gaussian is capped to on a hard reset",
        default_value = "0.01"
    )]
    pub opacity_reset_value: f32,

    #[arg(
        long,
        help_heading = "How many training iterations between soft opacity decays (0 disables)",
        default_value = "200"
    )]
    pub opacity_decay_every: u32,

    #[arg(
        long,
        help_heading = "Amount subtracted from each Gaussian's opacity on a soft decay",
        default_value = "0.004"
    )]
    pub opacity_decay_amount: f32,

    #[arg(
        long,
        help_heading = "Opacity given to Gaussians on initialization",
        default_value = "0.01"
    )]
    pub initial_opacity: f32,
}
