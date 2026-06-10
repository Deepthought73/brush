use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct IncrementalTrainConfig {
    #[arg(long, help_heading = "View sampling strategy", default_value = "random")]
    pub view_sampling_strategy: String
}
