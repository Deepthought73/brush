#![recursion_limit = "256"]

// Platform-specific modules.
#[cfg(target_os = "android")]
mod android;
#[cfg(target_family = "wasm")]
pub mod wasm;

pub mod ui;
