use std::sync::Arc;
use image::DynamicImage;

/// Cache budget for decoded source images. 6 GB on native; less on
/// wasm since the whole heap is bounded by browser limits.
#[cfg(not(target_family = "wasm"))]
const CACHE_BUDGET_MB: usize = 6 * 1024;
#[cfg(target_family = "wasm")]
const CACHE_BUDGET_MB: usize = 2 * 1024;

/// Shared decoded-image cache. Each slot holds at most one image; once
/// the running total passes `budget_mb`, new images bypass the cache
/// and just get re-decoded on every visit.
pub struct ImageCache {
    slots: Vec<Option<Arc<DynamicImage>>>,
    used_mb: usize,
    budget_mb: usize,
}

impl ImageCache {
    pub fn new(n_views: usize) -> Self {
        Self {
            slots: vec![None; n_views],
            used_mb: 0,
            budget_mb: CACHE_BUDGET_MB,
        }
    }

    pub  fn get(&self, index: usize) -> Option<Arc<DynamicImage>> {
        self.slots[index].clone()
    }

    pub fn insert(&mut self, index: usize, image: Arc<DynamicImage>) {
        if self.slots[index].is_some() {
            return;
        }
        let size_mb = image.as_bytes().len() / (1024 * 1024);
        if self.used_mb + size_mb < self.budget_mb {
            self.slots[index] = Some(image);
            self.used_mb += size_mb;
        }
    }
}
