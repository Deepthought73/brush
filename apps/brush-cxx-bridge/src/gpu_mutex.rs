use parking_lot::{ArcMutexGuard, Mutex, RawMutex};
use std::sync::Arc;

pub struct GpuMutex(Arc<Mutex<()>>);

pub struct GpuMutexGuard(#[allow(dead_code)] ArcMutexGuard<RawMutex, ()>);

pub fn new_gpu_mutex() -> Box<GpuMutex> {
    Box::new(GpuMutex(Arc::new(Mutex::new(()))))
}

impl GpuMutex {
    pub fn lock(&self) -> Box<GpuMutexGuard> {
        Box::new(GpuMutexGuard(self.0.lock_arc()))
    }

    pub fn clone(&self) -> Box<GpuMutex> {
        Box::new(GpuMutex(self.0.clone()))
    }

    pub fn arc(&self) -> Arc<Mutex<()>> {
        self.0.clone()
    }
}
