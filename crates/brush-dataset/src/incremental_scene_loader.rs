use crate::image_cache::ImageCache;
use crate::scene::{Scene, SceneBatch, SceneView, sample_to_packed_data, view_to_sample_image};
use brush_async::Actor;
use rand::{SeedableRng, seq::SliceRandom};
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::time::sleep;

pub struct IncrementalSceneLoader {
    rx: mpsc::Receiver<SceneBatch>,
    // Owns the loader actor threads. Dropping cancels them; their
    // senders then drop, the channel closes, and `next_batch` returns.
    _actors: Vec<Actor>,
}

impl IncrementalSceneLoader {
    pub fn new(mut all_views: Vec<SceneView>, seed: u64) -> Self {
        // Prefetch buffer: at most 4 batches ahead of the trainer.
        // Two tasks per actor share this buffer so one task's I/O can
        // overlap with the other's decode + GPU upload.
        let (tx, rx) = mpsc::channel(4);

        // Fan out only as many loaders as we have real parallelism.
        // Wasm shares one JS event loop, so extra actors just add
        // contention without overlapping I/O.
        let n_actors = if cfg!(target_family = "wasm") {
            1
        } else {
            std::thread::available_parallelism().map_or(8, |p| p.get())
        };
        const TASKS_PER_ACTOR: usize = 2;

        all_views.sort_by_key(|it| it.image.img_name());
        let mut all_views = VecDeque::from(all_views);
        let first_ts = all_views[0]
            .image
            .img_name()
            .split('.')
            .next()
            .unwrap()
            .parse::<usize>()
            .unwrap();

        let cache = Arc::new(Mutex::new(ImageCache::new(all_views.len())));
        let train_views = Arc::new(RwLock::new(vec![
            all_views.pop_front().unwrap(),
            all_views.pop_front().unwrap(),
            all_views.pop_front().unwrap(),
            all_views.pop_front().unwrap(),
            all_views.pop_front().unwrap(),
        ]));

        let mut task_idx: u64 = 0;
        let actors: Vec<Actor> = (0..n_actors)
            .map(|i| {
                let actor = Actor::new(&format!("dataloader-{i}"));
                for _ in 0..TASKS_PER_ACTOR {
                    let views = train_views.clone();
                    let cache = cache.clone();
                    let tx = tx.clone();
                    let task_seed = seed.wrapping_add(task_idx);
                    task_idx += 1;
                    actor
                        .run(move || run_loader(views, cache, tx, task_seed))
                        .detach();
                }
                actor
            })
            .collect();

        tokio::task::spawn_local(async move {
            let train_time = Instant::now();
            loop {
                if all_views.is_empty() {
                    break;
                }

                let ts = train_time.elapsed().as_nanos() as usize;
                let mut train_views_ = train_views.write().await;
                while let Some(first) = all_views.front() {
                    let img_ts = first
                        .image
                        .img_name()
                        .split('.')
                        .next()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap()
                        - first_ts;
                    if img_ts > ts {
                        break;
                    }
                    train_views_.push(all_views.pop_front().unwrap());
                }
                println!("Train views len: {}", train_views_.len());
                drop(train_views_);
                sleep(Duration::from_secs(1)).await;
            }

            println!("All images added!");
        });

        Self {
            rx,
            _actors: actors,
        }
    }

    pub async fn next_batch(&mut self) -> SceneBatch {
        self.rx
            .recv()
            .await
            .expect("Scene loader channel closed unexpectedly")
    }
}

async fn run_loader(
    views: Arc<RwLock<Vec<SceneView>>>,
    cache: Arc<Mutex<ImageCache>>,
    tx: mpsc::Sender<SceneBatch>,
    seed: u64,
) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut shuffled: Vec<usize> = Vec::new();

    loop {
        let views_ = views.read().await;
        if shuffled.is_empty() {
            shuffled = (0..views_.len()).collect();
            shuffled.shuffle(&mut rng);
            shuffled.truncate(20);
        }
        let index = shuffled.pop().expect("Need at least one view in dataset");
        let view = views_[index].clone();
        drop(views_);

        let sample = if let Some(image) = cache.lock().await.get(index) {
            image
        } else {
            let raw = view
                .image
                .load()
                .await
                .expect("Scene loader failed to load an image");
            let sample = Arc::new(view_to_sample_image(raw, view.image.alpha_mode()));
            cache.lock().await.insert(index, sample.clone());
            sample
        };

        let (img_packed, has_alpha) = sample_to_packed_data(sample.as_ref().clone());
        let batch = SceneBatch {
            img_packed,
            has_alpha,
            alpha_mode: view.image.alpha_mode(),
            camera: view.camera.clone(),
        };

        if tx.send(batch).await.is_err() {
            break;
        }
        brush_async::yield_now().await;
    }
}
