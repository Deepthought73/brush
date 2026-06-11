use crate::incremental_train_stream::FrameId;
use rand::RngExt;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

pub fn create_view_sampler(view_sampling_strategy: &str) -> Box<dyn ViewSampler> {
    if view_sampling_strategy == "random" {
        Box::new(RandomViewSampler::new())
    } else if view_sampling_strategy.starts_with("sliding") {
        let window_size: usize = view_sampling_strategy
            .split('-')
            .next_back()
            .unwrap()
            .parse()
            .unwrap();
        Box::new(SlidingWindowViewSampler::new(window_size))
    } else if view_sampling_strategy == "weighted" {
        Box::new(TrainFrequencyWeightedViewSampler::new())
    } else {
        panic!("invalid view samplings strategy")
    }
}

pub trait ViewSampler: Send + Sync {
    fn sample(&mut self, rng: &mut StdRng) -> FrameId;
    fn added_new_item(&mut self, frame_id: FrameId);
}

pub struct RandomViewSampler {
    frame_ids: Vec<FrameId>,
}

impl RandomViewSampler {
    pub fn new() -> Self {
        Self { frame_ids: vec![] }
    }
}

impl ViewSampler for RandomViewSampler {
    fn sample(&mut self, rng: &mut StdRng) -> FrameId {
        *self.frame_ids.choose(rng).unwrap()
    }

    fn added_new_item(&mut self, frame_id: FrameId) {
        self.frame_ids.push(frame_id);
    }
}

pub struct TrainFrequencyWeightedViewSampler {
    sampling_counts: Vec<f64>,
    frame_ids: Vec<FrameId>,
    dist: Option<WeightedIndex<f64>>,
}

impl TrainFrequencyWeightedViewSampler {
    pub fn new() -> Self {
        Self {
            sampling_counts: vec![],
            frame_ids: vec![],
            dist: None,
        }
    }
}

impl ViewSampler for TrainFrequencyWeightedViewSampler {
    fn sample(&mut self, rng: &mut StdRng) -> FrameId {
        let dist = self.dist.as_mut().unwrap();
        let idx = dist.sample(rng);

        self.sampling_counts[idx] += 1.;
        let new_weight = 1. / (self.sampling_counts[idx] + 1.);
        dist.update_weights(&[(idx, &new_weight)]).unwrap();

        self.frame_ids[idx]
    }

    fn added_new_item(&mut self, frame_id: FrameId) {
        self.sampling_counts.push(0.);
        self.frame_ids.push(frame_id);

        let weights: Vec<f64> = self
            .sampling_counts
            .iter()
            .map(|&c| 1.0 / (c + 1.0))
            .collect();
        self.dist = Some(WeightedIndex::new(&weights).unwrap());
    }
}

pub struct SlidingWindowViewSampler {
    frame_ids: Vec<FrameId>,
    window_size: usize,
}

impl SlidingWindowViewSampler {
    pub fn new(window_size: usize) -> Self {
        Self {
            frame_ids: vec![],
            window_size,
        }
    }
}

impl ViewSampler for SlidingWindowViewSampler {
    fn sample(&mut self, rng: &mut StdRng) -> FrameId {
        let count = self.frame_ids.len();
        let idx = rng.random_range(count.saturating_sub(self.window_size)..count);
        self.frame_ids[idx]
    }

    fn added_new_item(&mut self, frame_id: FrameId) {
        self.frame_ids.push(frame_id);
    }
}
