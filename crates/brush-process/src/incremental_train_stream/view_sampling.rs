use rand::RngExt;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;

pub fn create_view_sampler(view_sampling_strategy: &str) -> Box<dyn ViewSampler> {
    if view_sampling_strategy == "random" {
        Box::new(RandomViewSampler::new())
    } else if view_sampling_strategy.starts_with("sliding") {
        let window_size: usize = view_sampling_strategy
            .split("-")
            .last()
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
    fn sample(&mut self, rng: &mut StdRng) -> usize;
    fn added_new_item(&mut self);
}

pub struct RandomViewSampler {
    count: usize,
}

impl RandomViewSampler {
    pub fn new() -> Self {
        Self { count: 0 }
    }
}

impl ViewSampler for RandomViewSampler {
    fn sample(&mut self, rng: &mut StdRng) -> usize {
        rng.random_range(0..self.count)
    }

    fn added_new_item(&mut self) {
        self.count += 1;
    }
}

pub struct TrainFrequencyWeightedViewSampler {
    sampling_counts: Vec<f64>,
    dist: Option<WeightedIndex<f64>>,
}

impl TrainFrequencyWeightedViewSampler {
    pub fn new() -> Self {
        Self {
            sampling_counts: Vec::new(),
            dist: None,
        }
    }
}

impl ViewSampler for TrainFrequencyWeightedViewSampler {
    fn sample(&mut self, rng: &mut StdRng) -> usize {
        let dist = self.dist.as_mut().unwrap();
        let idx = dist.sample(rng);

        self.sampling_counts[idx] += 1.;
        let new_weight = 1. / (self.sampling_counts[idx] + 1.);
        dist.update_weights(&[(idx, &new_weight)]).unwrap();

        idx
    }

    fn added_new_item(&mut self) {
        self.sampling_counts.push(0.);
        let weights: Vec<f64> = self
            .sampling_counts
            .iter()
            .map(|&c| 1.0 / (c + 1.0))
            .collect();
        self.dist = Some(WeightedIndex::new(&weights).unwrap());
    }
}

pub struct SlidingWindowViewSampler {
    count: usize,
    window_size: usize,
}

impl SlidingWindowViewSampler {
    pub fn new(window_size: usize) -> Self {
        Self {
            count: 0,
            window_size,
        }
    }
}

impl ViewSampler for SlidingWindowViewSampler {
    fn sample(&mut self, rng: &mut StdRng) -> usize {
        rng.random_range(self.count.saturating_sub(self.window_size)..self.count)
    }

    fn added_new_item(&mut self) {
        self.count += 1;
    }
}
