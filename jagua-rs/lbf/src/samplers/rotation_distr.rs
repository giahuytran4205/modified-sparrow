use rand::Rng;
use rand::prelude::Distribution;
use rand::prelude::IndexedRandom;
use rand_distr::Normal;
use rand_distr::Uniform;
use std::f64::consts::PI;

use jagua_rs::entities::Item;
use jagua_rs::geometry::geo_enums::RotationRange;

/// Samples a rotation (radians).
pub trait RotationSampler {
    fn sample(&self, rng: &mut impl Rng) -> f64;
}

/// Samples a rotation from a uniform distribution over a given range or a discrete set of rotations.
pub enum UniformRotDistr {
    Range(Uniform<f64>),
    Discrete(Vec<f64>),
    None,
}

/// Samples a rotation from a normal distribution over a given range or a discrete set of rotations.
/// In case of discrete rotations the mean is always returned.
pub enum NormalRotDistr {
    Range(Normal<f64>),
    Discrete(f64),
    None,
}

impl UniformRotDistr {
    pub fn from_item(item: &Item) -> Self {
        match &item.allowed_rotation {
            RotationRange::None => UniformRotDistr::None,
            RotationRange::Continuous => {
                UniformRotDistr::Range(Uniform::new(0.0, 2.0 * PI).unwrap())
            }
            RotationRange::Discrete(a_o) => UniformRotDistr::Discrete(a_o.clone()),
        }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        match self {
            UniformRotDistr::None => 0.0,
            UniformRotDistr::Range(u) => u.sample(rng),
            UniformRotDistr::Discrete(a_o) => *a_o.choose(rng).unwrap(),
        }
    }
}

impl NormalRotDistr {
    pub fn from_item(item: &Item, r_ref: f64, stddev: f64) -> Self {
        match &item.allowed_rotation {
            RotationRange::None => NormalRotDistr::None,
            RotationRange::Continuous => NormalRotDistr::Range(Normal::new(r_ref, stddev).unwrap()),
            RotationRange::Discrete(_) => NormalRotDistr::Discrete(r_ref),
        }
    }

    pub fn set_mean(&mut self, mean: f64) {
        match self {
            NormalRotDistr::Range(n) => {
                *n = Normal::new(mean, n.std_dev()).unwrap();
            }
            NormalRotDistr::Discrete(_) | NormalRotDistr::None => {}
        }
    }

    pub fn set_stddev(&mut self, stddev: f64) {
        match self {
            NormalRotDistr::Range(n) => {
                *n = Normal::new(n.mean(), stddev).unwrap();
            }
            NormalRotDistr::Discrete(_) | NormalRotDistr::None => {}
        }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        match self {
            NormalRotDistr::None => 0.0,
            NormalRotDistr::Range(n) => n.sample(rng),
            NormalRotDistr::Discrete(r) => *r,
        }
    }
}
