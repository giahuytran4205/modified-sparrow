use jagua_rs::io::svg::{SvgDrawOptions, SvgLayoutTheme};
use crate::sample::search::SampleConfig;

pub const GLS_WEIGHT_MAX_INC_RATIO: f64 = 2.0;
pub const GLS_WEIGHT_MIN_INC_RATIO: f64 = 1.2;
pub const GLS_WEIGHT_DECAY: f64 = 0.95;
pub const OVERLAP_PROXY_EPSILON_DIAM_RATIO: f64 = 0.01;


/// Coordinate descent step multiplier on success
pub const CD_STEP_SUCCESS: f64 = 1.1;

/// Coordinate descent step multiplier on failure
pub const CD_STEP_FAIL: f64 = 0.5;

/// Ratio of the item's min dimension to be used as initial and limit step size for the first refinement
pub const PRE_REFINE_CD_TL_RATIOS: (f64, f64) = (0.25, 0.02);

/// Step sizes for rotation in the first refinement
pub const PRE_REFINE_CD_R_STEPS: (f64, f64) = (f64::to_radians(5.0), f64::to_radians(1.0));

/// Ratio of the item's min dimension to be used as initial and limit step size for the second (final) refinement
pub const SND_REFINE_CD_TL_RATIOS: (f64, f64) = (0.01, 0.001);

/// Step sizes for rotation in the second (final) refinement
pub const SND_REFINE_CD_R_STEPS: (f64, f64) = (f64::to_radians(0.5), f64::to_radians(0.05));

/// If two samples are closer than this ratio of the item's min dimension, they are considered duplicates
pub const UNIQUE_SAMPLE_THRESHOLD: f64 = 0.05;

pub const DEFAULT_EXPLORE_TIME_RATIO: f64 = 0.8;
pub const DEFAULT_COMPRESS_TIME_RATIO: f64 = 0.2;

pub const DEFAULT_MAX_CONSEQ_FAILS_EXPL: usize = 10;

pub const DEFAULT_FAIL_DECAY_RATIO_CMPR: f64 = 0.9;

pub const LOG_LEVEL_FILTER_RELEASE: log::LevelFilter = log::LevelFilter::Info;

pub const LOG_LEVEL_FILTER_DEBUG: log::LevelFilter = log::LevelFilter::Debug;

pub const DRAW_OPTIONS: SvgDrawOptions = SvgDrawOptions {
    theme: SvgLayoutTheme::GRAY,
    quadtree: false,
    surrogate: false,
    highlight_collisions: true,
    draw_cd_shapes: false,
    highlight_cd_shapes: true,
};

pub const LBF_SAMPLE_CONFIG: SampleConfig = SampleConfig {
    n_container_samples: 1000,
    n_focussed_samples: 0,
    n_coord_descents: 3,
};