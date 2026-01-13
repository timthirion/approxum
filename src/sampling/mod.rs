//! Point generation and sampling algorithms.

mod halton;
mod poisson;
mod sobol;

pub use halton::{
    halton_2d, halton_2d_point, halton_nd, halton_sequence, halton_sequence_in_rect,
    halton_sequence_scrambled, halton_sequence_skip, radical_inverse, radical_inverse_f,
    HaltonIterator,
};
pub use poisson::{poisson_disk, poisson_disk_with_seed, PoissonDiskSampler};
pub use sobol::{
    sobol_2d, sobol_2d_point, sobol_nd, sobol_sequence, sobol_sequence_in_rect,
    sobol_sequence_scrambled, sobol_sequence_skip, SobolIterator,
};
