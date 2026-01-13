//! Point generation and sampling algorithms.

mod halton;
mod poisson;

pub use halton::{
    halton_2d, halton_2d_point, halton_nd, halton_sequence, halton_sequence_in_rect,
    halton_sequence_scrambled, halton_sequence_skip, radical_inverse, radical_inverse_f,
    HaltonIterator,
};
pub use poisson::{poisson_disk, poisson_disk_with_seed, PoissonDiskSampler};
