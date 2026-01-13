//! Point generation and sampling algorithms.

mod poisson;

pub use poisson::{poisson_disk, poisson_disk_with_seed, PoissonDiskSampler};
