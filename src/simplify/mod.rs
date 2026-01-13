//! Polyline and polygon simplification algorithms.

mod rdp;
mod visvalingam;

pub use rdp::{rdp, rdp_indices};
pub use visvalingam::{
    visvalingam, visvalingam_by_count, visvalingam_indices, visvalingam_indices_by_count,
};
