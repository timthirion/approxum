//! Polyline and polygon simplification algorithms.

mod rdp;
mod topology;
mod visvalingam;

pub use rdp::{rdp, rdp_indices};
pub use topology::{
    topology_preserving, topology_preserving_by_count, topology_preserving_indices,
    topology_preserving_indices_by_count,
};
pub use visvalingam::{
    visvalingam, visvalingam_by_count, visvalingam_indices, visvalingam_indices_by_count,
};
