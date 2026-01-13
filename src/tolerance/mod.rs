//! Epsilon-aware geometric predicates and operations.
//!
//! All functions in this module take explicit tolerance parameters.
//! No hidden epsilons are used.

mod hausdorff;
mod predicates;

pub use hausdorff::{
    directed_hausdorff, directed_hausdorff_polyline, hausdorff_distance,
    hausdorff_distance_polyline, hausdorff_distance_polyline_exact,
};
pub use predicates::{
    orient2d, point_on_segment, segments_intersect, Orientation, SegmentIntersection,
};
