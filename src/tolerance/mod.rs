//! Epsilon-aware geometric predicates and operations.
//!
//! All functions in this module take explicit tolerance parameters.
//! No hidden epsilons are used.

mod predicates;

pub use predicates::{
    orient2d, point_on_segment, segments_intersect, Orientation, SegmentIntersection,
};
