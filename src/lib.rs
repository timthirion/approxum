//! approxum - Geometric approximation algorithms
//!
//! Not everything needs to be exact. This library provides algorithms for when
//! you need practical results — trading precision for speed, simplicity, or tractability.

pub mod bounds;
pub mod curves;
pub mod error;
pub mod primitives;
pub mod simplify;
pub mod spatial;
pub mod tolerance;

pub use bounds::{minimum_enclosing_circle, Aabb2, BoundingCircle};
pub use error::ApproxError;
pub use primitives::{Point2, Point3, Segment2, Segment3, Vec2, Vec3};
pub use tolerance::{
    orient2d, point_on_segment, segments_intersect, Orientation, SegmentIntersection,
};
