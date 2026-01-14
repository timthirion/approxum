//! Polygon operations including boolean operations and offsetting.
//!
//! This module provides algorithms for polygon manipulation including:
//! - Area and centroid calculation
//! - Point containment testing
//! - Boolean operations (union, intersection, difference, XOR)
//! - Polygon offsetting (inflate/deflate)
//! - Minkowski sum and difference
//!
//! # Example
//!
//! ```
//! use approxum::polygon::{Polygon, polygon_intersection};
//! use approxum::Point2;
//!
//! // Two overlapping squares
//! let square1 = Polygon::new(vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(2.0, 0.0),
//!     Point2::new(2.0, 2.0),
//!     Point2::new(0.0, 2.0),
//! ]);
//!
//! let square2 = Polygon::new(vec![
//!     Point2::new(1.0, 1.0),
//!     Point2::new(3.0, 1.0),
//!     Point2::new(3.0, 3.0),
//!     Point2::new(1.0, 3.0),
//! ]);
//!
//! let intersection = polygon_intersection(&square1, &square2);
//! assert_eq!(intersection.len(), 1); // One intersection polygon
//! ```

mod boolean;
mod clip;
mod core;
mod decompose;
mod minkowski;
mod offset;
mod skeleton;
mod triangulate;
mod validate;
mod visibility;

pub use boolean::{polygon_difference, polygon_intersection, polygon_union, polygon_xor};
pub use clip::{clip_polygon_by_convex, sutherland_hodgman};
pub use core::{polygon_area, polygon_centroid, polygon_contains, polygon_is_convex, Polygon};
pub use decompose::{
    convex_decomposition, count_reflex_vertices, find_reflex_vertices, optimal_convex_decomposition,
    triangulate_decomposition,
};
pub use minkowski::{minkowski_difference, minkowski_sum, minkowski_sum_convex, polygons_collide};
pub use offset::{offset_polygon, offset_polygon_simple, JoinStyle};
pub use skeleton::{medial_axis, straight_skeleton, Skeleton, SkeletonEdge, SkeletonNode};
pub use triangulate::{
    triangulate_polygon, triangulate_polygon_indexed, triangulation_area, PolygonTriangle,
    TriangulationResult,
};
pub use validate::{
    find_self_intersections, has_self_intersection, is_simple, is_valid, remove_duplicate_vertices,
    repair_self_intersections, validate, SelfIntersection, ValidationResult,
};
pub use visibility::{
    is_visible, is_visible_with_obstacles, visibility_polygon, visibility_polygon_with_obstacles,
    visible_area,
};
