//! Polygon operations including boolean operations and offsetting.
//!
//! This module provides algorithms for polygon manipulation including:
//! - Area and centroid calculation
//! - Point containment testing
//! - Boolean operations (union, intersection, difference, XOR)
//! - Polygon offsetting (inflate/deflate)
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
mod offset;

pub use boolean::{polygon_difference, polygon_intersection, polygon_union, polygon_xor};
pub use clip::{clip_polygon_by_convex, sutherland_hodgman};
pub use core::{polygon_area, polygon_centroid, polygon_contains, polygon_is_convex, Polygon};
pub use offset::{offset_polygon, offset_polygon_simple, JoinStyle};
