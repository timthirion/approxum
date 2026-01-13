//! SIMD-accelerated operations.
//!
//! This module provides vectorized versions of common geometric operations
//! for improved throughput when processing many points or evaluating curves.
//!
//! Enable with the `simd` feature flag:
//! ```toml
//! approxum = { version = "0.1", features = ["simd"] }
//! ```

mod bezier;
mod distance;
mod point;

pub use bezier::{eval_cubic_batch, eval_quadratic_batch, CubicBezier2x4, QuadraticBezier2x4};
pub use distance::{
    distances_squared_to_point, distances_to_point, distances_to_segment, nearest_point_index,
    points_within_radius,
};
pub use point::{Point2x4, Point2x8, Vec2x4, Vec2x8};
