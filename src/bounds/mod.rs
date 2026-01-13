//! Bounding volume computation.

mod aabb;
mod circle;

pub use aabb::Aabb2;
pub use circle::{minimum_enclosing_circle, BoundingCircle};
