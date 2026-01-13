//! Bounding volume computation.

mod aabb;
mod circle;
mod obb;

pub use aabb::Aabb2;
pub use circle::{minimum_enclosing_circle, BoundingCircle};
pub use obb::Obb2;
