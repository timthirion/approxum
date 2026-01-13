//! Bounding volume computation.

mod aabb;
mod capsule;
mod circle;
mod obb;

pub use aabb::Aabb2;
pub use capsule::BoundingCapsule;
pub use circle::{minimum_enclosing_circle, BoundingCircle};
pub use obb::Obb2;
