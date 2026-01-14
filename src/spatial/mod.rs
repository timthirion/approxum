//! Spatial data structures and queries.
//!
//! This module provides acceleration structures for efficient spatial queries:
//!
//! - [`KdTree`] - A 2D KD-tree optimized for point nearest neighbor and range queries
//! - [`Bvh`] - A bounding volume hierarchy for general bounded objects

mod bvh;
mod kdtree;

pub use bvh::{Bounded, Bvh, BvhNode};
pub use kdtree::KdTree;
