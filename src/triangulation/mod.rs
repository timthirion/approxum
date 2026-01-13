//! Triangulation algorithms for point sets.
//!
//! This module provides algorithms for computing triangulations of point sets,
//! including Delaunay triangulation.

mod delaunay;

pub use delaunay::{delaunay_triangulation, in_circumcircle, Triangle};
