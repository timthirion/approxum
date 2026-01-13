//! Triangulation algorithms for point sets.
//!
//! This module provides algorithms for computing triangulations of point sets,
//! including Delaunay triangulation and Voronoi diagrams.

mod delaunay;
mod voronoi;

pub use delaunay::{delaunay_triangulation, in_circumcircle, Triangle};
pub use voronoi::{
    triangle_circumcenter, voronoi_diagram, VoronoiCell, VoronoiDiagram, VoronoiEdge,
};
