//! Epsilon-aware geometric predicates and operations.
//!
//! All functions in this module take explicit tolerance parameters.
//! No hidden epsilons are used.

mod frechet;
mod hausdorff;
mod predicates;
mod weld;

pub use frechet::{
    discrete_frechet_distance, discrete_frechet_distance_linear_space, frechet_distance_approx,
    frechet_distance_at_most, frechet_distance_binary_search,
};
pub use hausdorff::{
    directed_hausdorff, directed_hausdorff_polyline, hausdorff_distance,
    hausdorff_distance_polyline, hausdorff_distance_polyline_exact,
};
pub use predicates::{
    orient2d, point_on_segment, segments_intersect, Orientation, SegmentIntersection,
};
pub use weld::{
    cleanup_polyline, remove_collinear_vertices, remove_degenerate_edges,
    remove_degenerate_edges_polygon, remove_duplicate_vertices, snap_and_weld, snap_to_grid,
    weld_vertices, weld_vertices_indexed, weld_vertices_keep_first,
};
