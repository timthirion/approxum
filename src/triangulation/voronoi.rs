//! Voronoi diagram computation from Delaunay triangulation.
//!
//! A Voronoi diagram partitions the plane into cells, where each cell contains
//! all points closer to its generating site than to any other site.
//!
//! # How It Works
//!
//! The Voronoi diagram is the dual of the Delaunay triangulation:
//! - Each Delaunay triangle's circumcenter becomes a Voronoi vertex
//! - Each Delaunay edge shared by two triangles becomes a Voronoi edge
//! - Edges on the convex hull create unbounded Voronoi rays
//!
//! # Example
//!
//! ```
//! use approxum::triangulation::{voronoi_diagram, VoronoiDiagram};
//! use approxum::Point2;
//!
//! let sites: Vec<Point2<f64>> = vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 0.0),
//!     Point2::new(0.5, 1.0),
//! ];
//!
//! let voronoi = voronoi_diagram(&sites);
//!
//! // One triangle means one Voronoi vertex (the circumcenter)
//! assert_eq!(voronoi.vertices.len(), 1);
//!
//! // Three sites means three cells
//! assert_eq!(voronoi.cells.len(), 3);
//! ```

use crate::primitives::{Point2, Vec2};
use crate::triangulation::delaunay_triangulation;
use num_traits::Float;
use std::collections::HashMap;

/// A Voronoi diagram.
#[derive(Debug, Clone)]
pub struct VoronoiDiagram<F> {
    /// The Voronoi vertices (circumcenters of Delaunay triangles).
    pub vertices: Vec<Point2<F>>,

    /// The Voronoi edges. Each edge is either:
    /// - `Finite(a, b)`: connects vertices[a] to vertices[b]
    /// - `Infinite(a, dir)`: ray from vertices[a] in direction dir
    pub edges: Vec<VoronoiEdge<F>>,

    /// For each input site, the indices of vertices forming its cell boundary.
    /// Vertices are in counter-clockwise order around the site.
    /// Empty if the site has an unbounded cell with no finite vertices.
    pub cells: Vec<VoronoiCell>,
}

/// A Voronoi edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoronoiEdge<F> {
    /// A finite edge connecting two Voronoi vertices.
    Finite(usize, usize),
    /// An infinite ray starting at a vertex and extending in the given direction.
    Ray(usize, Vec2<F>),
}

/// A Voronoi cell for a site.
#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiCell {
    /// Index of the site this cell belongs to.
    pub site: usize,
    /// Indices of Voronoi vertices forming the cell boundary (CCW order).
    /// May be empty for unbounded cells.
    pub vertices: Vec<usize>,
    /// Whether this cell extends to infinity.
    pub unbounded: bool,
}

/// An edge in the Delaunay triangulation, normalized for hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DelaunayEdge(usize, usize);

impl DelaunayEdge {
    fn new(a: usize, b: usize) -> Self {
        if a < b {
            DelaunayEdge(a, b)
        } else {
            DelaunayEdge(b, a)
        }
    }
}

/// Computes the circumcenter of a triangle.
///
/// The circumcenter is equidistant from all three vertices.
fn circumcenter<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> Point2<F> {
    let two = F::from(2.0).unwrap();

    let ax = a.x;
    let ay = a.y;
    let bx = b.x;
    let by = b.y;
    let cx = c.x;
    let cy = c.y;

    let d = two * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

    if d.abs() < F::epsilon() {
        // Degenerate case: collinear points, return centroid
        let three = F::from(3.0).unwrap();
        return Point2::new((ax + bx + cx) / three, (ay + by + cy) / three);
    }

    let ax2_ay2 = ax * ax + ay * ay;
    let bx2_by2 = bx * bx + by * by;
    let cx2_cy2 = cx * cx + cy * cy;

    let ux = (ax2_ay2 * (by - cy) + bx2_by2 * (cy - ay) + cx2_cy2 * (ay - by)) / d;
    let uy = (ax2_ay2 * (cx - bx) + bx2_by2 * (ax - cx) + cx2_cy2 * (bx - ax)) / d;

    Point2::new(ux, uy)
}

/// Computes the Voronoi diagram from a set of sites.
///
/// # Arguments
///
/// * `sites` - The generating points for the Voronoi cells
///
/// # Returns
///
/// A `VoronoiDiagram` containing vertices, edges, and cells.
/// Returns an empty diagram if fewer than 3 sites are provided.
///
/// # Example
///
/// ```
/// use approxum::triangulation::voronoi_diagram;
/// use approxum::Point2;
///
/// // Square of points
/// let sites: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
/// ];
///
/// let voronoi = voronoi_diagram(&sites);
///
/// // 4 sites in a square produce 2 triangles with different circumcenters
/// // but both circumcenters are at the center (0.5, 0.5) for a square
/// assert!(!voronoi.vertices.is_empty());
/// assert_eq!(voronoi.cells.len(), 4);
/// ```
pub fn voronoi_diagram<F: Float>(sites: &[Point2<F>]) -> VoronoiDiagram<F> {
    if sites.len() < 3 {
        return VoronoiDiagram {
            vertices: Vec::new(),
            edges: Vec::new(),
            cells: (0..sites.len())
                .map(|i| VoronoiCell {
                    site: i,
                    vertices: Vec::new(),
                    unbounded: true,
                })
                .collect(),
        };
    }

    // Compute Delaunay triangulation
    let triangles = delaunay_triangulation(sites);

    if triangles.is_empty() {
        // All points are collinear
        return VoronoiDiagram {
            vertices: Vec::new(),
            edges: Vec::new(),
            cells: (0..sites.len())
                .map(|i| VoronoiCell {
                    site: i,
                    vertices: Vec::new(),
                    unbounded: true,
                })
                .collect(),
        };
    }

    // Compute circumcenters (Voronoi vertices)
    let vertices: Vec<Point2<F>> = triangles
        .iter()
        .map(|tri| circumcenter(sites[tri.a], sites[tri.b], sites[tri.c]))
        .collect();

    // Build edge-to-triangle adjacency map
    // Maps each Delaunay edge to the triangles that share it
    let mut edge_to_triangles: HashMap<DelaunayEdge, Vec<usize>> = HashMap::new();

    for (tri_idx, tri) in triangles.iter().enumerate() {
        for (ea, eb) in tri.edges() {
            let edge = DelaunayEdge::new(ea, eb);
            edge_to_triangles.entry(edge).or_default().push(tri_idx);
        }
    }

    // Build Voronoi edges
    let mut edges: Vec<VoronoiEdge<F>> = Vec::new();

    // Track which vertices belong to which site's cell
    let mut site_to_vertices: HashMap<usize, Vec<usize>> = HashMap::new();

    for (edge, tri_indices) in &edge_to_triangles {
        let site_a = edge.0;
        let site_b = edge.1;

        if tri_indices.len() == 2 {
            // Interior edge: connects two circumcenters
            let v1 = tri_indices[0];
            let v2 = tri_indices[1];
            edges.push(VoronoiEdge::Finite(v1, v2));

            // Both sites adjacent to this Delaunay edge get these Voronoi vertices
            site_to_vertices.entry(site_a).or_default().push(v1);
            site_to_vertices.entry(site_a).or_default().push(v2);
            site_to_vertices.entry(site_b).or_default().push(v1);
            site_to_vertices.entry(site_b).or_default().push(v2);
        } else if tri_indices.len() == 1 {
            // Boundary edge: creates an infinite ray
            let v = tri_indices[0];

            // Direction perpendicular to the Delaunay edge, pointing outward
            let pa = sites[site_a];
            let pb = sites[site_b];
            let mid = Point2::new((pa.x + pb.x) / F::from(2.0).unwrap(), (pa.y + pb.y) / F::from(2.0).unwrap());

            // Edge direction
            let edge_dir = Vec2::new(pb.x - pa.x, pb.y - pa.y);
            // Perpendicular (rotate 90 degrees)
            let perp = Vec2::new(-edge_dir.y, edge_dir.x);

            // Make sure it points away from the triangle's third vertex
            let tri = &triangles[v];
            let third = if tri.a != site_a && tri.a != site_b {
                tri.a
            } else if tri.b != site_a && tri.b != site_b {
                tri.b
            } else {
                tri.c
            };
            let third_pt = sites[third];

            // Vector from midpoint to third vertex
            let to_third = Vec2::new(third_pt.x - mid.x, third_pt.y - mid.y);

            // If perpendicular points toward third vertex, flip it
            let dir = if perp.x * to_third.x + perp.y * to_third.y > F::zero() {
                Vec2::new(-perp.x, -perp.y)
            } else {
                perp
            };

            // Normalize direction
            let len = (dir.x * dir.x + dir.y * dir.y).sqrt();
            let dir = if len > F::epsilon() {
                Vec2::new(dir.x / len, dir.y / len)
            } else {
                dir
            };

            edges.push(VoronoiEdge::Ray(v, dir));

            // Sites still get this vertex
            site_to_vertices.entry(site_a).or_default().push(v);
            site_to_vertices.entry(site_b).or_default().push(v);
        }
    }

    // Determine which sites are on the convex hull (have unbounded cells)
    let mut unbounded_sites: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (edge, tri_indices) in &edge_to_triangles {
        if tri_indices.len() == 1 {
            unbounded_sites.insert(edge.0);
            unbounded_sites.insert(edge.1);
        }
    }

    // Build cells for each site
    let mut cells: Vec<VoronoiCell> = Vec::with_capacity(sites.len());

    for site in 0..sites.len() {
        let vertex_set = site_to_vertices.get(&site);
        let mut cell_vertices: Vec<usize> = match vertex_set {
            Some(v) => {
                // Deduplicate
                let mut unique: Vec<usize> = v.clone();
                unique.sort_unstable();
                unique.dedup();
                unique
            }
            None => Vec::new(),
        };

        // Sort vertices in CCW order around the site
        if cell_vertices.len() > 1 {
            let site_pt = sites[site];
            cell_vertices.sort_by(|&a, &b| {
                let va = vertices[a];
                let vb = vertices[b];
                let angle_a = (va.y - site_pt.y).atan2(va.x - site_pt.x);
                let angle_b = (vb.y - site_pt.y).atan2(vb.x - site_pt.x);
                angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        cells.push(VoronoiCell {
            site,
            vertices: cell_vertices,
            unbounded: unbounded_sites.contains(&site),
        });
    }

    VoronoiDiagram {
        vertices,
        edges,
        cells,
    }
}

/// Computes the circumcenter of a triangle defined by three points.
///
/// # Example
///
/// ```
/// use approxum::triangulation::triangle_circumcenter;
/// use approxum::Point2;
///
/// // Right triangle
/// let a = Point2::new(0.0_f64, 0.0);
/// let b = Point2::new(1.0, 0.0);
/// let c = Point2::new(0.0, 1.0);
///
/// let center = triangle_circumcenter(a, b, c);
///
/// // Circumcenter of right triangle is at midpoint of hypotenuse
/// assert!((center.x - 0.5).abs() < 1e-10);
/// assert!((center.y - 0.5).abs() < 1e-10);
/// ```
pub fn triangle_circumcenter<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> Point2<F> {
    circumcenter(a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_circumcenter_equilateral() {
        // Equilateral triangle centered at origin
        // Using exact value: sqrt(3)/2 ≈ 0.8660254037844386
        let sqrt3_2 = 3.0_f64.sqrt() / 2.0;
        let a = Point2::new(0.0_f64, 1.0);
        let b = Point2::new(-sqrt3_2, -0.5);
        let c = Point2::new(sqrt3_2, -0.5);

        let center = circumcenter(a, b, c);

        // Circumcenter should be at origin
        assert!(approx_eq(center.x, 0.0, 1e-10));
        assert!(approx_eq(center.y, 0.0, 1e-10));
    }

    #[test]
    fn test_circumcenter_right_triangle() {
        // Right triangle
        let a = Point2::new(0.0_f64, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.0, 1.0);

        let center = circumcenter(a, b, c);

        // Circumcenter is at midpoint of hypotenuse
        assert!(approx_eq(center.x, 0.5, 1e-10));
        assert!(approx_eq(center.y, 0.5, 1e-10));
    }

    #[test]
    fn test_voronoi_empty() {
        let sites: Vec<Point2<f64>> = vec![];
        let voronoi = voronoi_diagram(&sites);
        assert!(voronoi.vertices.is_empty());
        assert!(voronoi.edges.is_empty());
        assert!(voronoi.cells.is_empty());
    }

    #[test]
    fn test_voronoi_one_point() {
        let sites = vec![Point2::new(0.0_f64, 0.0)];
        let voronoi = voronoi_diagram(&sites);
        assert!(voronoi.vertices.is_empty());
        assert_eq!(voronoi.cells.len(), 1);
        assert!(voronoi.cells[0].unbounded);
    }

    #[test]
    fn test_voronoi_two_points() {
        let sites = vec![Point2::new(0.0_f64, 0.0), Point2::new(1.0, 0.0)];
        let voronoi = voronoi_diagram(&sites);
        assert!(voronoi.vertices.is_empty());
        assert_eq!(voronoi.cells.len(), 2);
        assert!(voronoi.cells[0].unbounded);
        assert!(voronoi.cells[1].unbounded);
    }

    #[test]
    fn test_voronoi_three_points() {
        let sites = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let voronoi = voronoi_diagram(&sites);

        // One triangle = one circumcenter = one Voronoi vertex
        assert_eq!(voronoi.vertices.len(), 1);

        // Three cells
        assert_eq!(voronoi.cells.len(), 3);

        // All cells are unbounded (all sites on convex hull)
        for cell in &voronoi.cells {
            assert!(cell.unbounded);
        }
    }

    #[test]
    fn test_voronoi_square() {
        // Square of points
        let sites = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let voronoi = voronoi_diagram(&sites);

        // Square produces 2 triangles, so 2 Voronoi vertices
        // But for a square, both circumcenters are at (0.5, 0.5)
        assert!(!voronoi.vertices.is_empty());

        // Four cells
        assert_eq!(voronoi.cells.len(), 4);
    }

    #[test]
    fn test_voronoi_with_interior_point() {
        // Square with center point
        let sites = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.5, 0.5),
        ];
        let voronoi = voronoi_diagram(&sites);

        // Five cells
        assert_eq!(voronoi.cells.len(), 5);

        // Center point should have a bounded cell
        let center_cell = &voronoi.cells[4];
        assert!(!center_cell.unbounded);
        assert!(!center_cell.vertices.is_empty());
    }

    #[test]
    fn test_voronoi_vertex_equidistant() {
        // Verify Voronoi vertices are equidistant from their generating sites
        let sites = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let voronoi = voronoi_diagram(&sites);

        // The single Voronoi vertex should be equidistant from all three sites
        let v = voronoi.vertices[0];
        let d0 = ((v.x - sites[0].x).powi(2) + (v.y - sites[0].y).powi(2)).sqrt();
        let d1 = ((v.x - sites[1].x).powi(2) + (v.y - sites[1].y).powi(2)).sqrt();
        let d2 = ((v.x - sites[2].x).powi(2) + (v.y - sites[2].y).powi(2)).sqrt();

        assert!(approx_eq(d0, d1, 1e-10));
        assert!(approx_eq(d1, d2, 1e-10));
    }

    #[test]
    fn test_voronoi_collinear() {
        // Collinear points
        let sites = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];
        let voronoi = voronoi_diagram(&sites);

        // No triangles possible, so no Voronoi vertices
        assert!(voronoi.vertices.is_empty());
        assert_eq!(voronoi.cells.len(), 3);
    }

    #[test]
    fn test_voronoi_has_rays() {
        let sites = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let voronoi = voronoi_diagram(&sites);

        // Should have infinite rays for the convex hull edges
        let ray_count = voronoi
            .edges
            .iter()
            .filter(|e| matches!(e, VoronoiEdge::Ray(_, _)))
            .count();

        // A triangle has 3 edges, all on the hull, so 3 rays
        assert_eq!(ray_count, 3);
    }

    #[test]
    fn test_voronoi_finite_edges() {
        // Pentagon with center - should have some finite edges
        let sites = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.5, 1.5),
            Point2::new(1.0, 2.5),
            Point2::new(-0.5, 1.5),
            Point2::new(1.0, 1.0), // Interior point
        ];
        let voronoi = voronoi_diagram(&sites);

        let finite_count = voronoi
            .edges
            .iter()
            .filter(|e| matches!(e, VoronoiEdge::Finite(_, _)))
            .count();

        // Should have some finite edges connecting interior Voronoi vertices
        assert!(finite_count > 0);
    }

    #[test]
    fn test_voronoi_f32() {
        let sites: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let voronoi = voronoi_diagram(&sites);
        assert_eq!(voronoi.vertices.len(), 1);
        assert_eq!(voronoi.cells.len(), 3);
    }

    #[test]
    fn test_voronoi_cell_site_indices() {
        let sites = vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let voronoi = voronoi_diagram(&sites);

        // Each cell should reference its corresponding site
        for (i, cell) in voronoi.cells.iter().enumerate() {
            assert_eq!(cell.site, i);
        }
    }

    #[test]
    fn test_triangle_circumcenter_public() {
        let a = Point2::new(0.0_f64, 0.0);
        let b = Point2::new(2.0, 0.0);
        let c = Point2::new(1.0, 1.0);

        let center = triangle_circumcenter(a, b, c);

        // Verify it's equidistant from all vertices
        let da = ((center.x - a.x).powi(2) + (center.y - a.y).powi(2)).sqrt();
        let db = ((center.x - b.x).powi(2) + (center.y - b.y).powi(2)).sqrt();
        let dc = ((center.x - c.x).powi(2) + (center.y - c.y).powi(2)).sqrt();

        assert!(approx_eq(da, db, 1e-10));
        assert!(approx_eq(db, dc, 1e-10));
    }
}
