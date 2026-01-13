//! Medial axis and straight skeleton computation.
//!
//! The **medial axis** of a polygon is the set of all points inside the polygon
//! that have more than one closest point on the boundary. It forms a tree-like
//! structure useful for shape analysis.
//!
//! The **straight skeleton** is a variant that produces straight edges by
//! propagating the polygon boundary inward at constant speed.
//!
//! # Example
//!
//! ```
//! use approxum::polygon::{Polygon, medial_axis};
//! use approxum::Point2;
//!
//! let rectangle = Polygon::new(vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(4.0, 0.0),
//!     Point2::new(4.0, 2.0),
//!     Point2::new(0.0, 2.0),
//! ]);
//!
//! let axis = medial_axis(&rectangle, 0.1);
//! // The medial axis of a rectangle is a horizontal line segment
//! ```

use super::core::Polygon;
use crate::primitives::Point2;
use num_traits::Float;

/// A node in the medial axis or skeleton.
#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonNode<F> {
    /// Position of the node
    pub point: Point2<F>,
    /// Distance to the nearest boundary point (radius of inscribed circle)
    pub radius: F,
}

/// An edge in the medial axis or skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkeletonEdge {
    /// Start node index
    pub start: usize,
    /// End node index
    pub end: usize,
}

/// Result of medial axis or skeleton computation.
#[derive(Debug, Clone)]
pub struct Skeleton<F> {
    /// Nodes (vertices) of the skeleton
    pub nodes: Vec<SkeletonNode<F>>,
    /// Edges connecting nodes
    pub edges: Vec<SkeletonEdge>,
}

impl<F: Float> Skeleton<F> {
    /// Creates an empty skeleton.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Returns the total length of all skeleton edges.
    pub fn total_length(&self) -> F {
        let mut length = F::zero();
        for edge in &self.edges {
            let p1 = self.nodes[edge.start].point;
            let p2 = self.nodes[edge.end].point;
            length = length + p1.distance(p2);
        }
        length
    }

    /// Returns skeleton edges as line segments.
    pub fn to_segments(&self) -> Vec<(Point2<F>, Point2<F>)> {
        self.edges
            .iter()
            .map(|e| (self.nodes[e.start].point, self.nodes[e.end].point))
            .collect()
    }

    /// Prunes short branches from the skeleton.
    ///
    /// Removes branches shorter than the given threshold.
    pub fn prune(&mut self, min_length: F) {
        // Find leaf nodes (degree 1)
        let mut degrees = vec![0usize; self.nodes.len()];
        for edge in &self.edges {
            degrees[edge.start] += 1;
            degrees[edge.end] += 1;
        }

        // Iteratively remove short leaf branches
        let mut changed = true;
        while changed {
            changed = false;

            // Recalculate degrees
            degrees.fill(0);
            for edge in &self.edges {
                degrees[edge.start] += 1;
                degrees[edge.end] += 1;
            }

            // Find edges to remove
            let mut to_remove = Vec::new();
            for (i, edge) in self.edges.iter().enumerate() {
                let is_leaf_edge =
                    degrees[edge.start] == 1 || degrees[edge.end] == 1;

                if is_leaf_edge {
                    let p1 = self.nodes[edge.start].point;
                    let p2 = self.nodes[edge.end].point;
                    let length = p1.distance(p2);

                    if length < min_length {
                        to_remove.push(i);
                        changed = true;
                    }
                }
            }

            // Remove edges in reverse order
            for i in to_remove.into_iter().rev() {
                self.edges.remove(i);
            }
        }
    }
}

impl<F: Float> Default for Skeleton<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes an approximate medial axis of a polygon.
///
/// Uses Voronoi diagram of sampled boundary points, filtered to keep
/// only edges inside the polygon.
///
/// # Arguments
///
/// * `polygon` - The input polygon (should be simple, non-self-intersecting)
/// * `sample_distance` - Distance between boundary sample points (smaller = more accurate)
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, medial_axis};
/// use approxum::Point2;
///
/// let square = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(0.0, 2.0),
/// ]);
///
/// let axis = medial_axis(&square, 0.2);
/// assert!(!axis.nodes.is_empty());
/// ```
pub fn medial_axis<F: Float>(polygon: &Polygon<F>, sample_distance: F) -> Skeleton<F> {
    if polygon.vertices.len() < 3 {
        return Skeleton::new();
    }

    // Sample points along boundary
    let boundary_points = sample_boundary(&polygon.vertices, sample_distance);
    if boundary_points.len() < 3 {
        return Skeleton::new();
    }

    // Compute Voronoi diagram
    let voronoi = compute_voronoi(&boundary_points);

    // Filter to keep only edges inside the polygon
    filter_voronoi_to_medial_axis(voronoi, polygon)
}

/// Computes an approximate straight skeleton of a polygon.
///
/// The straight skeleton is computed by shrinking the polygon inward
/// and tracking vertex paths.
///
/// # Arguments
///
/// * `polygon` - The input polygon
/// * `step_size` - Shrink step size (smaller = more accurate)
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, straight_skeleton};
/// use approxum::Point2;
///
/// let triangle = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(1.0, 2.0),
/// ]);
///
/// let skeleton = straight_skeleton(&triangle, 0.1);
/// // Triangle skeleton converges to a single point
/// ```
pub fn straight_skeleton<F: Float>(polygon: &Polygon<F>, step_size: F) -> Skeleton<F> {
    if polygon.vertices.len() < 3 {
        return Skeleton::new();
    }

    // Use wavefront propagation approach
    wavefront_skeleton(&polygon.vertices, step_size)
}

/// Samples points along the polygon boundary.
fn sample_boundary<F: Float>(vertices: &[Point2<F>], sample_distance: F) -> Vec<Point2<F>> {
    let mut points = Vec::new();
    let n = vertices.len();

    for i in 0..n {
        let p1 = vertices[i];
        let p2 = vertices[(i + 1) % n];

        let edge_len = p1.distance(p2);
        let num_samples = (edge_len / sample_distance).ceil().to_usize().unwrap().max(1);

        for j in 0..num_samples {
            let t = F::from(j).unwrap() / F::from(num_samples).unwrap();
            points.push(Point2::new(
                p1.x + t * (p2.x - p1.x),
                p1.y + t * (p2.y - p1.y),
            ));
        }
    }

    points
}

/// Simple Voronoi computation for medial axis.
fn compute_voronoi<F: Float>(points: &[Point2<F>]) -> Skeleton<F> {
    use crate::triangulation::{delaunay_triangulation, triangle_circumcenter};

    if points.len() < 3 {
        return Skeleton::new();
    }

    // Compute Delaunay triangulation
    let triangles = delaunay_triangulation(points);

    let mut skeleton = Skeleton::new();
    let mut node_map: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();

    let scale = F::from(1e6).unwrap();

    // Each triangle's circumcenter is a Voronoi vertex
    for tri in &triangles {
        let a = points[tri.a];
        let b = points[tri.b];
        let c = points[tri.c];

        let center = triangle_circumcenter(a, b, c);
        // Compute radius (distance to any vertex)
        let radius = center.distance(a);

        let key = (
            (center.x * scale).to_i64().unwrap_or(0),
            (center.y * scale).to_i64().unwrap_or(0),
        );

        if !node_map.contains_key(&key) {
            node_map.insert(key, skeleton.nodes.len());
            skeleton.nodes.push(SkeletonNode {
                point: center,
                radius,
            });
        }
    }

    // Connect adjacent triangles (Voronoi edges)
    for i in 0..triangles.len() {
        for j in (i + 1)..triangles.len() {
            if triangles_share_edge(&triangles[i], &triangles[j]) {
                let a = points[triangles[i].a];
                let b = points[triangles[i].b];
                let c = points[triangles[i].c];
                let d = points[triangles[j].a];
                let e = points[triangles[j].b];
                let f = points[triangles[j].c];

                let c1 = triangle_circumcenter(a, b, c);
                let c2 = triangle_circumcenter(d, e, f);

                let key1 = (
                    (c1.x * scale).to_i64().unwrap_or(0),
                    (c1.y * scale).to_i64().unwrap_or(0),
                );
                let key2 = (
                    (c2.x * scale).to_i64().unwrap_or(0),
                    (c2.y * scale).to_i64().unwrap_or(0),
                );

                if let (Some(&idx1), Some(&idx2)) = (node_map.get(&key1), node_map.get(&key2)) {
                    if idx1 != idx2 {
                        skeleton.edges.push(SkeletonEdge {
                            start: idx1,
                            end: idx2,
                        });
                    }
                }
            }
        }
    }

    skeleton
}

/// Checks if two triangles share an edge.
fn triangles_share_edge(t1: &crate::triangulation::Triangle, t2: &crate::triangulation::Triangle) -> bool {
    let v1 = [t1.a, t1.b, t1.c];
    let v2 = [t2.a, t2.b, t2.c];

    let mut shared = 0;
    for &a in &v1 {
        for &b in &v2 {
            if a == b {
                shared += 1;
            }
        }
    }
    shared == 2
}

/// Filters Voronoi diagram to keep only parts inside the polygon.
fn filter_voronoi_to_medial_axis<F: Float>(voronoi: Skeleton<F>, polygon: &Polygon<F>) -> Skeleton<F> {
    let mut result = Skeleton::new();
    let mut node_remap: Vec<Option<usize>> = vec![None; voronoi.nodes.len()];

    // Compute minimum radius threshold based on polygon size
    let min_radius = compute_min_medial_radius(polygon);

    // Keep only nodes inside the polygon with sufficient distance from boundary
    for (i, node) in voronoi.nodes.iter().enumerate() {
        if super::core::polygon_contains(&polygon.vertices, node.point) {
            // Check actual distance to boundary
            let dist = distance_to_polygon_boundary(node.point, &polygon.vertices);
            if dist >= min_radius {
                node_remap[i] = Some(result.nodes.len());
                result.nodes.push(SkeletonNode {
                    point: node.point,
                    radius: dist,  // Use actual distance, not circumradius
                });
            }
        }
    }

    // Keep only edges where both endpoints are inside
    for edge in &voronoi.edges {
        if let (Some(new_start), Some(new_end)) = (node_remap[edge.start], node_remap[edge.end]) {
            result.edges.push(SkeletonEdge {
                start: new_start,
                end: new_end,
            });
        }
    }

    result
}

/// Computes minimum radius for medial axis nodes.
fn compute_min_medial_radius<F: Float>(polygon: &Polygon<F>) -> F {
    // Use 10% of the minimum bounding box dimension
    let (min_x, max_x, min_y, max_y) = polygon.vertices.iter().fold(
        (F::infinity(), F::neg_infinity(), F::infinity(), F::neg_infinity()),
        |(min_x, max_x, min_y, max_y), p| {
            (min_x.min(p.x), max_x.max(p.x), min_y.min(p.y), max_y.max(p.y))
        },
    );
    let width = max_x - min_x;
    let height = max_y - min_y;
    width.min(height) * F::from(0.1).unwrap()
}

/// Computes distance from a point to the polygon boundary.
fn distance_to_polygon_boundary<F: Float>(point: Point2<F>, vertices: &[Point2<F>]) -> F {
    let n = vertices.len();
    let mut min_dist = F::infinity();

    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        let dist = point_to_segment_distance(point, a, b);
        min_dist = min_dist.min(dist);
    }

    min_dist
}

/// Distance from point to line segment.
fn point_to_segment_distance<F: Float>(p: Point2<F>, a: Point2<F>, b: Point2<F>) -> F {
    let ab = Point2::new(b.x - a.x, b.y - a.y);
    let ap = Point2::new(p.x - a.x, p.y - a.y);

    let ab_len_sq = ab.x * ab.x + ab.y * ab.y;
    if ab_len_sq < F::epsilon() {
        return p.distance(a);
    }

    let t = (ap.x * ab.x + ap.y * ab.y) / ab_len_sq;
    let t = t.max(F::zero()).min(F::one());

    let closest = Point2::new(a.x + t * ab.x, a.y + t * ab.y);
    p.distance(closest)
}

/// Wavefront-based straight skeleton approximation.
fn wavefront_skeleton<F: Float>(vertices: &[Point2<F>], step_size: F) -> Skeleton<F> {
    let mut skeleton = Skeleton::new();

    if vertices.len() < 3 {
        return skeleton;
    }

    // Track vertex positions as we shrink
    let mut current: Vec<Point2<F>> = vertices.to_vec();

    // Add initial corner nodes
    let mut vertex_node_idx: Vec<usize> = Vec::new();
    for v in vertices {
        vertex_node_idx.push(skeleton.nodes.len());
        skeleton.nodes.push(SkeletonNode {
            point: *v,
            radius: F::zero(),
        });
    }

    let mut total_offset = F::zero();
    let max_iterations = 1000;
    let mut iteration = 0;

    while current.len() >= 3 && iteration < max_iterations {
        iteration += 1;

        // Recompute bisectors for current polygon shape
        let normals = compute_vertex_bisectors(&current);

        // Move vertices inward
        let mut new_positions = Vec::with_capacity(current.len());
        for (i, pos) in current.iter().enumerate() {
            new_positions.push(Point2::new(
                pos.x + step_size * normals[i].x,
                pos.y + step_size * normals[i].y,
            ));
        }

        total_offset = total_offset + step_size;

        // Check for edge collapses
        let mut collapse_idx: Option<usize> = None;
        for i in 0..new_positions.len() {
            let next = (i + 1) % new_positions.len();
            let edge_len = new_positions[i].distance(new_positions[next]);

            if edge_len < step_size * F::from(0.5).unwrap() {
                collapse_idx = Some(i);
                break;
            }
        }

        // Handle collapse - add skeleton node and edges
        if let Some(collapse_i) = collapse_idx {
            let next_i = (collapse_i + 1) % new_positions.len();

            let collapse_point = Point2::new(
                (new_positions[collapse_i].x + new_positions[next_i].x) / (F::one() + F::one()),
                (new_positions[collapse_i].y + new_positions[next_i].y) / (F::one() + F::one()),
            );

            // Add skeleton node at collapse point
            let new_node_idx = skeleton.nodes.len();
            skeleton.nodes.push(SkeletonNode {
                point: collapse_point,
                radius: total_offset,
            });

            // Connect to the original corner nodes
            let idx1 = vertex_node_idx[collapse_i];
            let idx2 = vertex_node_idx[next_i];
            skeleton.edges.push(SkeletonEdge { start: idx1, end: new_node_idx });
            skeleton.edges.push(SkeletonEdge { start: idx2, end: new_node_idx });

            // Remove one vertex and update tracking
            new_positions.remove(next_i);
            vertex_node_idx.remove(next_i);
            if collapse_i < new_positions.len() {
                new_positions[collapse_i] = collapse_point;
                vertex_node_idx[collapse_i] = new_node_idx;
            }
        }

        // Check if polygon has collapsed to 2 or fewer vertices
        if new_positions.len() < 3 {
            // Add final center point if we have remaining vertices
            if new_positions.len() >= 2 {
                let center = polygon_centroid(&new_positions);
                let final_idx = skeleton.nodes.len();
                skeleton.nodes.push(SkeletonNode {
                    point: center,
                    radius: total_offset,
                });

                // Connect remaining vertices to center
                for &idx in &vertex_node_idx {
                    skeleton.edges.push(SkeletonEdge { start: idx, end: final_idx });
                }
            }
            break;
        }

        current = new_positions;
    }

    // If we maxed out iterations, add a center point
    if iteration >= max_iterations && current.len() >= 3 {
        let center = polygon_centroid(&current);
        let final_idx = skeleton.nodes.len();
        skeleton.nodes.push(SkeletonNode {
            point: center,
            radius: total_offset,
        });
        for &idx in &vertex_node_idx {
            skeleton.edges.push(SkeletonEdge { start: idx, end: final_idx });
        }
    }

    // Clean up duplicate edges
    skeleton.edges.sort_by(|a, b| {
        (a.start.min(a.end), a.start.max(a.end))
            .cmp(&(b.start.min(b.end), b.start.max(b.end)))
    });
    skeleton.edges.dedup_by(|a, b| {
        (a.start.min(a.end), a.start.max(a.end)) == (b.start.min(b.end), b.start.max(b.end))
    });

    skeleton
}

/// Computes inward bisector directions for each vertex.
fn compute_vertex_bisectors<F: Float>(vertices: &[Point2<F>]) -> Vec<Point2<F>> {
    let n = vertices.len();
    let mut bisectors = Vec::with_capacity(n);

    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = (i + 1) % n;

        let v1 = Point2::new(
            vertices[i].x - vertices[prev].x,
            vertices[i].y - vertices[prev].y,
        );
        let v2 = Point2::new(
            vertices[next].x - vertices[i].x,
            vertices[next].y - vertices[i].y,
        );

        // Normalize
        let len1 = (v1.x * v1.x + v1.y * v1.y).sqrt();
        let len2 = (v2.x * v2.x + v2.y * v2.y).sqrt();

        let (n1x, n1y) = if len1 > F::epsilon() {
            (v1.x / len1, v1.y / len1)
        } else {
            (F::zero(), F::zero())
        };

        let (n2x, n2y) = if len2 > F::epsilon() {
            (v2.x / len2, v2.y / len2)
        } else {
            (F::zero(), F::zero())
        };

        // Inward normals (rotate 90 degrees CW for CCW polygon)
        let in1 = Point2::new(n1y, -n1x);
        let in2 = Point2::new(n2y, -n2x);

        // Bisector is average of inward normals
        let bx = in1.x + in2.x;
        let by = in1.y + in2.y;
        let blen = (bx * bx + by * by).sqrt();

        if blen > F::epsilon() {
            bisectors.push(Point2::new(bx / blen, by / blen));
        } else {
            bisectors.push(Point2::new(F::zero(), F::zero()));
        }
    }

    bisectors
}

/// Computes centroid of a point set.
fn polygon_centroid<F: Float>(vertices: &[Point2<F>]) -> Point2<F> {
    if vertices.is_empty() {
        return Point2::new(F::zero(), F::zero());
    }

    let mut cx = F::zero();
    let mut cy = F::zero();
    let n = F::from(vertices.len()).unwrap();

    for v in vertices {
        cx = cx + v.x;
        cy = cy + v.y;
    }

    Point2::new(cx / n, cy / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square<F: Float>(size: F) -> Polygon<F> {
        Polygon::new(vec![
            Point2::new(F::zero(), F::zero()),
            Point2::new(size, F::zero()),
            Point2::new(size, size),
            Point2::new(F::zero(), size),
        ])
    }

    fn rectangle<F: Float>(width: F, height: F) -> Polygon<F> {
        Polygon::new(vec![
            Point2::new(F::zero(), F::zero()),
            Point2::new(width, F::zero()),
            Point2::new(width, height),
            Point2::new(F::zero(), height),
        ])
    }

    #[test]
    fn test_medial_axis_square() {
        let sq: Polygon<f64> = square(2.0);
        let axis = medial_axis(&sq, 0.2);

        // Should have some nodes
        assert!(!axis.nodes.is_empty());

        // All nodes should be inside the polygon
        for node in &axis.nodes {
            assert!(
                super::super::core::polygon_contains(&sq.vertices, node.point),
                "Node {:?} is outside polygon",
                node.point
            );
        }
    }

    #[test]
    fn test_medial_axis_rectangle() {
        let rect: Polygon<f64> = rectangle(4.0, 2.0);
        let axis = medial_axis(&rect, 0.15);

        assert!(!axis.nodes.is_empty());

        // Medial axis of rectangle should have nodes inside the polygon
        // with radius indicating distance to boundary
        for node in &axis.nodes {
            // All nodes should be inside polygon
            assert!(
                node.point.x > 0.0 && node.point.x < 4.0 &&
                node.point.y > 0.0 && node.point.y < 2.0,
                "Node {:?} is outside polygon",
                node.point
            );
            // Radius should be positive
            assert!(node.radius > 0.0, "Node should have positive radius");
        }
    }

    #[test]
    fn test_straight_skeleton_triangle() {
        let triangle: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 2.0),
        ]);

        let skeleton = straight_skeleton(&triangle, 0.1);

        // Should have some structure
        assert!(!skeleton.nodes.is_empty());
    }

    #[test]
    fn test_straight_skeleton_square() {
        let sq: Polygon<f64> = square(2.0);
        let skeleton = straight_skeleton(&sq, 0.05);

        assert!(!skeleton.nodes.is_empty());

        // Should have more nodes than just the 4 corners
        assert!(skeleton.nodes.len() > 4, "Should have interior skeleton nodes");

        // Should converge to center (with tolerance for approximate algorithm)
        let has_center_node = skeleton.nodes.iter().any(|n| {
            (n.point.x - 1.0).abs() < 0.5 && (n.point.y - 1.0).abs() < 0.5
        });
        assert!(has_center_node, "Should have node near center");
    }

    #[test]
    fn test_skeleton_total_length() {
        let sq: Polygon<f64> = square(2.0);
        let skeleton = straight_skeleton(&sq, 0.05);

        let length = skeleton.total_length();
        assert!(length > 0.0, "Skeleton should have positive total length");
    }

    #[test]
    fn test_skeleton_to_segments() {
        let sq: Polygon<f64> = square(2.0);
        let skeleton = straight_skeleton(&sq, 0.1);

        let segments = skeleton.to_segments();
        assert_eq!(segments.len(), skeleton.edges.len());
    }

    #[test]
    fn test_skeleton_prune() {
        let sq: Polygon<f64> = square(2.0);
        let mut skeleton = medial_axis(&sq, 0.2);

        let edges_before = skeleton.edges.len();
        skeleton.prune(0.5);
        // Pruning might remove some edges
        assert!(skeleton.edges.len() <= edges_before);
    }

    #[test]
    fn test_empty_polygon() {
        let empty: Polygon<f64> = Polygon::new(vec![]);
        let axis = medial_axis(&empty, 0.1);
        assert!(axis.nodes.is_empty());
    }

    #[test]
    fn test_sample_boundary() {
        let sq: Polygon<f64> = square(1.0);
        let samples = sample_boundary(&sq.vertices, 0.25);

        // Should have multiple samples per edge
        assert!(samples.len() >= 4);
    }

    #[test]
    fn test_l_shape_medial_axis() {
        let l_shape: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let axis = medial_axis(&l_shape, 0.15);

        assert!(!axis.nodes.is_empty());

        // All nodes should be inside
        for node in &axis.nodes {
            assert!(super::super::core::polygon_contains(
                &l_shape.vertices,
                node.point
            ));
        }
    }

    #[test]
    fn test_f32_support() {
        let sq: Polygon<f32> = square(2.0);
        let axis = medial_axis(&sq, 0.3);
        assert!(!axis.nodes.is_empty());
    }

    #[test]
    fn test_node_radius() {
        let sq: Polygon<f64> = square(2.0);
        let axis = medial_axis(&sq, 0.2);

        // Center node should have radius close to 1.0 (half the square size)
        let center_nodes: Vec<_> = axis
            .nodes
            .iter()
            .filter(|n| (n.point.x - 1.0).abs() < 0.3 && (n.point.y - 1.0).abs() < 0.3)
            .collect();

        if !center_nodes.is_empty() {
            let center = center_nodes[0];
            assert!(
                center.radius > 0.5 && center.radius < 1.5,
                "Center radius {} should be near 1.0",
                center.radius
            );
        }
    }
}
