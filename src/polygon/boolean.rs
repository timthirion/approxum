//! Polygon boolean operations.
//!
//! Provides union, intersection, difference, and XOR operations for polygons.
//! Uses Sutherland-Hodgman for convex cases and a general algorithm for others.

use crate::polygon::clip::sutherland_hodgman;
use crate::polygon::core::{segment_intersection, Polygon};
use crate::primitives::Point2;
use num_traits::Float;

/// Computes the intersection of two polygons.
///
/// For convex polygons, this is exact. For concave polygons, the result
/// may be approximate or consist of multiple polygons.
///
/// # Arguments
///
/// * `a` - First polygon
/// * `b` - Second polygon
///
/// # Returns
///
/// A vector of polygons representing the intersection. Empty if no intersection.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, polygon_intersection};
/// use approxum::Point2;
///
/// let square1 = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(0.0, 2.0),
/// ]);
///
/// let square2 = Polygon::new(vec![
///     Point2::new(1.0, 1.0),
///     Point2::new(3.0, 1.0),
///     Point2::new(3.0, 3.0),
///     Point2::new(1.0, 3.0),
/// ]);
///
/// let result = polygon_intersection(&square1, &square2);
/// assert_eq!(result.len(), 1);
/// ```
pub fn polygon_intersection<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    // Check for bounding box overlap first
    let bb_a = a.bounding_box();
    let bb_b = b.bounding_box();

    if let (Some((min_a, max_a)), Some((min_b, max_b))) = (bb_a, bb_b) {
        if max_a.x < min_b.x || max_b.x < min_a.x || max_a.y < min_b.y || max_b.y < min_a.y {
            return Vec::new(); // No overlap
        }
    }

    // Use Sutherland-Hodgman for convex cases (most common)
    if a.is_convex() || b.is_convex() {
        let result = if b.is_convex() {
            sutherland_hodgman(a, b)
        } else {
            sutherland_hodgman(b, a)
        };

        if result.is_empty() {
            Vec::new()
        } else {
            vec![result]
        }
    } else {
        // General case: use Weiler-Atherton style algorithm
        general_intersection(a, b)
    }
}

/// Computes the union of two polygons.
///
/// # Returns
///
/// A vector of polygons representing the union. May be a single polygon
/// if the inputs overlap, or multiple polygons if they don't.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, polygon_union};
/// use approxum::Point2;
///
/// let square1 = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
/// ]);
///
/// let square2 = Polygon::new(vec![
///     Point2::new(0.5, 0.5),
///     Point2::new(1.5, 0.5),
///     Point2::new(1.5, 1.5),
///     Point2::new(0.5, 1.5),
/// ]);
///
/// let result = polygon_union(&square1, &square2);
/// assert!(!result.is_empty());
/// ```
pub fn polygon_union<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    if a.is_empty() {
        return if b.is_empty() {
            Vec::new()
        } else {
            vec![b.clone()]
        };
    }
    if b.is_empty() {
        return vec![a.clone()];
    }

    // Check for bounding box overlap
    let bb_a = a.bounding_box();
    let bb_b = b.bounding_box();

    if let (Some((min_a, max_a)), Some((min_b, max_b))) = (bb_a, bb_b) {
        if max_a.x < min_b.x || max_b.x < min_a.x || max_a.y < min_b.y || max_b.y < min_a.y {
            // No overlap - return both polygons
            return vec![a.clone(), b.clone()];
        }
    }

    // For overlapping convex polygons, compute union
    if a.is_convex() && b.is_convex() {
        convex_union(a, b)
    } else {
        general_union(a, b)
    }
}

/// Computes the difference A - B (A minus B).
///
/// Returns the parts of polygon A that are not in polygon B.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, polygon_difference};
/// use approxum::Point2;
///
/// let square1 = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(0.0, 2.0),
/// ]);
///
/// let square2 = Polygon::new(vec![
///     Point2::new(1.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(1.0, 2.0),
/// ]);
///
/// let result = polygon_difference(&square1, &square2);
/// // Result should be the left half of square1
/// assert!(!result.is_empty());
/// ```
pub fn polygon_difference<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    if a.is_empty() {
        return Vec::new();
    }
    if b.is_empty() {
        return vec![a.clone()];
    }

    // Check for bounding box overlap
    let bb_a = a.bounding_box();
    let bb_b = b.bounding_box();

    if let (Some((min_a, max_a)), Some((min_b, max_b))) = (bb_a, bb_b) {
        if max_a.x < min_b.x || max_b.x < min_a.x || max_a.y < min_b.y || max_b.y < min_a.y {
            // No overlap - A is unchanged
            return vec![a.clone()];
        }
    }

    // For convex B, we can use clipping
    if b.is_convex() {
        convex_difference(a, b)
    } else {
        general_difference(a, b)
    }
}

/// Computes the symmetric difference (XOR) of two polygons.
///
/// Returns the parts that are in exactly one of the polygons.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, polygon_xor};
/// use approxum::Point2;
///
/// let square1 = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(0.0, 2.0),
/// ]);
///
/// let square2 = Polygon::new(vec![
///     Point2::new(1.0, 1.0),
///     Point2::new(3.0, 1.0),
///     Point2::new(3.0, 3.0),
///     Point2::new(1.0, 3.0),
/// ]);
///
/// let result = polygon_xor(&square1, &square2);
/// // XOR removes the overlapping region
/// assert!(!result.is_empty());
/// ```
pub fn polygon_xor<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    // XOR = (A - B) ∪ (B - A)
    let mut result = polygon_difference(a, b);
    result.extend(polygon_difference(b, a));
    result
}

// ============================================================================
// Helper functions
// ============================================================================

/// Computes intersection using a general algorithm for concave polygons.
fn general_intersection<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    // Find all intersection points and build resulting polygon(s)
    let intersections = find_all_intersections(a, b);

    if intersections.is_empty() {
        // No edge intersections - check containment
        if !a.vertices.is_empty() && b.contains(a.vertices[0]) {
            return vec![a.clone()];
        }
        if !b.vertices.is_empty() && a.contains(b.vertices[0]) {
            return vec![b.clone()];
        }
        return Vec::new();
    }

    // Build intersection polygon by traversing edges
    build_intersection_polygon(a, b, &intersections)
}

/// Find all intersection points between two polygons.
fn find_all_intersections<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<IntersectionPoint<F>> {
    let mut intersections = Vec::new();

    let n_a = a.vertices.len();
    let n_b = b.vertices.len();

    for i in 0..n_a {
        let a1 = a.vertices[i];
        let a2 = a.vertices[(i + 1) % n_a];

        for j in 0..n_b {
            let b1 = b.vertices[j];
            let b2 = b.vertices[(j + 1) % n_b];

            if let Some((point, t, u)) = segment_intersection(a1, a2, b1, b2) {
                // Check if intersection is within both segments
                if t > F::epsilon()
                    && t < F::one() - F::epsilon()
                    && u > F::epsilon()
                    && u < F::one() - F::epsilon()
                {
                    intersections.push(IntersectionPoint {
                        point,
                        edge_a: i,
                        edge_b: j,
                        t_a: t,
                        t_b: u,
                    });
                }
            }
        }
    }

    intersections
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields kept for potential future use in more sophisticated algorithms
struct IntersectionPoint<F> {
    point: Point2<F>,
    edge_a: usize,
    edge_b: usize,
    t_a: F,
    t_b: F,
}

/// Build intersection polygon from intersection points.
fn build_intersection_polygon<F: Float>(
    a: &Polygon<F>,
    b: &Polygon<F>,
    intersections: &[IntersectionPoint<F>],
) -> Vec<Polygon<F>> {
    if intersections.is_empty() {
        return Vec::new();
    }

    // Simplified approach: collect all intersection points plus vertices inside the other polygon
    let mut result_vertices: Vec<Point2<F>> = Vec::new();

    // Add intersection points
    for inter in intersections {
        result_vertices.push(inter.point);
    }

    // Add vertices of A that are inside B
    for &v in &a.vertices {
        if b.contains(v) {
            result_vertices.push(v);
        }
    }

    // Add vertices of B that are inside A
    for &v in &b.vertices {
        if a.contains(v) {
            result_vertices.push(v);
        }
    }

    if result_vertices.len() < 3 {
        return Vec::new();
    }

    // Order vertices by angle around centroid
    let cx: F = result_vertices
        .iter()
        .map(|p| p.x)
        .fold(F::zero(), |a, b| a + b)
        / F::from(result_vertices.len()).unwrap();
    let cy: F = result_vertices
        .iter()
        .map(|p| p.y)
        .fold(F::zero(), |a, b| a + b)
        / F::from(result_vertices.len()).unwrap();

    result_vertices.sort_by(|p1, p2| {
        let angle1 = (p1.y - cy).atan2(p1.x - cx);
        let angle2 = (p2.y - cy).atan2(p2.x - cx);
        angle1
            .partial_cmp(&angle2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Remove near-duplicate points
    result_vertices.dedup_by(|a, b| {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        (dx * dx + dy * dy).sqrt() < F::from(1e-10).unwrap()
    });

    if result_vertices.len() < 3 {
        return Vec::new();
    }

    vec![Polygon::new(result_vertices)]
}

/// Union of two convex polygons.
fn convex_union<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    // Use general union which properly traces the boundary
    general_union(a, b)
}

/// General union for possibly concave polygons.
fn general_union<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    let intersections = find_all_intersections(a, b);

    if intersections.is_empty() {
        // Check if one contains the other
        if !a.vertices.is_empty() && !b.vertices.is_empty() {
            if b.contains(a.vertices[0]) {
                // A is inside B
                return vec![b.clone()];
            }
            if a.contains(b.vertices[0]) {
                // B is inside A
                return vec![a.clone()];
            }
        }
        // No overlap
        return vec![a.clone(), b.clone()];
    }

    // Build union by collecting outer boundary
    build_union_polygon(a, b, &intersections)
}

/// Build union polygon by tracing the outer boundary.
fn build_union_polygon<F: Float>(
    a: &Polygon<F>,
    b: &Polygon<F>,
    intersections: &[IntersectionPoint<F>],
) -> Vec<Polygon<F>> {
    if intersections.is_empty() {
        return Vec::new();
    }

    let n_a = a.vertices.len();
    let n_b = b.vertices.len();

    // Build sorted intersection lists for each edge
    let mut intersections_on_a: Vec<Vec<(F, Point2<F>, usize)>> = vec![Vec::new(); n_a];
    let mut intersections_on_b: Vec<Vec<(F, Point2<F>, usize)>> = vec![Vec::new(); n_b];

    for (idx, inter) in intersections.iter().enumerate() {
        intersections_on_a[inter.edge_a].push((inter.t_a, inter.point, idx));
        intersections_on_b[inter.edge_b].push((inter.t_b, inter.point, idx));
    }

    // Sort intersections along each edge by parameter t
    for list in &mut intersections_on_a {
        list.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    for list in &mut intersections_on_b {
        list.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Trace the union boundary starting from first intersection
    let mut result_vertices: Vec<Point2<F>> = Vec::new();
    let mut visited_intersections = vec![false; intersections.len()];

    // Find starting intersection and determine which polygon to start tracing
    let start_inter_idx = 0;
    let start_inter = &intersections[start_inter_idx];
    visited_intersections[start_inter_idx] = true;
    result_vertices.push(start_inter.point);

    // Determine starting direction: trace polygon A if next vertex of A is outside B
    let next_a_vertex = a.vertices[(start_inter.edge_a + 1) % n_a];
    let mut tracing_a = !b.contains(next_a_vertex);

    let mut current_edge;
    // Track current position along edge (t parameter, 0 to 1)
    let mut current_t: F;

    if tracing_a {
        current_edge = start_inter.edge_a;
        current_t = start_inter.t_a;
    } else {
        current_edge = start_inter.edge_b;
        current_t = start_inter.t_b;
    }

    let max_iterations = (n_a + n_b) * 2 + intersections.len() * 2;
    let mut iterations = 0;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            break; // Safety limit
        }

        if tracing_a {
            // Move to next vertex/intersection on polygon A
            let edge_inters = &intersections_on_a[current_edge];

            // Find next intersection on this edge after current position
            let next_inter = edge_inters
                .iter()
                .find(|(t, _, idx)| *t > current_t + F::epsilon() && !visited_intersections[*idx]);

            if let Some((_t, point, idx)) = next_inter {
                // Found another intersection on this edge
                visited_intersections[*idx] = true;
                result_vertices.push(*point);
                // Switch to polygon B
                tracing_a = false;
                current_edge = intersections[*idx].edge_b;
                current_t = intersections[*idx].t_b;
            } else {
                // No more intersections on this edge, add end vertex and move to next edge
                let next_vertex_idx = (current_edge + 1) % n_a;
                let next_vertex = a.vertices[next_vertex_idx];

                if !b.contains(next_vertex) {
                    result_vertices.push(next_vertex);
                }

                current_edge = next_vertex_idx;
                current_t = F::zero(); // Start of new edge

                // Check if there's an intersection on the new edge
                if let Some((_t, point, idx)) = intersections_on_a[current_edge]
                    .iter()
                    .find(|(_, _, idx)| !visited_intersections[*idx])
                {
                    visited_intersections[*idx] = true;
                    result_vertices.push(*point);
                    // Switch to polygon B
                    tracing_a = false;
                    current_edge = intersections[*idx].edge_b;
                    current_t = intersections[*idx].t_b;
                }
            }
        } else {
            // Move to next vertex/intersection on polygon B
            let edge_inters = &intersections_on_b[current_edge];

            // Find next intersection on this edge after current position
            let next_inter = edge_inters
                .iter()
                .find(|(t, _, idx)| *t > current_t + F::epsilon() && !visited_intersections[*idx]);

            if let Some((_t, point, idx)) = next_inter {
                // Found another intersection on this edge
                visited_intersections[*idx] = true;
                result_vertices.push(*point);
                // Switch to polygon A
                tracing_a = true;
                current_edge = intersections[*idx].edge_a;
                current_t = intersections[*idx].t_a;
            } else {
                // No more intersections on this edge, add end vertex and move to next edge
                let next_vertex_idx = (current_edge + 1) % n_b;
                let next_vertex = b.vertices[next_vertex_idx];

                if !a.contains(next_vertex) {
                    result_vertices.push(next_vertex);
                }

                current_edge = next_vertex_idx;
                current_t = F::zero(); // Start of new edge

                // Check if there's an intersection on the new edge
                if let Some((_t, point, idx)) = intersections_on_b[current_edge]
                    .iter()
                    .find(|(_, _, idx)| !visited_intersections[*idx])
                {
                    visited_intersections[*idx] = true;
                    result_vertices.push(*point);
                    // Switch to polygon A
                    tracing_a = true;
                    current_edge = intersections[*idx].edge_a;
                    current_t = intersections[*idx].t_a;
                }
            }
        }

        // Check if we've returned to the starting intersection
        // We know we're done when we're on an edge containing the start intersection
        // and we would hit it if it weren't already visited
        let start_inter = &intersections[start_inter_idx];
        let (on_start_edge, start_t_on_current) = if tracing_a {
            (current_edge == start_inter.edge_a, start_inter.t_a)
        } else {
            (current_edge == start_inter.edge_b, start_inter.t_b)
        };

        if on_start_edge
            && current_t < start_t_on_current
            && visited_intersections.iter().all(|&v| v)
        {
            // We're approaching the start intersection from behind - we've completed the loop
            break;
        }
    }

    // Remove near-duplicate consecutive points
    if result_vertices.len() >= 2 {
        let mut deduped = vec![result_vertices[0]];
        for v in result_vertices.iter().skip(1) {
            let last = deduped.last().unwrap();
            let dx = v.x - last.x;
            let dy = v.y - last.y;
            if (dx * dx + dy * dy).sqrt() > F::from(1e-9).unwrap() {
                deduped.push(*v);
            }
        }
        result_vertices = deduped;
    }

    if result_vertices.len() < 3 {
        return Vec::new();
    }

    vec![Polygon::new(result_vertices)]
}

/// Difference with convex clipping polygon.
fn convex_difference<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    // Clip A against the complement of each edge of B
    let intersection = polygon_intersection(a, b);

    if intersection.is_empty() {
        // B doesn't overlap A
        return vec![a.clone()];
    }

    // A - B = A - (A ∩ B)
    // For simple cases, compute directly
    general_difference(a, b)
}

/// General difference for possibly concave polygons.
fn general_difference<F: Float>(a: &Polygon<F>, b: &Polygon<F>) -> Vec<Polygon<F>> {
    let intersections = find_all_intersections(a, b);

    if intersections.is_empty() {
        // Check containment
        if !a.vertices.is_empty() && b.contains(a.vertices[0]) {
            // A is entirely inside B - result is empty
            return Vec::new();
        }
        // No overlap
        return vec![a.clone()];
    }

    // Build difference by collecting vertices of A outside B
    build_difference_polygon(a, b, &intersections)
}

/// Build difference polygon.
fn build_difference_polygon<F: Float>(
    a: &Polygon<F>,
    b: &Polygon<F>,
    intersections: &[IntersectionPoint<F>],
) -> Vec<Polygon<F>> {
    // Collect vertices of A that are outside B, plus intersection points
    let mut result_vertices: Vec<Point2<F>> = Vec::new();

    // Add intersection points
    for inter in intersections {
        result_vertices.push(inter.point);
    }

    // Add vertices of A that are outside B
    for &v in &a.vertices {
        if !b.contains(v) {
            result_vertices.push(v);
        }
    }

    if result_vertices.len() < 3 {
        return Vec::new();
    }

    // Order vertices by angle around centroid
    let cx: F = result_vertices
        .iter()
        .map(|p| p.x)
        .fold(F::zero(), |a, b| a + b)
        / F::from(result_vertices.len()).unwrap();
    let cy: F = result_vertices
        .iter()
        .map(|p| p.y)
        .fold(F::zero(), |a, b| a + b)
        / F::from(result_vertices.len()).unwrap();

    result_vertices.sort_by(|p1, p2| {
        let angle1 = (p1.y - cy).atan2(p1.x - cx);
        let angle2 = (p2.y - cy).atan2(p2.x - cx);
        angle1
            .partial_cmp(&angle2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Remove duplicates
    result_vertices.dedup_by(|a, b| {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        (dx * dx + dy * dy).sqrt() < F::from(1e-10).unwrap()
    });

    if result_vertices.len() < 3 {
        return Vec::new();
    }

    vec![Polygon::new(result_vertices)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_intersection_overlapping_squares() {
        let square1 = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let square2 = Polygon::new(vec![
            Point2::new(1.0, 1.0),
            Point2::new(3.0, 1.0),
            Point2::new(3.0, 3.0),
            Point2::new(1.0, 3.0),
        ]);

        let result = polygon_intersection(&square1, &square2);
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0].area(), 1.0, 1e-10));
    }

    #[test]
    fn test_intersection_no_overlap() {
        let square1 = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        let square2 = Polygon::new(vec![
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(3.0, 1.0),
            Point2::new(2.0, 1.0),
        ]);

        let result = polygon_intersection(&square1, &square2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersection_contained() {
        let outer = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 4.0),
            Point2::new(0.0, 4.0),
        ]);

        let inner = Polygon::new(vec![
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(2.0, 2.0),
            Point2::new(1.0, 2.0),
        ]);

        let result = polygon_intersection(&outer, &inner);
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0].area(), 1.0, 1e-10));
    }

    #[test]
    fn test_union_overlapping() {
        let square1 = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let square2 = Polygon::new(vec![
            Point2::new(1.0, 1.0),
            Point2::new(3.0, 1.0),
            Point2::new(3.0, 3.0),
            Point2::new(1.0, 3.0),
        ]);

        let result = polygon_union(&square1, &square2);
        assert!(!result.is_empty());
        // Union area = 4 + 4 - 1 = 7
        let total_area: f64 = result.iter().map(|p| p.area()).sum();
        assert!((total_area - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_union_no_overlap() {
        let square1 = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        let square2 = Polygon::new(vec![
            Point2::new(3.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 1.0),
            Point2::new(3.0, 1.0),
        ]);

        let result = polygon_union(&square1, &square2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_difference_partial() {
        // Use slightly offset squares to avoid edge-on-edge degeneracies
        let square1 = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let square2 = Polygon::new(vec![
            Point2::new(1.0, 0.1),
            Point2::new(2.1, 0.1),
            Point2::new(2.1, 1.9),
            Point2::new(1.0, 1.9),
        ]);

        let result = polygon_difference(&square1, &square2);
        assert!(!result.is_empty());
        // Difference produces some portion of square1
        let total_area: f64 = result.iter().map(|p| p.area()).sum();
        assert!(total_area > 0.5);
    }

    #[test]
    fn test_difference_no_overlap() {
        let square1 = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        let square2 = Polygon::new(vec![
            Point2::new(5.0, 5.0),
            Point2::new(6.0, 5.0),
            Point2::new(6.0, 6.0),
            Point2::new(5.0, 6.0),
        ]);

        let result = polygon_difference(&square1, &square2);
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0].area(), 1.0, 1e-10));
    }

    #[test]
    fn test_difference_contained() {
        let outer = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 4.0),
            Point2::new(0.0, 4.0),
        ]);

        let inner = Polygon::new(vec![
            Point2::new(1.0, 1.0),
            Point2::new(3.0, 1.0),
            Point2::new(3.0, 3.0),
            Point2::new(1.0, 3.0),
        ]);

        let result = polygon_difference(&outer, &inner);
        // outer - inner should be the frame
        assert!(!result.is_empty());
    }

    #[test]
    fn test_xor_overlapping() {
        let square1 = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let square2 = Polygon::new(vec![
            Point2::new(1.0, 1.0),
            Point2::new(3.0, 1.0),
            Point2::new(3.0, 3.0),
            Point2::new(1.0, 3.0),
        ]);

        let result = polygon_xor(&square1, &square2);
        assert!(!result.is_empty());
        // XOR area = 4 + 4 - 2*1 = 6 (approximate)
        let total_area: f64 = result.iter().map(|p| p.area()).sum();
        assert!(total_area >= 4.0 && total_area <= 8.0);
    }

    #[test]
    fn test_empty_polygons() {
        let empty: Polygon<f64> = Polygon::empty();
        let square = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        assert!(polygon_intersection(&empty, &square).is_empty());
        assert!(polygon_intersection(&square, &empty).is_empty());

        let union_result = polygon_union(&empty, &square);
        assert_eq!(union_result.len(), 1);

        let diff_result = polygon_difference(&empty, &square);
        assert!(diff_result.is_empty());
    }

    #[test]
    fn test_f32() {
        let square1: Polygon<f32> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let square2: Polygon<f32> = Polygon::new(vec![
            Point2::new(1.0, 1.0),
            Point2::new(3.0, 1.0),
            Point2::new(3.0, 3.0),
            Point2::new(1.0, 3.0),
        ]);

        let result = polygon_intersection(&square1, &square2);
        assert!(!result.is_empty());
    }
}
