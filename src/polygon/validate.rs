//! Polygon validation and self-intersection repair.
//!
//! Provides utilities for detecting and fixing self-intersecting polygons.
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, polygon::{Polygon, has_self_intersection, repair_self_intersections}};
//!
//! // A figure-8 polygon that crosses itself
//! let figure8 = Polygon::new(vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(2.0, 2.0),
//!     Point2::new(2.0, 0.0),
//!     Point2::new(0.0, 2.0),
//! ]);
//!
//! assert!(has_self_intersection(&figure8));
//!
//! let repaired = repair_self_intersections(&figure8, 1e-9);
//! assert_eq!(repaired.len(), 2); // Two triangles
//! ```

use super::core::Polygon;
use crate::primitives::Point2;
use num_traits::Float;

/// Information about a self-intersection.
#[derive(Debug, Clone)]
pub struct SelfIntersection<F> {
    /// The intersection point.
    pub point: Point2<F>,
    /// Index of the first edge (0-based).
    pub edge1: usize,
    /// Parameter along first edge (0 to 1).
    pub t1: F,
    /// Index of the second edge (0-based).
    pub edge2: usize,
    /// Parameter along second edge (0 to 1).
    pub t2: F,
}

/// Result of polygon validation.
#[derive(Debug, Clone)]
pub struct ValidationResult<F> {
    /// Whether the polygon is valid (no issues found).
    pub is_valid: bool,
    /// Whether the polygon has self-intersections.
    pub has_self_intersections: bool,
    /// List of self-intersection points.
    pub self_intersections: Vec<SelfIntersection<F>>,
    /// Whether the polygon has duplicate consecutive vertices.
    pub has_duplicate_vertices: bool,
    /// Whether the polygon has zero area.
    pub has_zero_area: bool,
    /// Whether the polygon has fewer than 3 vertices.
    pub is_degenerate: bool,
}

/// Checks if a polygon has any self-intersections.
///
/// Returns true if any non-adjacent edges cross each other.
///
/// # Example
///
/// ```
/// use approxum::{Point2, polygon::{Polygon, has_self_intersection}};
///
/// // Simple square - no self-intersection
/// let square = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
/// ]);
/// assert!(!has_self_intersection(&square));
///
/// // Figure-8 - has self-intersection
/// let figure8 = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(0.0, 2.0),
/// ]);
/// assert!(has_self_intersection(&figure8));
/// ```
pub fn has_self_intersection<F: Float>(polygon: &Polygon<F>) -> bool {
    let n = polygon.vertices.len();
    if n < 4 {
        return false; // Need at least 4 vertices for self-intersection
    }

    for i in 0..n {
        let i_next = (i + 1) % n;
        let a1 = polygon.vertices[i];
        let a2 = polygon.vertices[i_next];

        // Check against non-adjacent edges
        for j in (i + 2)..n {
            // Skip if edges share a vertex
            let j_next = (j + 1) % n;
            if j_next == i {
                continue;
            }

            let b1 = polygon.vertices[j];
            let b2 = polygon.vertices[j_next];

            if segments_properly_intersect(a1, a2, b1, b2) {
                return true;
            }
        }
    }

    false
}

/// Finds all self-intersections in a polygon.
///
/// Returns a list of intersection points with the edge indices and parameters.
pub fn find_self_intersections<F: Float>(polygon: &Polygon<F>) -> Vec<SelfIntersection<F>> {
    let mut intersections = Vec::new();
    let n = polygon.vertices.len();

    if n < 4 {
        return intersections;
    }

    for i in 0..n {
        let i_next = (i + 1) % n;
        let a1 = polygon.vertices[i];
        let a2 = polygon.vertices[i_next];

        for j in (i + 2)..n {
            let j_next = (j + 1) % n;
            if j_next == i {
                continue;
            }

            let b1 = polygon.vertices[j];
            let b2 = polygon.vertices[j_next];

            if let Some((point, t1, t2)) = segment_intersection(a1, a2, b1, b2) {
                intersections.push(SelfIntersection {
                    point,
                    edge1: i,
                    t1,
                    edge2: j,
                    t2,
                });
            }
        }
    }

    intersections
}

/// Validates a polygon and returns detailed information about any issues.
pub fn validate<F: Float>(polygon: &Polygon<F>, tolerance: F) -> ValidationResult<F> {
    let n = polygon.vertices.len();

    let is_degenerate = n < 3;
    let has_zero_area = if n >= 3 {
        polygon.signed_area().abs() < tolerance
    } else {
        true
    };

    let has_duplicate_vertices = has_consecutive_duplicates(polygon, tolerance);
    let self_intersections = find_self_intersections(polygon);
    let has_self_intersections = !self_intersections.is_empty();

    let is_valid = !is_degenerate && !has_zero_area && !has_duplicate_vertices && !has_self_intersections;

    ValidationResult {
        is_valid,
        has_self_intersections,
        self_intersections,
        has_duplicate_vertices,
        has_zero_area,
        is_degenerate,
    }
}

/// Checks if a polygon is valid (no self-intersections, degeneracies, etc.).
pub fn is_valid<F: Float>(polygon: &Polygon<F>, tolerance: F) -> bool {
    validate(polygon, tolerance).is_valid
}

/// Repairs a self-intersecting polygon by splitting it into non-intersecting parts.
///
/// Returns a vector of simple polygons. The input polygon is split at each
/// intersection point, and the resulting loops are extracted.
///
/// # Arguments
///
/// * `polygon` - The polygon to repair
/// * `tolerance` - Tolerance for considering points equal
///
/// # Returns
///
/// A vector of simple (non-self-intersecting) polygons.
///
/// # Example
///
/// ```
/// use approxum::{Point2, polygon::{Polygon, repair_self_intersections}};
///
/// // Figure-8 shape
/// let figure8 = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(2.0, 0.0),
///     Point2::new(0.0, 2.0),
/// ]);
///
/// let repaired = repair_self_intersections(&figure8, 1e-9);
/// assert_eq!(repaired.len(), 2);
/// ```
pub fn repair_self_intersections<F: Float>(
    polygon: &Polygon<F>,
    tolerance: F,
) -> Vec<Polygon<F>> {
    let intersections = find_self_intersections(polygon);

    if intersections.is_empty() {
        return vec![polygon.clone()];
    }

    // Build a graph of the polygon with intersection points inserted
    split_at_intersections(polygon, &intersections, tolerance)
}

/// Removes consecutive duplicate vertices from a polygon.
pub fn remove_duplicate_vertices<F: Float>(polygon: &Polygon<F>, tolerance: F) -> Polygon<F> {
    if polygon.vertices.len() < 2 {
        return polygon.clone();
    }

    let mut result = Vec::with_capacity(polygon.vertices.len());
    result.push(polygon.vertices[0]);

    for i in 1..polygon.vertices.len() {
        let prev = result.last().unwrap();
        let curr = polygon.vertices[i];

        if prev.distance(curr) > tolerance {
            result.push(curr);
        }
    }

    // Check last vs first
    if result.len() > 1 {
        let first = result[0];
        let last = *result.last().unwrap();
        if first.distance(last) <= tolerance {
            result.pop();
        }
    }

    Polygon::new(result)
}

/// Checks if a polygon is simple (no self-intersections).
pub fn is_simple<F: Float>(polygon: &Polygon<F>) -> bool {
    !has_self_intersection(polygon)
}

// ============================================================================
// Internal implementation
// ============================================================================

/// Checks if two segments properly intersect (cross each other).
fn segments_properly_intersect<F: Float>(
    a1: Point2<F>,
    a2: Point2<F>,
    b1: Point2<F>,
    b2: Point2<F>,
) -> bool {
    let d1 = cross_product_sign(b1, b2, a1);
    let d2 = cross_product_sign(b1, b2, a2);
    let d3 = cross_product_sign(a1, a2, b1);
    let d4 = cross_product_sign(a1, a2, b2);

    // Proper intersection requires opposite signs
    d1 * d2 < F::zero() && d3 * d4 < F::zero()
}

/// Computes segment-segment intersection point and parameters.
fn segment_intersection<F: Float>(
    a1: Point2<F>,
    a2: Point2<F>,
    b1: Point2<F>,
    b2: Point2<F>,
) -> Option<(Point2<F>, F, F)> {
    let d1x = a2.x - a1.x;
    let d1y = a2.y - a1.y;
    let d2x = b2.x - b1.x;
    let d2y = b2.y - b1.y;

    let cross = d1x * d2y - d1y * d2x;

    if cross.abs() < F::epsilon() {
        return None; // Parallel
    }

    let dx = b1.x - a1.x;
    let dy = b1.y - a1.y;

    let t1 = (dx * d2y - dy * d2x) / cross;
    let t2 = (dx * d1y - dy * d1x) / cross;

    let eps = F::from(1e-10).unwrap();

    // Must be strictly interior (not at endpoints)
    if t1 > eps && t1 < F::one() - eps && t2 > eps && t2 < F::one() - eps {
        let point = Point2::new(a1.x + t1 * d1x, a1.y + t1 * d1y);
        Some((point, t1, t2))
    } else {
        None
    }
}

/// Cross product sign for orientation test.
fn cross_product_sign<F: Float>(p1: Point2<F>, p2: Point2<F>, p3: Point2<F>) -> F {
    (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
}

/// Checks for consecutive duplicate vertices.
fn has_consecutive_duplicates<F: Float>(polygon: &Polygon<F>, tolerance: F) -> bool {
    let n = polygon.vertices.len();
    if n < 2 {
        return false;
    }

    for i in 0..n {
        let j = (i + 1) % n;
        if polygon.vertices[i].distance(polygon.vertices[j]) <= tolerance {
            return true;
        }
    }

    false
}

/// Splits polygon at intersections and extracts simple polygons.
fn split_at_intersections<F: Float>(
    polygon: &Polygon<F>,
    intersections: &[SelfIntersection<F>],
    _tolerance: F,
) -> Vec<Polygon<F>> {
    let n = polygon.vertices.len();

    if n < 3 || intersections.is_empty() {
        return vec![polygon.clone()];
    }

    // For a single intersection (most common case), we can handle it directly
    if intersections.len() == 1 {
        return split_single_intersection(polygon, &intersections[0]);
    }

    // For multiple intersections, use a more general approach
    split_multiple_intersections(polygon, intersections)
}

/// Handle the common case of a single self-intersection.
fn split_single_intersection<F: Float>(
    polygon: &Polygon<F>,
    intersection: &SelfIntersection<F>,
) -> Vec<Polygon<F>> {
    let n = polygon.vertices.len();
    let e1 = intersection.edge1;
    let e2 = intersection.edge2;
    let int_point = intersection.point;

    // Create two polygons by splitting at the intersection
    // Polygon 1: e1_start -> intersection -> e2_end -> ... -> e1_start
    // Polygon 2: intersection -> e1_end -> ... -> e2_start -> intersection

    let mut poly1_verts = Vec::new();
    let mut poly2_verts = Vec::new();

    // Polygon 1: from vertex after e1 to e2, through the intersection
    // Start at vertex e1+1, go to vertex e2, then to intersection
    let mut i = (e1 + 1) % n;
    while i != (e2 + 1) % n {
        poly1_verts.push(polygon.vertices[i]);
        i = (i + 1) % n;
    }
    poly1_verts.push(int_point);

    // Polygon 2: from vertex after e2 to e1, through the intersection
    i = (e2 + 1) % n;
    while i != (e1 + 1) % n {
        poly2_verts.push(polygon.vertices[i]);
        i = (i + 1) % n;
    }
    poly2_verts.push(int_point);

    let mut result = Vec::new();

    if poly1_verts.len() >= 3 {
        let p1 = Polygon::new(poly1_verts);
        if p1.area() > F::epsilon() {
            result.push(p1);
        }
    }

    if poly2_verts.len() >= 3 {
        let p2 = Polygon::new(poly2_verts);
        if p2.area() > F::epsilon() {
            result.push(p2);
        }
    }

    result
}

/// Handle multiple self-intersections.
fn split_multiple_intersections<F: Float>(
    polygon: &Polygon<F>,
    intersections: &[SelfIntersection<F>],
) -> Vec<Polygon<F>> {
    // For multiple intersections, we use a greedy approach:
    // Fix one intersection at a time
    let mut current_polygons = vec![polygon.clone()];

    for _int in intersections {
        let mut new_polygons = Vec::new();

        for poly in &current_polygons {
            // Check if this polygon still has this intersection
            let poly_ints = find_self_intersections(poly);

            if poly_ints.is_empty() {
                new_polygons.push(poly.clone());
            } else {
                // Split at the first intersection found
                let split = split_single_intersection(poly, &poly_ints[0]);
                new_polygons.extend(split);
            }
        }

        current_polygons = new_polygons;
    }

    // Final cleanup - remove any remaining self-intersections
    let mut result = Vec::new();
    for poly in current_polygons {
        if !has_self_intersection(&poly) && poly.vertices.len() >= 3 && poly.area() > F::epsilon() {
            result.push(poly);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn square() -> Polygon<f64> {
        Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ])
    }

    fn figure_8() -> Polygon<f64> {
        Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(2.0, 0.0),
            Point2::new(0.0, 2.0),
        ])
    }

    fn bowtie() -> Polygon<f64> {
        Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(0.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 1.0),
            Point2::new(2.0, 0.0),
        ])
    }

    #[test]
    fn test_simple_polygon_no_self_intersection() {
        let poly = square();
        assert!(!has_self_intersection(&poly));
    }

    #[test]
    fn test_figure_8_has_self_intersection() {
        let poly = figure_8();
        assert!(has_self_intersection(&poly));
    }

    #[test]
    fn test_bowtie_has_self_intersection() {
        let poly = bowtie();
        assert!(has_self_intersection(&poly));
    }

    #[test]
    fn test_find_self_intersections_square() {
        let poly = square();
        let intersections = find_self_intersections(&poly);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_find_self_intersections_figure_8() {
        let poly = figure_8();
        let intersections = find_self_intersections(&poly);

        assert_eq!(intersections.len(), 1);

        // Intersection should be at center (1, 1)
        assert_relative_eq!(intersections[0].point.x, 1.0, epsilon = 0.01);
        assert_relative_eq!(intersections[0].point.y, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_validate_simple_polygon() {
        let poly = square();
        let result = validate(&poly, 1e-9);

        assert!(result.is_valid);
        assert!(!result.has_self_intersections);
        assert!(!result.has_duplicate_vertices);
        assert!(!result.has_zero_area);
        assert!(!result.is_degenerate);
    }

    #[test]
    fn test_validate_self_intersecting() {
        let poly = figure_8();
        let result = validate(&poly, 1e-9);

        assert!(!result.is_valid);
        assert!(result.has_self_intersections);
    }

    #[test]
    fn test_validate_degenerate() {
        let poly = Polygon::new(vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)]);
        let result = validate(&poly, 1e-9);

        assert!(!result.is_valid);
        assert!(result.is_degenerate);
    }

    #[test]
    fn test_validate_zero_area() {
        // Collinear points
        let poly = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ]);
        let result = validate(&poly, 1e-9);

        assert!(!result.is_valid);
        assert!(result.has_zero_area);
    }

    #[test]
    fn test_repair_simple_polygon() {
        let poly = square();
        let repaired = repair_self_intersections(&poly, 1e-9);

        assert_eq!(repaired.len(), 1);
        assert!(!has_self_intersection(&repaired[0]));
    }

    #[test]
    fn test_repair_figure_8() {
        let poly = figure_8();
        let repaired = repair_self_intersections(&poly, 1e-9);

        // Should produce 2 triangles
        assert_eq!(repaired.len(), 2);

        for p in &repaired {
            assert!(!has_self_intersection(p));
            assert!(p.area() > 0.0);
        }

        // Total area should be preserved (approximately)
        let total_area: f64 = repaired.iter().map(|p| p.area()).sum();
        assert!(total_area > 0.5 && total_area < 3.0);
    }

    #[test]
    fn test_is_simple() {
        assert!(is_simple(&square()));
        assert!(!is_simple(&figure_8()));
    }

    #[test]
    fn test_is_valid_fn() {
        assert!(is_valid(&square(), 1e-9));
        assert!(!is_valid(&figure_8(), 1e-9));
    }

    #[test]
    fn test_remove_duplicate_vertices() {
        let poly = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 0.0), // duplicate
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 1.0), // duplicate
            Point2::new(0.0, 1.0),
        ]);

        let cleaned = remove_duplicate_vertices(&poly, 1e-9);
        assert_eq!(cleaned.vertices.len(), 4);
    }

    #[test]
    fn test_triangle_no_self_intersection() {
        let poly = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ]);

        assert!(!has_self_intersection(&poly));
        assert!(is_valid(&poly, 1e-9));
    }

    #[test]
    fn test_complex_self_intersecting() {
        // Star-like shape with multiple crossings
        let star = Polygon::new(vec![
            Point2::new(0.0, 0.5),
            Point2::new(1.0, 1.0),
            Point2::new(0.5, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 0.5),
        ]);

        assert!(has_self_intersection(&star));

        let intersections = find_self_intersections(&star);
        assert!(intersections.len() >= 1);
    }

    #[test]
    fn test_f32_support() {
        let poly: Polygon<f32> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(2.0, 0.0),
            Point2::new(0.0, 2.0),
        ]);

        assert!(has_self_intersection(&poly));
    }

    #[test]
    fn test_adjacent_edges_not_counted() {
        // Make sure adjacent edges don't count as self-intersections
        let poly = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        let intersections = find_self_intersections(&poly);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_validation_result_fields() {
        let poly = figure_8();
        let result = validate(&poly, 1e-9);

        assert!(!result.is_valid);
        assert!(result.has_self_intersections);
        assert!(!result.self_intersections.is_empty());
        assert!(!result.has_duplicate_vertices);
        // Note: figure-8 may have near-zero signed area due to crossing
        assert!(!result.is_degenerate);
    }
}
