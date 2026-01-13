//! Polygon offsetting (buffering/inflating/deflating).
//!
//! Offset a polygon inward or outward by a given distance.
//! Different join styles handle corners differently.
//!
//! # Example
//!
//! ```
//! use approxum::polygon::{Polygon, offset_polygon, JoinStyle};
//! use approxum::Point2;
//!
//! let square = Polygon::new(vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(10.0, 0.0),
//!     Point2::new(10.0, 10.0),
//!     Point2::new(0.0, 10.0),
//! ]);
//!
//! // Expand the square outward by 1 unit
//! let expanded = offset_polygon(&square, 1.0, JoinStyle::Miter, 2.0);
//! assert!(!expanded.is_empty());
//! assert!(expanded.area() > square.area());
//!
//! // Shrink the square inward by 1 unit
//! let shrunk = offset_polygon(&square, -1.0, JoinStyle::Miter, 2.0);
//! assert!(!shrunk.is_empty());
//! assert!(shrunk.area() < square.area());
//! ```

use crate::polygon::Polygon;
use crate::primitives::Point2;
use num_traits::Float;
use std::f64::consts::PI;

/// Style for handling corners when offsetting polygons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinStyle {
    /// Extend edges until they meet. Can create long spikes at sharp angles.
    /// Use miter_limit to control maximum extension.
    Miter,
    /// Cut off corners with a straight line.
    Bevel,
    /// Round corners with arc segments.
    Round,
}

/// Offsets a polygon by a given distance.
///
/// Positive offset expands the polygon outward, negative offset shrinks it inward.
///
/// # Arguments
///
/// * `polygon` - The polygon to offset
/// * `distance` - Offset distance (positive = outward, negative = inward)
/// * `join_style` - How to handle corners
/// * `miter_limit` - For miter joins, the maximum ratio of miter length to offset distance
///
/// # Returns
///
/// The offset polygon. May be empty if inward offset exceeds polygon dimensions.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, offset_polygon, JoinStyle};
/// use approxum::Point2;
///
/// let triangle = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(4.0, 0.0),
///     Point2::new(2.0, 3.0),
/// ]);
///
/// let offset = offset_polygon(&triangle, 0.5, JoinStyle::Round, 2.0);
/// assert!(!offset.is_empty());
/// ```
pub fn offset_polygon<F: Float>(
    polygon: &Polygon<F>,
    distance: F,
    join_style: JoinStyle,
    miter_limit: F,
) -> Polygon<F> {
    if polygon.len() < 3 {
        return Polygon::empty();
    }

    // Ensure CCW winding for consistent offset direction
    let vertices = if polygon.signed_area() < F::zero() {
        polygon.vertices.iter().rev().copied().collect::<Vec<_>>()
    } else {
        polygon.vertices.clone()
    };

    let n = vertices.len();
    let mut result: Vec<Point2<F>> = Vec::new();

    // For each vertex, compute the offset corner
    for i in 0..n {
        let prev = vertices[(i + n - 1) % n];
        let curr = vertices[i];
        let next = vertices[(i + 1) % n];

        // Compute edge normals (perpendicular to edges, pointing outward for CCW polygon)
        let edge1 = normalize(Point2::new(curr.x - prev.x, curr.y - prev.y));
        let edge2 = normalize(Point2::new(next.x - curr.x, next.y - curr.y));

        // Outward normals (rotate 90 degrees CW for CCW polygon)
        // Rotating (x, y) by 90 CW gives (y, -x)
        let normal1 = Point2::new(edge1.y, -edge1.x);
        let normal2 = Point2::new(edge2.y, -edge2.x);

        // Offset points on each edge
        let p1 = Point2::new(curr.x + normal1.x * distance, curr.y + normal1.y * distance);
        let p2 = Point2::new(curr.x + normal2.x * distance, curr.y + normal2.y * distance);

        // Compute the corner based on join style
        match join_style {
            JoinStyle::Miter => {
                if let Some(corner) = miter_corner(prev, curr, next, distance, miter_limit) {
                    result.push(corner);
                } else {
                    // Fall back to bevel if miter limit exceeded
                    result.push(p1);
                    result.push(p2);
                }
            }
            JoinStyle::Bevel => {
                result.push(p1);
                if !points_equal(p1, p2) {
                    result.push(p2);
                }
            }
            JoinStyle::Round => {
                // Add arc from p1 to p2
                add_round_corner(&mut result, curr, p1, p2, distance, normal1, normal2);
            }
        }
    }

    // Remove near-duplicate consecutive vertices
    result.dedup_by(|a, b| points_equal(*a, *b));

    // Close the polygon if needed
    if !result.is_empty() && points_equal(result[0], result[result.len() - 1]) {
        result.pop();
    }

    if result.len() < 3 {
        return Polygon::empty();
    }

    Polygon::new(result)
}

/// Offsets a polygon with default miter limit.
pub fn offset_polygon_simple<F: Float>(
    polygon: &Polygon<F>,
    distance: F,
    join_style: JoinStyle,
) -> Polygon<F> {
    offset_polygon(polygon, distance, join_style, F::from(2.0).unwrap())
}

/// Computes the miter corner point.
fn miter_corner<F: Float>(
    prev: Point2<F>,
    curr: Point2<F>,
    next: Point2<F>,
    distance: F,
    miter_limit: F,
) -> Option<Point2<F>> {
    // Edge directions
    let d1 = normalize(Point2::new(curr.x - prev.x, curr.y - prev.y));
    let d2 = normalize(Point2::new(next.x - curr.x, next.y - curr.y));

    // Outward normals (rotate 90 degrees CW for CCW polygon)
    let n1 = Point2::new(d1.y, -d1.x);
    let n2 = Point2::new(d2.y, -d2.x);

    // Bisector direction
    let bisector_x = n1.x + n2.x;
    let bisector_y = n1.y + n2.y;
    let bisector_len = (bisector_x * bisector_x + bisector_y * bisector_y).sqrt();

    if bisector_len < F::epsilon() {
        // Edges are parallel - just offset perpendicular
        return Some(Point2::new(curr.x + n1.x * distance, curr.y + n1.y * distance));
    }

    // Normalized bisector
    let bx = bisector_x / bisector_len;
    let by = bisector_y / bisector_len;

    // Compute the miter length
    // The miter length is distance / sin(half_angle)
    // sin(half_angle) = |cross(n1, bisector)| = |n1.x * by - n1.y * bx|
    let sin_half = (n1.x * by - n1.y * bx).abs();

    if sin_half < F::epsilon() {
        return Some(Point2::new(curr.x + n1.x * distance, curr.y + n1.y * distance));
    }

    let miter_length = distance / sin_half;

    // Check miter limit
    if miter_length.abs() > miter_limit * distance.abs() {
        return None; // Exceeded limit, use bevel instead
    }

    Some(Point2::new(curr.x + bx * miter_length, curr.y + by * miter_length))
}

/// Adds a rounded corner as a series of arc points.
fn add_round_corner<F: Float>(
    result: &mut Vec<Point2<F>>,
    center: Point2<F>,
    p1: Point2<F>,
    p2: Point2<F>,
    distance: F,
    n1: Point2<F>,
    n2: Point2<F>,
) {
    // Compute angles of the two normals
    let angle1 = n1.y.atan2(n1.x);
    let angle2 = n2.y.atan2(n2.x);

    // Determine sweep direction
    let mut sweep = angle2 - angle1;

    // Normalize sweep to be in the right direction based on offset sign
    if distance > F::zero() {
        // Outward offset: sweep should be positive (CCW)
        if sweep < F::zero() {
            sweep = sweep + F::from(2.0 * PI).unwrap();
        }
    } else {
        // Inward offset: sweep should be negative (CW)
        if sweep > F::zero() {
            sweep = sweep - F::from(2.0 * PI).unwrap();
        }
    }

    // Number of segments for the arc (based on sweep angle)
    let abs_sweep = sweep.abs();
    let segments = (abs_sweep * F::from(8.0 / PI).unwrap()).ceil().max(F::one());
    let segments_usize = segments.to_usize().unwrap_or(8).min(32);

    if segments_usize <= 1 {
        result.push(p1);
        if !points_equal(p1, p2) {
            result.push(p2);
        }
        return;
    }

    let abs_distance = distance.abs();
    let step = sweep / F::from(segments_usize).unwrap();

    for i in 0..=segments_usize {
        let angle = angle1 + step * F::from(i).unwrap();
        let x = center.x + angle.cos() * abs_distance;
        let y = center.y + angle.sin() * abs_distance;
        let pt = Point2::new(x, y);

        if result.is_empty() || !points_equal(pt, result[result.len() - 1]) {
            result.push(pt);
        }
    }
}

/// Normalizes a vector (represented as a Point2 for convenience).
#[inline]
fn normalize<F: Float>(v: Point2<F>) -> Point2<F> {
    let len = (v.x * v.x + v.y * v.y).sqrt();
    if len < F::epsilon() {
        Point2::new(F::zero(), F::zero())
    } else {
        Point2::new(v.x / len, v.y / len)
    }
}

/// Checks if two points are approximately equal.
#[inline]
fn points_equal<F: Float>(a: Point2<F>, b: Point2<F>) -> bool {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt() < F::from(1e-10).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_offset_square_outward_miter() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let offset = offset_polygon(&square, 1.0, JoinStyle::Miter, 2.0);
        assert!(!offset.is_empty());

        // Offset square should be 12x12 (expanded by 1 on each side)
        // Area should be 144
        assert!(approx_eq(offset.area(), 144.0, 1.0));
    }

    #[test]
    fn test_offset_square_inward_miter() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let offset = offset_polygon(&square, -1.0, JoinStyle::Miter, 2.0);
        assert!(!offset.is_empty());

        // Offset square should be 8x8 (shrunk by 1 on each side)
        // Area should be 64
        assert!(approx_eq(offset.area(), 64.0, 1.0));
    }

    #[test]
    fn test_offset_square_bevel() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let offset = offset_polygon(&square, 1.0, JoinStyle::Bevel, 2.0);
        assert!(!offset.is_empty());

        // Bevel creates an octagon (corners cut off)
        assert_eq!(offset.len(), 8);
    }

    #[test]
    fn test_offset_square_round() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let offset = offset_polygon(&square, 1.0, JoinStyle::Round, 2.0);
        assert!(!offset.is_empty());

        // Round creates more vertices at corners
        assert!(offset.len() > 8);
    }

    #[test]
    fn test_offset_triangle() {
        let triangle = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(5.0, 8.66),
        ]);

        let offset = offset_polygon(&triangle, 1.0, JoinStyle::Miter, 2.0);
        assert!(!offset.is_empty());
        assert!(offset.area() > triangle.area());
    }

    #[test]
    fn test_offset_preserves_shape() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        // Offset out then in by same amount should give similar area
        let expanded = offset_polygon(&square, 2.0, JoinStyle::Miter, 2.0);
        let back = offset_polygon(&expanded, -2.0, JoinStyle::Miter, 2.0);

        assert!(approx_eq(back.area(), square.area(), 5.0));
    }

    #[test]
    fn test_offset_inward_collapse() {
        let small_square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 4.0),
            Point2::new(0.0, 4.0),
        ]);

        // Offset inward by more than half the width
        let offset = offset_polygon(&small_square, -3.0, JoinStyle::Miter, 2.0);

        // Result should be significantly smaller (self-intersection handling is approximate)
        assert!(offset.area() < small_square.area() * 0.5);
    }

    #[test]
    fn test_offset_zero() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let offset = offset_polygon(&square, 0.0, JoinStyle::Miter, 2.0);
        assert!(!offset.is_empty());
        assert!(approx_eq(offset.area(), square.area(), 0.1));
    }

    #[test]
    fn test_offset_empty_polygon() {
        let empty: Polygon<f64> = Polygon::empty();
        let offset = offset_polygon(&empty, 1.0, JoinStyle::Miter, 2.0);
        assert!(offset.is_empty());
    }

    #[test]
    fn test_offset_simple_alias() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let offset = offset_polygon_simple(&square, 1.0, JoinStyle::Miter);
        assert!(!offset.is_empty());
    }

    #[test]
    fn test_offset_cw_polygon() {
        // CW polygon (should be auto-corrected)
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(0.0, 10.0),
            Point2::new(10.0, 10.0),
            Point2::new(10.0, 0.0),
        ]);

        let offset = offset_polygon(&square, 1.0, JoinStyle::Miter, 2.0);
        assert!(!offset.is_empty());
    }

    #[test]
    fn test_offset_f32() {
        let square: Polygon<f32> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let offset = offset_polygon(&square, 1.0, JoinStyle::Miter, 2.0);
        assert!(!offset.is_empty());
        assert!((offset.area() - 144.0).abs() < 2.0);
    }

    #[test]
    fn test_miter_limit() {
        // Sharp angle triangle - miter would create a long spike
        let sharp = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(5.0, 1.0), // Very sharp angle at top
        ]);

        // With low miter limit, should fall back to bevel
        let offset = offset_polygon(&sharp, 1.0, JoinStyle::Miter, 1.5);
        assert!(!offset.is_empty());
    }

    #[test]
    fn test_join_style_eq() {
        assert_eq!(JoinStyle::Miter, JoinStyle::Miter);
        assert_ne!(JoinStyle::Miter, JoinStyle::Bevel);
        assert_ne!(JoinStyle::Bevel, JoinStyle::Round);
    }

    #[test]
    fn test_offset_hexagon() {
        // Regular hexagon
        let hex = Polygon::new(vec![
            Point2::new(2.0_f64, 0.0),
            Point2::new(1.0, 1.732),
            Point2::new(-1.0, 1.732),
            Point2::new(-2.0, 0.0),
            Point2::new(-1.0, -1.732),
            Point2::new(1.0, -1.732),
        ]);

        let offset = offset_polygon(&hex, 0.5, JoinStyle::Miter, 2.0);
        assert!(!offset.is_empty());
        assert!(offset.area() > hex.area());
    }

    #[test]
    fn test_round_corner_arc_count() {
        let square = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let offset = offset_polygon(&square, 2.0, JoinStyle::Round, 2.0);

        // Each 90-degree corner should produce several arc points
        // 4 corners * ~4 points each + 4 edge endpoints = roughly 16-24 points
        assert!(offset.len() >= 8);
    }
}
