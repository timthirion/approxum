//! Curve offset algorithms for creating parallel curves.
//!
//! The offset of a Bézier curve is generally not a Bézier curve, so this module
//! provides approximations using either polylines or sequences of Bézier curves.
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, curves::{CubicBezier2, offset_cubic_to_polyline}};
//!
//! let curve = CubicBezier2::new(
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 2.0),
//!     Point2::new(3.0, 2.0),
//!     Point2::new(4.0, 0.0),
//! );
//!
//! // Offset by 0.5 units (positive = left side when traveling along curve)
//! let offset_points = offset_cubic_to_polyline(&curve, 0.5, 0.01);
//! ```

use super::bezier::{CubicBezier2, QuadraticBezier2};
use crate::primitives::Point2;
use num_traits::Float;

/// Offset options for controlling the output quality.
#[derive(Debug, Clone, Copy)]
pub struct OffsetOptions<F> {
    /// Maximum deviation from the true offset curve.
    pub tolerance: F,
    /// Maximum number of subdivisions for adaptive sampling.
    pub max_subdivisions: usize,
    /// Whether to remove self-intersections from the result.
    pub remove_self_intersections: bool,
}

impl<F: Float> Default for OffsetOptions<F> {
    fn default() -> Self {
        Self {
            tolerance: F::from(0.01).unwrap(),
            max_subdivisions: 100,
            remove_self_intersections: false,
        }
    }
}

impl<F: Float> OffsetOptions<F> {
    /// Creates options with the specified tolerance.
    pub fn with_tolerance(tolerance: F) -> Self {
        Self {
            tolerance,
            ..Default::default()
        }
    }
}

/// Offsets a quadratic Bézier curve, returning a polyline approximation.
///
/// # Arguments
///
/// * `curve` - The quadratic Bézier curve to offset
/// * `distance` - Offset distance (positive = left, negative = right)
/// * `tolerance` - Maximum deviation from the true offset curve
///
/// # Returns
///
/// A vector of points representing the offset curve as a polyline.
///
/// # Example
///
/// ```
/// use approxum::{Point2, curves::{QuadraticBezier2, offset_quadratic_to_polyline}};
///
/// let curve = QuadraticBezier2::new(
///     Point2::new(0.0, 0.0),
///     Point2::new(2.0, 4.0),
///     Point2::new(4.0, 0.0),
/// );
///
/// let offset = offset_quadratic_to_polyline(&curve, 0.5, 0.01);
/// assert!(offset.len() >= 2);
/// ```
pub fn offset_quadratic_to_polyline<F: Float>(
    curve: &QuadraticBezier2<F>,
    distance: F,
    tolerance: F,
) -> Vec<Point2<F>> {
    let options = OffsetOptions::with_tolerance(tolerance);
    offset_quadratic_adaptive(curve, distance, &options)
}

/// Offsets a cubic Bézier curve, returning a polyline approximation.
///
/// # Arguments
///
/// * `curve` - The cubic Bézier curve to offset
/// * `distance` - Offset distance (positive = left, negative = right)
/// * `tolerance` - Maximum deviation from the true offset curve
///
/// # Returns
///
/// A vector of points representing the offset curve as a polyline.
///
/// # Example
///
/// ```
/// use approxum::{Point2, curves::{CubicBezier2, offset_cubic_to_polyline}};
///
/// let curve = CubicBezier2::new(
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 2.0),
///     Point2::new(3.0, 2.0),
///     Point2::new(4.0, 0.0),
/// );
///
/// let offset = offset_cubic_to_polyline(&curve, 0.5, 0.01);
/// assert!(offset.len() >= 2);
/// ```
pub fn offset_cubic_to_polyline<F: Float>(
    curve: &CubicBezier2<F>,
    distance: F,
    tolerance: F,
) -> Vec<Point2<F>> {
    let options = OffsetOptions::with_tolerance(tolerance);
    offset_cubic_adaptive(curve, distance, &options)
}

/// Offsets a quadratic Bézier curve with full options control.
pub fn offset_quadratic_with_options<F: Float>(
    curve: &QuadraticBezier2<F>,
    distance: F,
    options: &OffsetOptions<F>,
) -> Vec<Point2<F>> {
    let mut points = offset_quadratic_adaptive(curve, distance, options);

    if options.remove_self_intersections && points.len() > 3 {
        remove_self_intersections(&mut points);
    }

    points
}

/// Offsets a cubic Bézier curve with full options control.
pub fn offset_cubic_with_options<F: Float>(
    curve: &CubicBezier2<F>,
    distance: F,
    options: &OffsetOptions<F>,
) -> Vec<Point2<F>> {
    let mut points = offset_cubic_adaptive(curve, distance, options);

    if options.remove_self_intersections && points.len() > 3 {
        remove_self_intersections(&mut points);
    }

    points
}

/// Offsets a quadratic Bézier and approximates the result with quadratic Béziers.
///
/// Returns a sequence of quadratic Bézier curves that approximate the offset.
pub fn offset_quadratic_to_quadratics<F: Float>(
    curve: &QuadraticBezier2<F>,
    distance: F,
    tolerance: F,
) -> Vec<QuadraticBezier2<F>> {
    let points = offset_quadratic_to_polyline(curve, distance, tolerance);

    if points.len() < 3 {
        return vec![];
    }

    // Fit quadratic Béziers to point triples
    fit_quadratics_to_polyline(&points)
}

/// Offsets a cubic Bézier and approximates the result with cubic Béziers.
///
/// Returns a sequence of cubic Bézier curves that approximate the offset.
///
/// # Example
///
/// ```
/// use approxum::{Point2, curves::{CubicBezier2, offset_cubic_to_cubics}};
///
/// let curve = CubicBezier2::new(
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 2.0),
///     Point2::new(3.0, 2.0),
///     Point2::new(4.0, 0.0),
/// );
///
/// let offset_curves = offset_cubic_to_cubics(&curve, 0.5, 0.01);
/// ```
pub fn offset_cubic_to_cubics<F: Float>(
    curve: &CubicBezier2<F>,
    distance: F,
    tolerance: F,
) -> Vec<CubicBezier2<F>> {
    let points = offset_cubic_to_polyline(curve, distance, tolerance);

    if points.len() < 4 {
        return vec![];
    }

    // Fit cubic Béziers to the polyline
    fit_cubics_to_polyline(&points, tolerance)
}

/// Computes the signed curvature of a quadratic Bézier at parameter t.
///
/// Positive curvature means the curve bends left, negative means right.
pub fn quadratic_curvature<F: Float>(curve: &QuadraticBezier2<F>, t: F) -> F {
    let d1 = curve.derivative_at(t);
    let two = F::one() + F::one();

    // Second derivative of quadratic is constant: 2(p0 - 2p1 + p2)
    let d2x = two * (curve.p0.x - two * curve.p1.x + curve.p2.x);
    let d2y = two * (curve.p0.y - two * curve.p1.y + curve.p2.y);

    let speed_sq = d1.x * d1.x + d1.y * d1.y;
    let speed = speed_sq.sqrt();

    if speed < F::epsilon() {
        return F::zero();
    }

    // Curvature = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    let cross = d1.x * d2y - d1.y * d2x;
    cross / (speed_sq * speed)
}

/// Computes the signed curvature of a cubic Bézier at parameter t.
pub fn cubic_curvature<F: Float>(curve: &CubicBezier2<F>, t: F) -> F {
    let d1 = curve.derivative_at(t);

    // Second derivative
    let deriv1 = curve.derivative();
    let d2 = deriv1.derivative_at(t);

    let speed_sq = d1.x * d1.x + d1.y * d1.y;
    let speed = speed_sq.sqrt();

    if speed < F::epsilon() {
        return F::zero();
    }

    let cross = d1.x * d2.y - d1.y * d2.x;
    cross / (speed_sq * speed)
}

// ============================================================================
// Internal implementation
// ============================================================================

/// Offset point along the normal at parameter t.
fn offset_point_quadratic<F: Float>(curve: &QuadraticBezier2<F>, t: F, distance: F) -> Point2<F> {
    let point = curve.eval(t);
    let tangent = curve.derivative_at(t);

    let len = (tangent.x * tangent.x + tangent.y * tangent.y).sqrt();

    if len < F::epsilon() {
        return point;
    }

    // Normal is perpendicular to tangent (rotate 90 degrees CCW)
    let normal_x = -tangent.y / len;
    let normal_y = tangent.x / len;

    Point2::new(point.x + distance * normal_x, point.y + distance * normal_y)
}

fn offset_point_cubic<F: Float>(curve: &CubicBezier2<F>, t: F, distance: F) -> Point2<F> {
    let point = curve.eval(t);
    let tangent = curve.derivative_at(t);

    let len = (tangent.x * tangent.x + tangent.y * tangent.y).sqrt();

    if len < F::epsilon() {
        return point;
    }

    let normal_x = -tangent.y / len;
    let normal_y = tangent.x / len;

    Point2::new(point.x + distance * normal_x, point.y + distance * normal_y)
}

/// Adaptive sampling for quadratic offset.
fn offset_quadratic_adaptive<F: Float>(
    curve: &QuadraticBezier2<F>,
    distance: F,
    options: &OffsetOptions<F>,
) -> Vec<Point2<F>> {
    let mut points = Vec::new();
    points.push(offset_point_quadratic(curve, F::zero(), distance));

    subdivide_quadratic_offset(
        curve,
        distance,
        F::zero(),
        F::one(),
        options.tolerance,
        &mut points,
        0,
        options.max_subdivisions,
    );

    points
}

#[allow(clippy::too_many_arguments)]
fn subdivide_quadratic_offset<F: Float>(
    curve: &QuadraticBezier2<F>,
    distance: F,
    t_start: F,
    t_end: F,
    tolerance: F,
    points: &mut Vec<Point2<F>>,
    depth: usize,
    max_depth: usize,
) {
    if depth >= max_depth {
        points.push(offset_point_quadratic(curve, t_end, distance));
        return;
    }

    let half = F::from(0.5).unwrap();
    let t_mid = t_start + half * (t_end - t_start);

    let p_start = offset_point_quadratic(curve, t_start, distance);
    let p_mid = offset_point_quadratic(curve, t_mid, distance);
    let p_end = offset_point_quadratic(curve, t_end, distance);

    // Check if midpoint deviates from straight line
    let deviation = point_to_line_distance(p_mid, p_start, p_end);

    if deviation <= tolerance {
        points.push(p_end);
    } else {
        subdivide_quadratic_offset(
            curve,
            distance,
            t_start,
            t_mid,
            tolerance,
            points,
            depth + 1,
            max_depth,
        );
        subdivide_quadratic_offset(
            curve,
            distance,
            t_mid,
            t_end,
            tolerance,
            points,
            depth + 1,
            max_depth,
        );
    }
}

/// Adaptive sampling for cubic offset.
fn offset_cubic_adaptive<F: Float>(
    curve: &CubicBezier2<F>,
    distance: F,
    options: &OffsetOptions<F>,
) -> Vec<Point2<F>> {
    let mut points = Vec::new();
    points.push(offset_point_cubic(curve, F::zero(), distance));

    subdivide_cubic_offset(
        curve,
        distance,
        F::zero(),
        F::one(),
        options.tolerance,
        &mut points,
        0,
        options.max_subdivisions,
    );

    points
}

#[allow(clippy::too_many_arguments)]
fn subdivide_cubic_offset<F: Float>(
    curve: &CubicBezier2<F>,
    distance: F,
    t_start: F,
    t_end: F,
    tolerance: F,
    points: &mut Vec<Point2<F>>,
    depth: usize,
    max_depth: usize,
) {
    if depth >= max_depth {
        points.push(offset_point_cubic(curve, t_end, distance));
        return;
    }

    let half = F::from(0.5).unwrap();
    let t_mid = t_start + half * (t_end - t_start);

    let p_start = offset_point_cubic(curve, t_start, distance);
    let p_mid = offset_point_cubic(curve, t_mid, distance);
    let p_end = offset_point_cubic(curve, t_end, distance);

    let deviation = point_to_line_distance(p_mid, p_start, p_end);

    if deviation <= tolerance {
        points.push(p_end);
    } else {
        subdivide_cubic_offset(
            curve,
            distance,
            t_start,
            t_mid,
            tolerance,
            points,
            depth + 1,
            max_depth,
        );
        subdivide_cubic_offset(
            curve,
            distance,
            t_mid,
            t_end,
            tolerance,
            points,
            depth + 1,
            max_depth,
        );
    }
}

/// Distance from point to line defined by two points.
fn point_to_line_distance<F: Float>(
    point: Point2<F>,
    line_start: Point2<F>,
    line_end: Point2<F>,
) -> F {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;
    let len_sq = dx * dx + dy * dy;

    if len_sq < F::epsilon() {
        return point.distance(line_start);
    }

    let cross = (point.x - line_start.x) * dy - (point.y - line_start.y) * dx;
    cross.abs() / len_sq.sqrt()
}

/// Fit quadratic Béziers to a polyline.
fn fit_quadratics_to_polyline<F: Float>(points: &[Point2<F>]) -> Vec<QuadraticBezier2<F>> {
    if points.len() < 3 {
        return vec![];
    }

    let mut curves = Vec::new();

    // Simple approach: every 2 segments becomes one quadratic
    let mut i = 0;
    while i + 2 < points.len() {
        let p0 = points[i];
        let p2 = points[i + 2];

        // Control point: intersection of tangent lines, or midpoint as fallback
        let p1 = if i + 1 < points.len() {
            // Use the middle point projected onto the control polygon

            // Simple approximation: use the actual midpoint
            points[i + 1]
        } else {
            Point2::new(
                (p0.x + p2.x) / (F::one() + F::one()),
                (p0.y + p2.y) / (F::one() + F::one()),
            )
        };

        curves.push(QuadraticBezier2::new(p0, p1, p2));
        i += 2;
    }

    // Handle remaining points
    if i < points.len() - 1 {
        let p0 = points[i];
        let p2 = *points.last().unwrap();
        let p1 = if i + 1 < points.len() - 1 {
            points[i + 1]
        } else {
            Point2::new(
                (p0.x + p2.x) / (F::one() + F::one()),
                (p0.y + p2.y) / (F::one() + F::one()),
            )
        };
        curves.push(QuadraticBezier2::new(p0, p1, p2));
    }

    curves
}

/// Fit cubic Béziers to a polyline using Catmull-Rom style fitting.
fn fit_cubics_to_polyline<F: Float>(points: &[Point2<F>], _tolerance: F) -> Vec<CubicBezier2<F>> {
    if points.len() < 2 {
        return vec![];
    }

    if points.len() == 2 {
        // Straight line
        let third = F::one() / F::from(3.0).unwrap();
        let p0 = points[0];
        let p3 = points[1];
        let p1 = Point2::new(p0.x + third * (p3.x - p0.x), p0.y + third * (p3.y - p0.y));
        let p2 = Point2::new(
            p0.x + (F::one() - third) * (p3.x - p0.x),
            p0.y + (F::one() - third) * (p3.y - p0.y),
        );
        return vec![CubicBezier2::new(p0, p1, p2, p3)];
    }

    let mut curves = Vec::new();
    let n = points.len();

    // Compute tangents at each point
    let mut tangents: Vec<Point2<F>> = Vec::with_capacity(n);

    // First tangent
    tangents.push(Point2::new(
        points[1].x - points[0].x,
        points[1].y - points[0].y,
    ));

    // Interior tangents (average of adjacent segments)
    for i in 1..n - 1 {
        let tx = (points[i + 1].x - points[i - 1].x) / (F::one() + F::one());
        let ty = (points[i + 1].y - points[i - 1].y) / (F::one() + F::one());
        tangents.push(Point2::new(tx, ty));
    }

    // Last tangent
    tangents.push(Point2::new(
        points[n - 1].x - points[n - 2].x,
        points[n - 1].y - points[n - 2].y,
    ));

    // Create cubic segments
    let third = F::one() / F::from(3.0).unwrap();

    for i in 0..n - 1 {
        let p0 = points[i];
        let p3 = points[i + 1];

        let p1 = Point2::new(p0.x + third * tangents[i].x, p0.y + third * tangents[i].y);

        let p2 = Point2::new(
            p3.x - third * tangents[i + 1].x,
            p3.y - third * tangents[i + 1].y,
        );

        curves.push(CubicBezier2::new(p0, p1, p2, p3));
    }

    curves
}

/// Remove self-intersections from a polyline by cutting loops.
fn remove_self_intersections<F: Float>(points: &mut Vec<Point2<F>>) {
    if points.len() < 4 {
        return;
    }

    let mut i = 0;
    while i < points.len().saturating_sub(3) {
        let mut found_intersection = false;
        let mut j = i + 2;

        while j < points.len() - 1 {
            if let Some(_intersection) =
                segment_intersection(points[i], points[i + 1], points[j], points[j + 1])
            {
                // Remove points between i+1 and j (inclusive)
                points.drain(i + 1..=j);
                found_intersection = true;
                break;
            }
            j += 1;
        }

        if !found_intersection {
            i += 1;
        }
    }
}

/// Segment-segment intersection for self-intersection removal.
fn segment_intersection<F: Float>(
    a1: Point2<F>,
    a2: Point2<F>,
    b1: Point2<F>,
    b2: Point2<F>,
) -> Option<Point2<F>> {
    let d1x = a2.x - a1.x;
    let d1y = a2.y - a1.y;
    let d2x = b2.x - b1.x;
    let d2y = b2.y - b1.y;

    let cross = d1x * d2y - d1y * d2x;

    if cross.abs() < F::epsilon() {
        return None;
    }

    let dx = b1.x - a1.x;
    let dy = b1.y - a1.y;

    let t1 = (dx * d2y - dy * d2x) / cross;
    let t2 = (dx * d1y - dy * d1x) / cross;

    let eps = F::from(0.001).unwrap();

    if t1 > eps && t1 < F::one() - eps && t2 > eps && t2 < F::one() - eps {
        Some(Point2::new(a1.x + t1 * d1x, a1.y + t1 * d1y))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_offset_quadratic_to_polyline() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 4.0),
            Point2::new(4.0, 0.0),
        );

        let offset = offset_quadratic_to_polyline(&curve, 0.5, 0.01);

        assert!(offset.len() >= 2);

        // First point should be offset from curve start
        let expected_start = offset_point_quadratic(&curve, 0.0, 0.5);
        assert_relative_eq!(offset[0].x, expected_start.x, epsilon = 0.01);
        assert_relative_eq!(offset[0].y, expected_start.y, epsilon = 0.01);
    }

    #[test]
    fn test_offset_cubic_to_polyline() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let offset = offset_cubic_to_polyline(&curve, 0.5, 0.01);

        assert!(offset.len() >= 2);

        // Check that offset points are approximately distance away from original
        for (i, &p) in offset.iter().enumerate() {
            let t = i as f64 / (offset.len() - 1) as f64;
            let orig = curve.eval(t);
            let dist = p.distance(orig);
            // Distance should be close to offset distance (some variation expected)
            assert!(dist > 0.3 && dist < 0.7, "Distance {} at t={}", dist, t);
        }
    }

    #[test]
    fn test_offset_negative_distance() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let offset_left = offset_cubic_to_polyline(&curve, 0.5, 0.01);
        let offset_right = offset_cubic_to_polyline(&curve, -0.5, 0.01);

        // Offsets should be on opposite sides
        let mid_left = offset_left[offset_left.len() / 2];
        let mid_right = offset_right[offset_right.len() / 2];

        // They should be different
        assert!(mid_left.distance(mid_right) > 0.5);
    }

    #[test]
    fn test_offset_straight_line() {
        // Straight line curve
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        );

        let offset = offset_cubic_to_polyline(&curve, 1.0, 0.01);

        // All points should be at y = 1.0 (offset up from y = 0)
        for p in &offset {
            assert_relative_eq!(p.y, 1.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_offset_to_cubics() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let offset_curves = offset_cubic_to_cubics(&curve, 0.5, 0.1);

        assert!(!offset_curves.is_empty());

        // Curves should be connected
        for i in 1..offset_curves.len() {
            let prev_end = offset_curves[i - 1].p3;
            let curr_start = offset_curves[i].p0;
            assert!(prev_end.distance(curr_start) < 0.01);
        }
    }

    #[test]
    fn test_offset_to_quadratics() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 4.0),
            Point2::new(4.0, 0.0),
        );

        let offset_curves = offset_quadratic_to_quadratics(&curve, 0.5, 0.1);

        assert!(!offset_curves.is_empty());
    }

    #[test]
    fn test_quadratic_curvature() {
        // Symmetric parabola
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        let curvature_mid = quadratic_curvature(&curve, 0.5);

        // Curvature should be negative (bending right/down)
        assert!(curvature_mid < 0.0);
    }

    #[test]
    fn test_cubic_curvature() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let curvature_start = cubic_curvature(&curve, 0.0);
        let curvature_mid = cubic_curvature(&curve, 0.5);
        let curvature_end = cubic_curvature(&curve, 1.0);

        // Curvature values should be finite
        assert!(curvature_start.is_finite());
        assert!(curvature_mid.is_finite());
        assert!(curvature_end.is_finite());

        // Curvature should be non-zero for curved parts
        assert!(curvature_start.abs() > 0.0 || curvature_mid.abs() > 0.0);
    }

    #[test]
    fn test_offset_with_options() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let options = OffsetOptions {
            tolerance: 0.1,
            max_subdivisions: 50,
            remove_self_intersections: true,
        };

        let offset = offset_cubic_with_options(&curve, 0.5, &options);
        assert!(!offset.is_empty());
    }

    #[test]
    fn test_offset_zero_distance() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let offset = offset_cubic_to_polyline(&curve, 0.0, 0.01);

        // Zero offset should give points on the original curve
        for (i, &p) in offset.iter().enumerate() {
            let t = i as f64 / (offset.len() - 1) as f64;
            let orig = curve.eval(t);
            assert!(p.distance(orig) < 0.01);
        }
    }

    #[test]
    fn test_f32_support() {
        let curve: CubicBezier2<f32> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let offset = offset_cubic_to_polyline(&curve, 0.5, 0.1);
        assert!(!offset.is_empty());
    }

    #[test]
    fn test_offset_high_curvature() {
        // S-curve that has varying curvature
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, -1.0),
            Point2::new(3.0, 0.0),
        );

        let offset = offset_cubic_to_polyline(&curve, 0.2, 0.01);

        // Should have at least start and end points
        assert!(offset.len() >= 2);

        // Offset curve should be different from original
        let orig_start = curve.eval(0.0);
        assert!(offset[0].distance(orig_start) > 0.1);
    }

    #[test]
    fn test_point_to_line_distance() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(2.0, 0.0);
        let p = Point2::new(1.0, 1.0);

        let dist = point_to_line_distance(p, a, b);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_large_offset() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        // Large offset
        let offset = offset_cubic_to_polyline(&curve, 1.5, 0.1);
        assert!(!offset.is_empty());
        assert!(offset.len() >= 2);
    }
}
