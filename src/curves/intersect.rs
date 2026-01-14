//! Curve intersection algorithms.
//!
//! Provides intersection detection between Bézier curves and other geometric primitives.
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, curves::{CubicBezier2, intersect_cubic_cubic}};
//!
//! let curve1 = CubicBezier2::new(
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 2.0),
//!     Point2::new(3.0, 2.0),
//!     Point2::new(4.0, 0.0),
//! );
//!
//! let curve2 = CubicBezier2::new(
//!     Point2::new(0.0, 1.0),
//!     Point2::new(4.0, 1.0),
//!     Point2::new(4.0, 1.0),
//!     Point2::new(4.0, -1.0),
//! );
//!
//! let intersections = intersect_cubic_cubic(&curve1, &curve2, 1e-6);
//! ```

use super::bezier::{CubicBezier2, QuadraticBezier2};
use crate::bounds::Aabb2;
use crate::primitives::{Point2, Segment2};
use num_traits::Float;

/// Result of a curve intersection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveIntersection<F> {
    /// The intersection point.
    pub point: Point2<F>,
    /// Parameter on the first curve (0 to 1).
    pub t1: F,
    /// Parameter on the second curve (0 to 1).
    pub t2: F,
}

/// Finds intersections between two quadratic Bézier curves.
///
/// Uses recursive subdivision with bounding box culling.
///
/// # Arguments
///
/// * `curve1` - First quadratic Bézier curve
/// * `curve2` - Second quadratic Bézier curve
/// * `tolerance` - Spatial tolerance for considering curves as intersecting
///
/// # Returns
///
/// A vector of intersection points with parameters on both curves.
pub fn intersect_quadratic_quadratic<F: Float>(
    curve1: &QuadraticBezier2<F>,
    curve2: &QuadraticBezier2<F>,
    tolerance: F,
) -> Vec<CurveIntersection<F>> {
    let mut intersections = Vec::new();
    intersect_quadratic_recursive(
        curve1,
        F::zero(),
        F::one(),
        curve2,
        F::zero(),
        F::one(),
        tolerance,
        &mut intersections,
        0,
    );
    deduplicate_intersections(&mut intersections, tolerance);
    intersections
}

/// Finds intersections between two cubic Bézier curves.
///
/// Uses recursive subdivision with bounding box culling.
///
/// # Arguments
///
/// * `curve1` - First cubic Bézier curve
/// * `curve2` - Second cubic Bézier curve
/// * `tolerance` - Spatial tolerance for considering curves as intersecting
///
/// # Returns
///
/// A vector of intersection points with parameters on both curves.
///
/// # Example
///
/// ```
/// use approxum::{Point2, curves::{CubicBezier2, intersect_cubic_cubic}};
///
/// // Two curves that cross
/// let curve1 = CubicBezier2::new(
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 3.0),
///     Point2::new(3.0, 3.0),
///     Point2::new(4.0, 0.0),
/// );
///
/// let curve2 = CubicBezier2::new(
///     Point2::new(2.0, -1.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(2.0, 3.0),
/// );
///
/// let intersections = intersect_cubic_cubic(&curve1, &curve2, 1e-6);
/// assert!(!intersections.is_empty());
/// ```
pub fn intersect_cubic_cubic<F: Float>(
    curve1: &CubicBezier2<F>,
    curve2: &CubicBezier2<F>,
    tolerance: F,
) -> Vec<CurveIntersection<F>> {
    let mut intersections = Vec::new();
    intersect_cubic_recursive(
        curve1,
        F::zero(),
        F::one(),
        curve2,
        F::zero(),
        F::one(),
        tolerance,
        &mut intersections,
        0,
    );
    deduplicate_intersections(&mut intersections, tolerance);
    intersections
}

/// Finds intersections between a quadratic Bézier curve and a line segment.
///
/// # Arguments
///
/// * `curve` - The quadratic Bézier curve
/// * `segment` - The line segment
/// * `tolerance` - Tolerance for endpoint detection
///
/// # Returns
///
/// A vector of (point, t_curve, t_segment) tuples where t values are in [0, 1].
pub fn intersect_quadratic_segment<F: Float>(
    curve: &QuadraticBezier2<F>,
    segment: &Segment2<F>,
    tolerance: F,
) -> Vec<(Point2<F>, F, F)> {
    // Convert segment to implicit form: ax + by + c = 0
    let dx = segment.end.x - segment.start.x;
    let dy = segment.end.y - segment.start.y;

    let a = -dy;
    let b = dx;
    let c = dy * segment.start.x - dx * segment.start.y;

    // Substitute Bézier parameterization into line equation
    // B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    // We get a quadratic in t: At² + Bt + C = 0

    let p0_val = a * curve.p0.x + b * curve.p0.y + c;
    let p1_val = a * curve.p1.x + b * curve.p1.y + c;
    let p2_val = a * curve.p2.x + b * curve.p2.y + c;

    let coeff_a = p0_val - F::from(2.0).unwrap() * p1_val + p2_val;
    let coeff_b = F::from(2.0).unwrap() * (p1_val - p0_val);
    let coeff_c = p0_val;

    let t_values = solve_quadratic(coeff_a, coeff_b, coeff_c);

    let mut results = Vec::new();
    let seg_len_sq = dx * dx + dy * dy;

    for t_curve in t_values {
        if t_curve >= -tolerance && t_curve <= F::one() + tolerance {
            let t_curve = t_curve.max(F::zero()).min(F::one());
            let point = curve.eval(t_curve);

            // Find parameter on segment
            let t_seg = if seg_len_sq > F::epsilon() {
                let px = point.x - segment.start.x;
                let py = point.y - segment.start.y;
                (px * dx + py * dy) / seg_len_sq
            } else {
                F::zero()
            };

            if t_seg >= -tolerance && t_seg <= F::one() + tolerance {
                let t_seg = t_seg.max(F::zero()).min(F::one());
                results.push((point, t_curve, t_seg));
            }
        }
    }

    results
}

/// Finds intersections between a cubic Bézier curve and a line segment.
///
/// # Arguments
///
/// * `curve` - The cubic Bézier curve
/// * `segment` - The line segment
/// * `tolerance` - Tolerance for endpoint detection
///
/// # Returns
///
/// A vector of (point, t_curve, t_segment) tuples.
///
/// # Example
///
/// ```
/// use approxum::{Point2, Segment2, curves::{CubicBezier2, intersect_cubic_segment}};
///
/// let curve = CubicBezier2::new(
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 2.0),
///     Point2::new(3.0, 2.0),
///     Point2::new(4.0, 0.0),
/// );
///
/// let segment = Segment2::new(Point2::new(0.0, 1.0), Point2::new(4.0, 1.0));
///
/// let intersections = intersect_cubic_segment(&curve, &segment, 1e-6);
/// assert_eq!(intersections.len(), 2); // Curve crosses horizontal line twice
/// ```
pub fn intersect_cubic_segment<F: Float>(
    curve: &CubicBezier2<F>,
    segment: &Segment2<F>,
    tolerance: F,
) -> Vec<(Point2<F>, F, F)> {
    // Convert segment to implicit form: ax + by + c = 0
    let dx = segment.end.x - segment.start.x;
    let dy = segment.end.y - segment.start.y;

    let a = -dy;
    let b = dx;
    let c = dy * segment.start.x - dx * segment.start.y;

    // Substitute Bézier parameterization into line equation
    // B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    // We get a cubic in t

    let p0_val = a * curve.p0.x + b * curve.p0.y + c;
    let p1_val = a * curve.p1.x + b * curve.p1.y + c;
    let p2_val = a * curve.p2.x + b * curve.p2.y + c;
    let p3_val = a * curve.p3.x + b * curve.p3.y + c;

    // Coefficients of the cubic: at³ + bt² + ct + d = 0
    let three = F::from(3.0).unwrap();
    let coeff_a = -p0_val + three * p1_val - three * p2_val + p3_val;
    let coeff_b = three * p0_val - F::from(6.0).unwrap() * p1_val + three * p2_val;
    let coeff_c = -three * p0_val + three * p1_val;
    let coeff_d = p0_val;

    let t_values = solve_cubic(coeff_a, coeff_b, coeff_c, coeff_d);

    let mut results = Vec::new();
    let seg_len_sq = dx * dx + dy * dy;

    for t_curve in t_values {
        if t_curve >= -tolerance && t_curve <= F::one() + tolerance {
            let t_curve = t_curve.max(F::zero()).min(F::one());
            let point = curve.eval(t_curve);

            // Find parameter on segment
            let t_seg = if seg_len_sq > F::epsilon() {
                let px = point.x - segment.start.x;
                let py = point.y - segment.start.y;
                (px * dx + py * dy) / seg_len_sq
            } else {
                F::zero()
            };

            if t_seg >= -tolerance && t_seg <= F::one() + tolerance {
                let t_seg = t_seg.max(F::zero()).min(F::one());
                results.push((point, t_curve, t_seg));
            }
        }
    }

    results
}

/// Checks if two quadratic Bézier curves intersect (without computing intersection points).
///
/// More efficient than `intersect_quadratic_quadratic` when you only need a boolean result.
pub fn quadratics_intersect<F: Float>(
    curve1: &QuadraticBezier2<F>,
    curve2: &QuadraticBezier2<F>,
    tolerance: F,
) -> bool {
    quadratics_intersect_recursive(curve1, curve2, tolerance, 0)
}

/// Checks if two cubic Bézier curves intersect (without computing intersection points).
///
/// More efficient than `intersect_cubic_cubic` when you only need a boolean result.
pub fn cubics_intersect<F: Float>(
    curve1: &CubicBezier2<F>,
    curve2: &CubicBezier2<F>,
    tolerance: F,
) -> bool {
    cubics_intersect_recursive(curve1, curve2, tolerance, 0)
}

/// Self-intersection detection for a cubic Bézier curve.
///
/// Cubic Bézier curves can have at most one self-intersection (loop).
///
/// # Returns
///
/// `Some((t1, t2, point))` if the curve self-intersects at parameters t1 and t2,
/// `None` otherwise.
pub fn cubic_self_intersection<F: Float>(
    curve: &CubicBezier2<F>,
    tolerance: F,
) -> Option<(F, F, Point2<F>)> {
    // Split curve into segments and check non-adjacent pairs
    let num_segments = 4usize;
    let step = F::one() / F::from(num_segments).unwrap();

    let mut segments: Vec<(CubicBezier2<F>, F, F)> = Vec::new();
    let mut remaining = *curve;
    let mut t_start = F::zero();

    for i in 0..num_segments {
        if i == num_segments - 1 {
            segments.push((remaining, t_start, F::one()));
        } else {
            let split_t = F::one() / F::from(num_segments - i).unwrap();
            let (left, right) = remaining.split(split_t);
            let t_end = t_start + step;
            segments.push((left, t_start, t_end));
            remaining = right;
            t_start = t_end;
        }
    }

    // Check all non-adjacent segment pairs
    for i in 0..segments.len() {
        for j in (i + 2)..segments.len() {
            let (seg_i, t_i_start, t_i_end) = &segments[i];
            let (seg_j, t_j_start, t_j_end) = &segments[j];

            let intersections = intersect_cubic_cubic(seg_i, seg_j, tolerance);

            for int in intersections {
                let t1_orig = *t_i_start + int.t1 * (*t_i_end - *t_i_start);
                let t2_orig = *t_j_start + int.t2 * (*t_j_end - *t_j_start);

                // Skip endpoint connections
                if (t1_orig - t2_orig).abs() > tolerance * F::from(10.0).unwrap() {
                    let t_min = t1_orig.min(t2_orig);
                    let t_max = t1_orig.max(t2_orig);
                    return Some((t_min, t_max, int.point));
                }
            }
        }
    }

    None
}

// ============================================================================
// Internal implementation
// ============================================================================

const MAX_RECURSION_DEPTH: usize = 50;

#[allow(clippy::too_many_arguments)]
fn intersect_quadratic_recursive<F: Float>(
    curve1: &QuadraticBezier2<F>,
    t1_min: F,
    t1_max: F,
    curve2: &QuadraticBezier2<F>,
    t2_min: F,
    t2_max: F,
    tolerance: F,
    results: &mut Vec<CurveIntersection<F>>,
    depth: usize,
) {
    if depth > MAX_RECURSION_DEPTH {
        return;
    }

    // Bounding box check
    let bounds1 = quadratic_bounds(curve1);
    let bounds2 = quadratic_bounds(curve2);

    if !bounds1.intersects(bounds2) {
        return;
    }

    // Check if curves are flat enough to be treated as line segments
    let flat1 = curve1.flatness() < tolerance;
    let flat2 = curve2.flatness() < tolerance;

    if flat1 && flat2 {
        // Intersect as line segments
        let seg1 = Segment2::new(curve1.p0, curve1.p2);
        let seg2 = Segment2::new(curve2.p0, curve2.p2);

        if let Some((point, s1, s2)) = segment_segment_intersection(&seg1, &seg2) {
            let t1 = t1_min + s1 * (t1_max - t1_min);
            let t2 = t2_min + s2 * (t2_max - t2_min);
            results.push(CurveIntersection { point, t1, t2 });
        }
        return;
    }

    // Subdivide the less flat curve (or both if similar)
    let half = F::from(0.5).unwrap();

    if !flat1 {
        let (left1, right1) = curve1.split(half);
        let t1_mid = t1_min + half * (t1_max - t1_min);

        if flat2 {
            intersect_quadratic_recursive(
                &left1,
                t1_min,
                t1_mid,
                curve2,
                t2_min,
                t2_max,
                tolerance,
                results,
                depth + 1,
            );
            intersect_quadratic_recursive(
                &right1,
                t1_mid,
                t1_max,
                curve2,
                t2_min,
                t2_max,
                tolerance,
                results,
                depth + 1,
            );
        } else {
            let (left2, right2) = curve2.split(half);
            let t2_mid = t2_min + half * (t2_max - t2_min);

            intersect_quadratic_recursive(
                &left1,
                t1_min,
                t1_mid,
                &left2,
                t2_min,
                t2_mid,
                tolerance,
                results,
                depth + 1,
            );
            intersect_quadratic_recursive(
                &left1,
                t1_min,
                t1_mid,
                &right2,
                t2_mid,
                t2_max,
                tolerance,
                results,
                depth + 1,
            );
            intersect_quadratic_recursive(
                &right1,
                t1_mid,
                t1_max,
                &left2,
                t2_min,
                t2_mid,
                tolerance,
                results,
                depth + 1,
            );
            intersect_quadratic_recursive(
                &right1,
                t1_mid,
                t1_max,
                &right2,
                t2_mid,
                t2_max,
                tolerance,
                results,
                depth + 1,
            );
        }
    } else {
        // curve1 is flat, curve2 is not
        let (left2, right2) = curve2.split(half);
        let t2_mid = t2_min + half * (t2_max - t2_min);

        intersect_quadratic_recursive(
            curve1,
            t1_min,
            t1_max,
            &left2,
            t2_min,
            t2_mid,
            tolerance,
            results,
            depth + 1,
        );
        intersect_quadratic_recursive(
            curve1,
            t1_min,
            t1_max,
            &right2,
            t2_mid,
            t2_max,
            tolerance,
            results,
            depth + 1,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn intersect_cubic_recursive<F: Float>(
    curve1: &CubicBezier2<F>,
    t1_min: F,
    t1_max: F,
    curve2: &CubicBezier2<F>,
    t2_min: F,
    t2_max: F,
    tolerance: F,
    results: &mut Vec<CurveIntersection<F>>,
    depth: usize,
) {
    if depth > MAX_RECURSION_DEPTH {
        return;
    }

    // Bounding box check
    let bounds1 = cubic_bounds(curve1);
    let bounds2 = cubic_bounds(curve2);

    if !bounds1.intersects(bounds2) {
        return;
    }

    // Check if curves are flat enough to be treated as line segments
    let flat1 = curve1.flatness() < tolerance;
    let flat2 = curve2.flatness() < tolerance;

    if flat1 && flat2 {
        // Intersect as line segments
        let seg1 = Segment2::new(curve1.p0, curve1.p3);
        let seg2 = Segment2::new(curve2.p0, curve2.p3);

        if let Some((point, s1, s2)) = segment_segment_intersection(&seg1, &seg2) {
            let t1 = t1_min + s1 * (t1_max - t1_min);
            let t2 = t2_min + s2 * (t2_max - t2_min);
            results.push(CurveIntersection { point, t1, t2 });
        }
        return;
    }

    // Subdivide
    let half = F::from(0.5).unwrap();

    if !flat1 {
        let (left1, right1) = curve1.split(half);
        let t1_mid = t1_min + half * (t1_max - t1_min);

        if flat2 {
            intersect_cubic_recursive(
                &left1,
                t1_min,
                t1_mid,
                curve2,
                t2_min,
                t2_max,
                tolerance,
                results,
                depth + 1,
            );
            intersect_cubic_recursive(
                &right1,
                t1_mid,
                t1_max,
                curve2,
                t2_min,
                t2_max,
                tolerance,
                results,
                depth + 1,
            );
        } else {
            let (left2, right2) = curve2.split(half);
            let t2_mid = t2_min + half * (t2_max - t2_min);

            intersect_cubic_recursive(
                &left1,
                t1_min,
                t1_mid,
                &left2,
                t2_min,
                t2_mid,
                tolerance,
                results,
                depth + 1,
            );
            intersect_cubic_recursive(
                &left1,
                t1_min,
                t1_mid,
                &right2,
                t2_mid,
                t2_max,
                tolerance,
                results,
                depth + 1,
            );
            intersect_cubic_recursive(
                &right1,
                t1_mid,
                t1_max,
                &left2,
                t2_min,
                t2_mid,
                tolerance,
                results,
                depth + 1,
            );
            intersect_cubic_recursive(
                &right1,
                t1_mid,
                t1_max,
                &right2,
                t2_mid,
                t2_max,
                tolerance,
                results,
                depth + 1,
            );
        }
    } else {
        let (left2, right2) = curve2.split(half);
        let t2_mid = t2_min + half * (t2_max - t2_min);

        intersect_cubic_recursive(
            curve1,
            t1_min,
            t1_max,
            &left2,
            t2_min,
            t2_mid,
            tolerance,
            results,
            depth + 1,
        );
        intersect_cubic_recursive(
            curve1,
            t1_min,
            t1_max,
            &right2,
            t2_mid,
            t2_max,
            tolerance,
            results,
            depth + 1,
        );
    }
}

fn quadratics_intersect_recursive<F: Float>(
    curve1: &QuadraticBezier2<F>,
    curve2: &QuadraticBezier2<F>,
    tolerance: F,
    depth: usize,
) -> bool {
    if depth > MAX_RECURSION_DEPTH {
        return false;
    }

    let bounds1 = quadratic_bounds(curve1);
    let bounds2 = quadratic_bounds(curve2);

    if !bounds1.intersects(bounds2) {
        return false;
    }

    let flat1 = curve1.flatness() < tolerance;
    let flat2 = curve2.flatness() < tolerance;

    if flat1 && flat2 {
        let seg1 = Segment2::new(curve1.p0, curve1.p2);
        let seg2 = Segment2::new(curve2.p0, curve2.p2);
        return segment_segment_intersection(&seg1, &seg2).is_some();
    }

    let half = F::from(0.5).unwrap();

    if !flat1 {
        let (left1, right1) = curve1.split(half);
        if flat2 {
            quadratics_intersect_recursive(&left1, curve2, tolerance, depth + 1)
                || quadratics_intersect_recursive(&right1, curve2, tolerance, depth + 1)
        } else {
            let (left2, right2) = curve2.split(half);
            quadratics_intersect_recursive(&left1, &left2, tolerance, depth + 1)
                || quadratics_intersect_recursive(&left1, &right2, tolerance, depth + 1)
                || quadratics_intersect_recursive(&right1, &left2, tolerance, depth + 1)
                || quadratics_intersect_recursive(&right1, &right2, tolerance, depth + 1)
        }
    } else {
        let (left2, right2) = curve2.split(half);
        quadratics_intersect_recursive(curve1, &left2, tolerance, depth + 1)
            || quadratics_intersect_recursive(curve1, &right2, tolerance, depth + 1)
    }
}

fn cubics_intersect_recursive<F: Float>(
    curve1: &CubicBezier2<F>,
    curve2: &CubicBezier2<F>,
    tolerance: F,
    depth: usize,
) -> bool {
    if depth > MAX_RECURSION_DEPTH {
        return false;
    }

    let bounds1 = cubic_bounds(curve1);
    let bounds2 = cubic_bounds(curve2);

    if !bounds1.intersects(bounds2) {
        return false;
    }

    let flat1 = curve1.flatness() < tolerance;
    let flat2 = curve2.flatness() < tolerance;

    if flat1 && flat2 {
        let seg1 = Segment2::new(curve1.p0, curve1.p3);
        let seg2 = Segment2::new(curve2.p0, curve2.p3);
        return segment_segment_intersection(&seg1, &seg2).is_some();
    }

    let half = F::from(0.5).unwrap();

    if !flat1 {
        let (left1, right1) = curve1.split(half);
        if flat2 {
            cubics_intersect_recursive(&left1, curve2, tolerance, depth + 1)
                || cubics_intersect_recursive(&right1, curve2, tolerance, depth + 1)
        } else {
            let (left2, right2) = curve2.split(half);
            cubics_intersect_recursive(&left1, &left2, tolerance, depth + 1)
                || cubics_intersect_recursive(&left1, &right2, tolerance, depth + 1)
                || cubics_intersect_recursive(&right1, &left2, tolerance, depth + 1)
                || cubics_intersect_recursive(&right1, &right2, tolerance, depth + 1)
        }
    } else {
        let (left2, right2) = curve2.split(half);
        cubics_intersect_recursive(curve1, &left2, tolerance, depth + 1)
            || cubics_intersect_recursive(curve1, &right2, tolerance, depth + 1)
    }
}

fn quadratic_bounds<F: Float>(curve: &QuadraticBezier2<F>) -> Aabb2<F> {
    let (min, max) = curve.control_bounds();
    Aabb2::new(min, max)
}

fn cubic_bounds<F: Float>(curve: &CubicBezier2<F>) -> Aabb2<F> {
    let (min, max) = curve.control_bounds();
    Aabb2::new(min, max)
}

fn segment_segment_intersection<F: Float>(
    seg1: &Segment2<F>,
    seg2: &Segment2<F>,
) -> Option<(Point2<F>, F, F)> {
    let d1x = seg1.end.x - seg1.start.x;
    let d1y = seg1.end.y - seg1.start.y;
    let d2x = seg2.end.x - seg2.start.x;
    let d2y = seg2.end.y - seg2.start.y;

    let cross = d1x * d2y - d1y * d2x;

    if cross.abs() < F::epsilon() {
        return None; // Parallel
    }

    let dx = seg2.start.x - seg1.start.x;
    let dy = seg2.start.y - seg1.start.y;

    let t1 = (dx * d2y - dy * d2x) / cross;
    let t2 = (dx * d1y - dy * d1x) / cross;

    if t1 >= F::zero() && t1 <= F::one() && t2 >= F::zero() && t2 <= F::one() {
        let point = Point2::new(seg1.start.x + t1 * d1x, seg1.start.y + t1 * d1y);
        Some((point, t1, t2))
    } else {
        None
    }
}

fn deduplicate_intersections<F: Float>(
    intersections: &mut Vec<CurveIntersection<F>>,
    tolerance: F,
) {
    if intersections.len() <= 1 {
        return;
    }

    // Sort by t1 parameter
    intersections.sort_by(|a, b| a.t1.partial_cmp(&b.t1).unwrap_or(std::cmp::Ordering::Equal));

    // Remove duplicates
    let mut write = 0;
    for read in 1..intersections.len() {
        let dist = intersections[write]
            .point
            .distance(intersections[read].point);
        if dist > tolerance {
            write += 1;
            intersections[write] = intersections[read];
        }
    }
    intersections.truncate(write + 1);
}

/// Solves the quadratic equation ax² + bx + c = 0.
fn solve_quadratic<F: Float>(a: F, b: F, c: F) -> Vec<F> {
    let eps = F::epsilon() * F::from(1000.0).unwrap();

    if a.abs() < eps {
        if b.abs() < eps {
            return vec![];
        }
        return vec![-c / b];
    }

    let two = F::one() + F::one();
    let four = two + two;
    let discriminant = b * b - four * a * c;

    if discriminant < F::zero() {
        vec![]
    } else if discriminant < eps {
        vec![-b / (two * a)]
    } else {
        let sqrt_d = discriminant.sqrt();
        vec![(-b - sqrt_d) / (two * a), (-b + sqrt_d) / (two * a)]
    }
}

/// Solves the cubic equation ax³ + bx² + cx + d = 0.
fn solve_cubic<F: Float>(a: F, b: F, c: F, d: F) -> Vec<F> {
    let eps = F::epsilon() * F::from(1000.0).unwrap();

    // If a ≈ 0, reduce to quadratic
    if a.abs() < eps {
        return solve_quadratic(b, c, d);
    }

    // Normalize: x³ + px² + qx + r = 0
    let p = b / a;
    let q = c / a;
    let r = d / a;

    // Substitute x = t - p/3 to get t³ + pt + q = 0 (depressed cubic)
    let three = F::from(3.0).unwrap();
    let one_third = F::one() / three;
    let p_over_3 = p * one_third;

    let p_new = q - p * p * one_third;
    let q_new = r - p * q * one_third + F::from(2.0).unwrap() * p * p * p / F::from(27.0).unwrap();

    // Use Cardano's formula
    let half = F::from(0.5).unwrap();
    let discriminant =
        q_new * q_new / F::from(4.0).unwrap() + p_new * p_new * p_new / F::from(27.0).unwrap();

    let mut roots = Vec::new();

    if discriminant > eps {
        // One real root
        let sqrt_d = discriminant.sqrt();
        let u = (-half * q_new + sqrt_d).cbrt();
        let v = (-half * q_new - sqrt_d).cbrt();
        roots.push(u + v - p_over_3);
    } else if discriminant < -eps {
        // Three real roots (casus irreducibilis)
        let m = (-p_new / three).sqrt();
        let theta = (F::from(-3.0).unwrap() * q_new / (F::from(2.0).unwrap() * p_new * m)).acos();

        let two = F::from(2.0).unwrap();
        let two_pi = F::from(std::f64::consts::PI).unwrap() * two;

        roots.push(two * m * (theta * one_third).cos() - p_over_3);
        roots.push(two * m * ((theta + two_pi) * one_third).cos() - p_over_3);
        roots.push(two * m * ((theta - two_pi) * one_third).cos() - p_over_3);
    } else {
        // Discriminant ≈ 0: repeated roots
        if q_new.abs() < eps {
            roots.push(-p_over_3);
        } else {
            let u = (-half * q_new).cbrt();
            roots.push(F::from(2.0).unwrap() * u - p_over_3);
            roots.push(-u - p_over_3);
        }
    }

    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cubic_cubic_intersection_cross() {
        // Horizontal arch
        let curve1: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        // Vertical line-like curve
        let curve2: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(2.0, -1.0),
            Point2::new(2.0, 0.5),
            Point2::new(2.0, 1.5),
            Point2::new(2.0, 3.0),
        );

        let intersections = intersect_cubic_cubic(&curve1, &curve2, 1e-6);

        assert_eq!(intersections.len(), 1);
        assert_relative_eq!(intersections[0].point.x, 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_cubic_cubic_no_intersection() {
        let curve1: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(3.0, 0.0),
        );

        let curve2: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 5.0),
            Point2::new(1.0, 6.0),
            Point2::new(2.0, 6.0),
            Point2::new(3.0, 5.0),
        );

        let intersections = intersect_cubic_cubic(&curve1, &curve2, 1e-6);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_cubic_cubic_two_intersections() {
        // Arch curve
        let curve1: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 4.0),
            Point2::new(4.0, 4.0),
            Point2::new(6.0, 0.0),
        );

        // Inverted arch
        let curve2: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 2.0),
            Point2::new(2.0, -2.0),
            Point2::new(4.0, -2.0),
            Point2::new(6.0, 2.0),
        );

        let intersections = intersect_cubic_cubic(&curve1, &curve2, 1e-6);
        assert_eq!(intersections.len(), 2);
    }

    #[test]
    fn test_cubic_segment_intersection() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        // Horizontal line at y=1
        let segment = Segment2::new(Point2::new(-1.0, 1.0), Point2::new(5.0, 1.0));

        let intersections = intersect_cubic_segment(&curve, &segment, 1e-6);

        // Curve should cross horizontal line twice
        assert_eq!(intersections.len(), 2);

        for (point, t_curve, _t_seg) in &intersections {
            assert_relative_eq!(point.y, 1.0, epsilon = 0.01);
            assert!(*t_curve >= 0.0 && *t_curve <= 1.0);
        }
    }

    #[test]
    fn test_cubic_segment_no_intersection() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(3.0, 0.0),
        );

        // Segment above curve
        let segment = Segment2::new(Point2::new(0.0, 5.0), Point2::new(3.0, 5.0));

        let intersections = intersect_cubic_segment(&curve, &segment, 1e-6);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_quadratic_quadratic_intersection() {
        let curve1: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 4.0),
            Point2::new(4.0, 0.0),
        );

        let curve2: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 2.0),
            Point2::new(2.0, -2.0),
            Point2::new(4.0, 2.0),
        );

        let intersections = intersect_quadratic_quadratic(&curve1, &curve2, 1e-6);
        assert_eq!(intersections.len(), 2);
    }

    #[test]
    fn test_quadratic_segment_intersection() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 4.0),
            Point2::new(4.0, 0.0),
        );

        let segment = Segment2::new(Point2::new(0.0, 1.0), Point2::new(4.0, 1.0));

        let intersections = intersect_quadratic_segment(&curve, &segment, 1e-6);
        assert_eq!(intersections.len(), 2);

        for (point, _, _) in &intersections {
            assert_relative_eq!(point.y, 1.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_cubics_intersect_bool() {
        let curve1: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let curve2: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(2.0, -1.0),
            Point2::new(2.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(2.0, 3.0),
        );

        assert!(cubics_intersect(&curve1, &curve2, 1e-6));

        // Non-intersecting
        let curve3: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(10.0, 10.0),
            Point2::new(11.0, 12.0),
            Point2::new(13.0, 12.0),
            Point2::new(14.0, 10.0),
        );

        assert!(!cubics_intersect(&curve1, &curve3, 1e-6));
    }

    #[test]
    fn test_quadratics_intersect_bool() {
        let curve1: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 4.0),
            Point2::new(4.0, 0.0),
        );

        let curve2: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 2.0),
            Point2::new(2.0, -2.0),
            Point2::new(4.0, 2.0),
        );

        assert!(quadratics_intersect(&curve1, &curve2, 1e-6));

        // Non-intersecting
        let curve3: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(10.0, 10.0),
            Point2::new(12.0, 14.0),
            Point2::new(14.0, 10.0),
        );

        assert!(!quadratics_intersect(&curve1, &curve3, 1e-6));
    }

    #[test]
    fn test_cubic_self_intersection() {
        // A loop curve that crosses itself - control points cross over
        let loop_curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(3.0, 3.0),
            Point2::new(-1.0, 3.0),
            Point2::new(2.0, 0.0),
        );

        let result = cubic_self_intersection(&loop_curve, 1e-5);
        assert!(result.is_some());

        let (t1, t2, point) = result.unwrap();
        assert!(t1 < t2);
        assert!(t1 >= 0.0 && t1 <= 1.0);
        assert!(t2 >= 0.0 && t2 <= 1.0);

        // Verify both parameters give approximately the same point
        let p1 = loop_curve.eval(t1);
        let p2 = loop_curve.eval(t2);
        assert!(p1.distance(p2) < 0.1);
        assert!(point.distance(p1) < 0.1);
    }

    #[test]
    fn test_cubic_no_self_intersection() {
        // Simple arch - no self-intersection
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let result = cubic_self_intersection(&curve, 1e-6);
        assert!(result.is_none());
    }

    #[test]
    fn test_intersection_at_endpoints() {
        // Two curves sharing an endpoint
        let curve1: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(3.0, 0.0),
        );

        let curve2: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(3.0, 0.0),
            Point2::new(4.0, 1.0),
            Point2::new(5.0, 1.0),
            Point2::new(6.0, 0.0),
        );

        let intersections = intersect_cubic_cubic(&curve1, &curve2, 1e-6);
        assert_eq!(intersections.len(), 1);

        assert_relative_eq!(intersections[0].t1, 1.0, epsilon = 0.01);
        assert_relative_eq!(intersections[0].t2, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_f32_support() {
        let curve1: CubicBezier2<f32> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let curve2: CubicBezier2<f32> = CubicBezier2::new(
            Point2::new(2.0, -1.0),
            Point2::new(2.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(2.0, 3.0),
        );

        let intersections = intersect_cubic_cubic(&curve1, &curve2, 1e-4);
        assert!(!intersections.is_empty());
    }

    #[test]
    fn test_parallel_curves() {
        // Two parallel curves that don't intersect
        let curve1: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        );

        let curve2: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(3.0, 1.0),
        );

        let intersections = intersect_cubic_cubic(&curve1, &curve2, 1e-6);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_solve_quadratic() {
        // x² - 3x + 2 = 0 => x = 1, 2
        let roots = solve_quadratic(1.0, -3.0, 2.0);
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&1.0) || roots.iter().any(|&r| (r - 1.0).abs() < 1e-10));
        assert!(roots.contains(&2.0) || roots.iter().any(|&r| (r - 2.0).abs() < 1e-10));
    }

    #[test]
    fn test_solve_cubic() {
        // x³ - 6x² + 11x - 6 = 0 => x = 1, 2, 3
        let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
        assert_eq!(roots.len(), 3);

        let mut sorted = roots.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_relative_eq!(sorted[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(sorted[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(sorted[2], 3.0, epsilon = 1e-6);
    }
}
