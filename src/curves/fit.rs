//! Curve fitting algorithms.
//!
//! Fits Bézier curves to point data using least-squares optimization.

use crate::primitives::Point2;
use crate::curves::CubicBezier2;
use num_traits::Float;

/// Result of a curve fitting operation.
#[derive(Debug, Clone)]
pub struct FitResult<F> {
    /// The fitted curve.
    pub curve: CubicBezier2<F>,
    /// Maximum error (distance from any point to the curve).
    pub max_error: F,
    /// Index of the point with maximum error.
    pub max_error_index: usize,
}

/// Fits a cubic Bézier curve to a sequence of points.
///
/// Uses least-squares fitting with chord-length parameterization.
/// The curve endpoints are fixed to the first and last input points.
///
/// # Arguments
///
/// * `points` - The points to fit (must have at least 2 points)
///
/// # Returns
///
/// The fitted curve, or `None` if fitting is not possible (< 2 points).
///
/// # Example
///
/// ```
/// use approxum::{Point2, curves::fit_cubic};
///
/// let points = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 2.0),
///     Point2::new(2.0, 2.5),
///     Point2::new(3.0, 1.0),
///     Point2::new(4.0, 0.0),
/// ];
///
/// let result = fit_cubic(&points).unwrap();
/// // result.curve approximates the points
/// // result.max_error tells us how well it fits
/// ```
pub fn fit_cubic<F: Float>(points: &[Point2<F>]) -> Option<FitResult<F>> {
    let n = points.len();

    if n < 2 {
        return None;
    }

    if n == 2 {
        // Degenerate case: straight line
        let p0 = points[0];
        let p3 = points[1];
        let third = F::one() / (F::one() + F::one() + F::one());
        let p1 = p0.lerp(p3, third);
        let p2 = p0.lerp(p3, third + third);
        return Some(FitResult {
            curve: CubicBezier2::new(p0, p1, p2, p3),
            max_error: F::zero(),
            max_error_index: 0,
        });
    }

    // Compute chord-length parameterization
    let params = chord_length_parameterize(points);

    // Fit using least squares
    let curve = fit_cubic_internal(points, &params);

    // Compute error
    let (max_error, max_error_index) = compute_max_error(points, &curve, &params);

    Some(FitResult {
        curve,
        max_error,
        max_error_index,
    })
}

/// Fits a cubic Bézier with iterative refinement.
///
/// Iteratively improves the fit by reparameterizing based on the current curve.
///
/// # Arguments
///
/// * `points` - The points to fit
/// * `tolerance` - Target maximum error
/// * `max_iterations` - Maximum refinement iterations
///
/// # Returns
///
/// The best fit found within the iteration limit.
pub fn fit_cubic_iterative<F: Float>(
    points: &[Point2<F>],
    tolerance: F,
    max_iterations: usize,
) -> Option<FitResult<F>> {
    let n = points.len();

    if n < 2 {
        return None;
    }

    if n == 2 {
        return fit_cubic(points);
    }

    // Initial parameterization
    let mut params = chord_length_parameterize(points);
    let mut curve = fit_cubic_internal(points, &params);
    let (mut max_error, mut max_error_index) = compute_max_error(points, &curve, &params);

    // Iterative refinement
    for _ in 0..max_iterations {
        if max_error <= tolerance {
            break;
        }

        // Reparameterize using Newton-Raphson
        params = reparameterize(points, &curve, &params);
        curve = fit_cubic_internal(points, &params);
        let (new_error, new_index) = compute_max_error(points, &curve, &params);

        // Check for improvement
        if new_error >= max_error {
            break; // No improvement, stop
        }

        max_error = new_error;
        max_error_index = new_index;
    }

    Some(FitResult {
        curve,
        max_error,
        max_error_index,
    })
}

/// Computes chord-length parameterization for a set of points.
fn chord_length_parameterize<F: Float>(points: &[Point2<F>]) -> Vec<F> {
    let n = points.len();
    let mut params = vec![F::zero(); n];

    if n <= 1 {
        return params;
    }

    // Compute cumulative chord lengths
    let mut total = F::zero();
    for i in 1..n {
        total = total + points[i - 1].distance(points[i]);
        params[i] = total;
    }

    // Normalize to [0, 1]
    if total > F::epsilon() {
        for i in 1..n {
            params[i] = params[i] / total;
        }
    }

    // Ensure last parameter is exactly 1
    params[n - 1] = F::one();

    params
}

/// Fits a cubic Bézier given fixed parameterization.
fn fit_cubic_internal<F: Float>(points: &[Point2<F>], params: &[F]) -> CubicBezier2<F> {
    let n = points.len();
    let p0 = points[0];
    let p3 = points[n - 1];

    if n == 2 {
        let third = F::one() / (F::one() + F::one() + F::one());
        return CubicBezier2::new(
            p0,
            p0.lerp(p3, third),
            p0.lerp(p3, third + third),
            p3,
        );
    }

    let one = F::one();
    let three = one + one + one;

    // Build least-squares system for P1 and P2
    // B(t) = (1-t)³P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³P3
    //
    // For each point, we have:
    // P[i] ≈ B(t[i])
    // P[i] - (1-t)³P0 - t³P3 = 3(1-t)²t·P1 + 3(1-t)t²·P2
    //
    // Let A1(t) = 3(1-t)²t, A2(t) = 3(1-t)t²
    // Then we solve: [A1, A2] · [P1, P2]ᵀ = RHS

    let mut a11 = F::zero(); // sum of A1²
    let mut a12 = F::zero(); // sum of A1·A2
    let mut a22 = F::zero(); // sum of A2²
    let mut rhs1_x = F::zero();
    let mut rhs1_y = F::zero();
    let mut rhs2_x = F::zero();
    let mut rhs2_y = F::zero();

    for i in 1..n - 1 {
        let t = params[i];
        let mt = one - t;

        let a1 = three * mt * mt * t;      // 3(1-t)²t
        let a2 = three * mt * t * t;       // 3(1-t)t²
        let b0 = mt * mt * mt;             // (1-t)³
        let b3 = t * t * t;                // t³

        // RHS = P[i] - b0·P0 - b3·P3
        let rhs_x = points[i].x - b0 * p0.x - b3 * p3.x;
        let rhs_y = points[i].y - b0 * p0.y - b3 * p3.y;

        a11 = a11 + a1 * a1;
        a12 = a12 + a1 * a2;
        a22 = a22 + a2 * a2;

        rhs1_x = rhs1_x + a1 * rhs_x;
        rhs1_y = rhs1_y + a1 * rhs_y;
        rhs2_x = rhs2_x + a2 * rhs_x;
        rhs2_y = rhs2_y + a2 * rhs_y;
    }

    // Solve 2x2 system: [a11 a12; a12 a22] · [p1; p2] = [rhs1; rhs2]
    let det = a11 * a22 - a12 * a12;

    let (p1, p2) = if det.abs() > F::epsilon() {
        let inv_det = one / det;
        let p1_x = (a22 * rhs1_x - a12 * rhs2_x) * inv_det;
        let p1_y = (a22 * rhs1_y - a12 * rhs2_y) * inv_det;
        let p2_x = (a11 * rhs2_x - a12 * rhs1_x) * inv_det;
        let p2_y = (a11 * rhs2_y - a12 * rhs1_y) * inv_det;
        (Point2::new(p1_x, p1_y), Point2::new(p2_x, p2_y))
    } else {
        // Degenerate case: use thirds
        let third = one / three;
        (p0.lerp(p3, third), p0.lerp(p3, third + third))
    };

    CubicBezier2::new(p0, p1, p2, p3)
}

/// Computes the maximum error between points and a fitted curve.
fn compute_max_error<F: Float>(
    points: &[Point2<F>],
    curve: &CubicBezier2<F>,
    params: &[F],
) -> (F, usize) {
    let mut max_error = F::zero();
    let mut max_index = 0;

    for (i, (&param, &point)) in params.iter().zip(points.iter()).enumerate() {
        let curve_point = curve.eval(param);
        let error = point.distance(curve_point);
        if error > max_error {
            max_error = error;
            max_index = i;
        }
    }

    (max_error, max_index)
}

/// Reparameterizes points using Newton-Raphson iteration.
fn reparameterize<F: Float>(
    points: &[Point2<F>],
    curve: &CubicBezier2<F>,
    params: &[F],
) -> Vec<F> {
    let mut new_params = params.to_vec();

    for (i, (param, point)) in params.iter().zip(points.iter()).enumerate() {
        // Skip endpoints (fixed at 0 and 1)
        if i == 0 || i == points.len() - 1 {
            continue;
        }

        new_params[i] = newton_raphson_root(*point, curve, *param);
    }

    new_params
}

/// Uses Newton-Raphson to find a better parameter value for a point.
fn newton_raphson_root<F: Float>(
    point: Point2<F>,
    curve: &CubicBezier2<F>,
    initial_t: F,
) -> F {
    let mut t = initial_t;
    let one = F::one();

    // A few iterations of Newton-Raphson
    for _ in 0..3 {
        let q = curve.eval(t);
        let q_prime = eval_derivative(curve, t);
        let q_prime2 = eval_second_derivative(curve, t);

        // f(t) = (Q(t) - P) · Q'(t)
        // f'(t) = Q'(t)·Q'(t) + (Q(t) - P)·Q''(t)
        let diff_x = q.x - point.x;
        let diff_y = q.y - point.y;

        let numerator = diff_x * q_prime.x + diff_y * q_prime.y;
        let denominator = q_prime.x * q_prime.x + q_prime.y * q_prime.y
            + diff_x * q_prime2.x + diff_y * q_prime2.y;

        if denominator.abs() <= F::epsilon() {
            break;
        }

        t = t - numerator / denominator;

        // Clamp to valid range
        t = t.max(F::zero()).min(one);
    }

    t
}

/// Evaluates the first derivative of a cubic Bézier at t.
fn eval_derivative<F: Float>(curve: &CubicBezier2<F>, t: F) -> Point2<F> {
    let one = F::one();
    let two = one + one;
    let three = two + one;
    let mt = one - t;

    // B'(t) = 3(1-t)²(P1-P0) + 6(1-t)t(P2-P1) + 3t²(P3-P2)
    let c0 = three * mt * mt;
    let c1 = two * three * mt * t;
    let c2 = three * t * t;

    Point2::new(
        c0 * (curve.p1.x - curve.p0.x)
            + c1 * (curve.p2.x - curve.p1.x)
            + c2 * (curve.p3.x - curve.p2.x),
        c0 * (curve.p1.y - curve.p0.y)
            + c1 * (curve.p2.y - curve.p1.y)
            + c2 * (curve.p3.y - curve.p2.y),
    )
}

/// Evaluates the second derivative of a cubic Bézier at t.
fn eval_second_derivative<F: Float>(curve: &CubicBezier2<F>, t: F) -> Point2<F> {
    let one = F::one();
    let two = one + one;
    let six = two + two + two;
    let mt = one - t;

    // B''(t) = 6(1-t)(P2-2P1+P0) + 6t(P3-2P2+P1)
    let c0 = six * mt;
    let c1 = six * t;

    let d1_x = curve.p2.x - two * curve.p1.x + curve.p0.x;
    let d1_y = curve.p2.y - two * curve.p1.y + curve.p0.y;
    let d2_x = curve.p3.x - two * curve.p2.x + curve.p1.x;
    let d2_y = curve.p3.y - two * curve.p2.y + curve.p1.y;

    Point2::new(c0 * d1_x + c1 * d2_x, c0 * d1_y + c1 * d2_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fit_cubic_two_points() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
        ];

        let result: FitResult<f64> = fit_cubic(&points).unwrap();

        assert_relative_eq!(result.curve.p0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.curve.p3.x, 4.0, epsilon = 1e-10);
        assert_eq!(result.max_error, 0.0);
    }

    #[test]
    fn test_fit_cubic_straight_line() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 3.0),
            Point2::new(4.0, 4.0),
        ];

        let result = fit_cubic(&points).unwrap();

        // Should fit perfectly (or near-perfectly)
        assert!(result.max_error < 0.1);

        // Endpoints should match
        assert_relative_eq!(result.curve.p0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.curve.p3.x, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fit_cubic_curve() {
        // Generate points from a known curve
        let original = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 3.0),
            Point2::new(3.0, 3.0),
            Point2::new(4.0, 0.0),
        );

        let points: Vec<Point2<f64>> = (0..=10)
            .map(|i| original.eval(i as f64 / 10.0))
            .collect();

        let result = fit_cubic(&points).unwrap();

        // Should recover the original curve approximately
        // Note: Least-squares fitting may not perfectly recover the original
        // control points, but should produce a curve that fits the data well
        assert!(result.max_error < 0.5);
    }

    #[test]
    fn test_fit_cubic_iterative() {
        // Generate points from a curve
        let original = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 3.0),
            Point2::new(3.0, 3.0),
            Point2::new(4.0, 0.0),
        );

        let points: Vec<Point2<f64>> = (0..=20)
            .map(|i| original.eval(i as f64 / 20.0))
            .collect();

        let result = fit_cubic_iterative(&points, 0.001, 10).unwrap();

        // Iterative should be more accurate
        assert!(result.max_error < 0.01);
    }

    #[test]
    fn test_fit_cubic_noisy() {
        // Points that don't lie exactly on a Bézier
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.1),
            Point2::new(2.0, 2.9),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        ];

        let result = fit_cubic(&points).unwrap();

        // Should still produce a reasonable fit
        assert!(result.max_error < 1.0);

        // Endpoints should match exactly
        assert_relative_eq!(result.curve.p0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.curve.p0.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.curve.p3.x, 4.0, epsilon = 1e-10);
        assert_relative_eq!(result.curve.p3.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fit_cubic_none_for_empty() {
        let points: Vec<Point2<f64>> = vec![];
        assert!(fit_cubic(&points).is_none());
    }

    #[test]
    fn test_fit_cubic_none_for_single() {
        let points = vec![Point2::new(1.0, 2.0)];
        assert!(fit_cubic(&points).is_none());
    }

    #[test]
    fn test_chord_length_parameterize() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(4.0, 0.0), // Total length = 4
        ];

        let params = chord_length_parameterize(&points);

        assert_relative_eq!(params[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(params[1], 0.25, epsilon = 1e-10);
        assert_relative_eq!(params[2], 0.5, epsilon = 1e-10);
        assert_relative_eq!(params[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fit_cubic_f32() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 0.0),
        ];

        let result = fit_cubic(&points);
        assert!(result.is_some());
    }

    #[test]
    fn test_fit_preserves_endpoints() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(1.5, 2.5),
            Point2::new(2.0, 4.0),
            Point2::new(3.0, 3.5),
            Point2::new(4.5, 1.5),
        ];

        let result = fit_cubic(&points).unwrap();

        // First and last points should be exactly preserved
        assert_relative_eq!(result.curve.p0.x, 1.5, epsilon = 1e-10);
        assert_relative_eq!(result.curve.p0.y, 2.5, epsilon = 1e-10);
        assert_relative_eq!(result.curve.p3.x, 4.5, epsilon = 1e-10);
        assert_relative_eq!(result.curve.p3.y, 1.5, epsilon = 1e-10);
    }
}
