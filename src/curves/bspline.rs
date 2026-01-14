//! B-spline curves with arbitrary degree and knot vectors.
//!
//! B-splines are a generalization of Bézier curves that provide local control
//! over the curve shape. Unlike Bézier curves where moving any control point
//! affects the entire curve, B-spline control points only influence a local
//! region determined by the knot vector.
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, curves::BSpline2};
//!
//! // Create a cubic B-spline (degree 3) with uniform knots
//! let control_points = vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 2.0),
//!     Point2::new(2.0, 2.0),
//!     Point2::new(3.0, 0.0),
//!     Point2::new(4.0, 1.0),
//! ];
//!
//! let spline = BSpline2::with_uniform_knots(control_points, 3);
//!
//! // Evaluate at parameter t
//! let point = spline.eval(0.5);
//! ```

use crate::primitives::Point2;
use num_traits::Float;

/// A B-spline curve in 2D.
///
/// The curve is defined by:
/// - A set of control points
/// - A degree (order - 1)
/// - A knot vector
///
/// The knot vector must satisfy: `knots.len() == control_points.len() + degree + 1`
#[derive(Debug, Clone, PartialEq)]
pub struct BSpline2<F> {
    /// Control points defining the curve shape
    pub control_points: Vec<Point2<F>>,
    /// Degree of the spline (0 = constant, 1 = linear, 2 = quadratic, 3 = cubic)
    pub degree: usize,
    /// Knot vector (non-decreasing sequence)
    pub knots: Vec<F>,
}

impl<F: Float> BSpline2<F> {
    /// Creates a new B-spline with the given control points, degree, and knot vector.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `knots.len() != control_points.len() + degree + 1`
    /// - `degree >= control_points.len()`
    /// - Knot vector is not non-decreasing
    pub fn new(control_points: Vec<Point2<F>>, degree: usize, knots: Vec<F>) -> Self {
        let n = control_points.len();
        assert!(
            degree < n,
            "Degree must be less than number of control points"
        );
        assert!(
            knots.len() == n + degree + 1,
            "Knot vector length must be control_points.len() + degree + 1"
        );

        // Verify knots are non-decreasing
        for i in 1..knots.len() {
            assert!(
                knots[i] >= knots[i - 1],
                "Knot vector must be non-decreasing"
            );
        }

        Self {
            control_points,
            degree,
            knots,
        }
    }

    /// Creates a B-spline with a uniform knot vector.
    ///
    /// The knot vector is clamped (repeated knots at ends) so the curve
    /// passes through the first and last control points.
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, curves::BSpline2};
    ///
    /// let points = vec![
    ///     Point2::new(0.0_f64, 0.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(2.0, 0.0),
    /// ];
    ///
    /// // Quadratic B-spline
    /// let spline = BSpline2::with_uniform_knots(points, 2);
    /// ```
    pub fn with_uniform_knots(control_points: Vec<Point2<F>>, degree: usize) -> Self {
        let n = control_points.len();
        assert!(
            degree < n,
            "Degree must be less than number of control points"
        );

        let num_knots = n + degree + 1;
        let mut knots = Vec::with_capacity(num_knots);

        // Clamped uniform knot vector
        // [0, 0, ..., 0, 1, 2, ..., n-degree-1, n-degree, n-degree, ..., n-degree]
        //  ^-- degree+1 zeros                            ^-- degree+1 at end

        let zero = F::zero();

        // First degree+1 knots are 0
        for _ in 0..=degree {
            knots.push(zero);
        }

        // Interior knots
        let num_interior = n - degree - 1;
        for i in 1..=num_interior {
            knots.push(F::from(i).unwrap());
        }

        // Last degree+1 knots are max value
        let max_knot = F::from(num_interior + 1).unwrap();
        for _ in 0..=degree {
            knots.push(max_knot);
        }

        Self {
            control_points,
            degree,
            knots,
        }
    }

    /// Creates a B-spline with an open uniform knot vector (not clamped).
    ///
    /// Unlike clamped knots, the curve does not pass through the first
    /// and last control points.
    pub fn with_open_uniform_knots(control_points: Vec<Point2<F>>, degree: usize) -> Self {
        let n = control_points.len();
        assert!(
            degree < n,
            "Degree must be less than number of control points"
        );

        let num_knots = n + degree + 1;
        let knots: Vec<F> = (0..num_knots).map(|i| F::from(i).unwrap()).collect();

        Self {
            control_points,
            degree,
            knots,
        }
    }

    /// Returns the valid parameter range [t_min, t_max] for the curve.
    ///
    /// For clamped knot vectors, this is typically [0, max_knot].
    /// For open knot vectors, this is [knots[degree], knots[n]].
    pub fn domain(&self) -> (F, F) {
        let p = self.degree;
        let n = self.control_points.len();
        (self.knots[p], self.knots[n])
    }

    /// Evaluates the B-spline at parameter `t` using de Boor's algorithm.
    ///
    /// # Panics
    ///
    /// Panics if `t` is outside the valid domain.
    pub fn eval(&self, t: F) -> Point2<F> {
        let (t_min, t_max) = self.domain();

        // Clamp t to domain (with small epsilon for numerical stability)
        let eps = F::epsilon() * F::from(100.0).unwrap();
        let t = if t < t_min - eps {
            panic!("Parameter t is below the valid domain");
        } else if t > t_max + eps {
            panic!("Parameter t is above the valid domain");
        } else {
            t.max(t_min).min(t_max)
        };

        // Find knot span index k such that knots[k] <= t < knots[k+1]
        let k = self.find_knot_span(t);

        // de Boor's algorithm
        self.de_boor(t, k)
    }

    /// Evaluates the B-spline at parameter `t`, returning None if outside domain.
    pub fn try_eval(&self, t: F) -> Option<Point2<F>> {
        let (t_min, t_max) = self.domain();
        let eps = F::epsilon() * F::from(100.0).unwrap();

        if t < t_min - eps || t > t_max + eps {
            return None;
        }

        let t = t.max(t_min).min(t_max);
        let k = self.find_knot_span(t);
        Some(self.de_boor(t, k))
    }

    /// Finds the knot span index for parameter t.
    fn find_knot_span(&self, t: F) -> usize {
        let n = self.control_points.len();
        let p = self.degree;

        // Special case: t at the end of the domain
        if t >= self.knots[n] {
            return n - 1;
        }

        // Binary search for knot span
        let mut low = p;
        let mut high = n;

        while low < high {
            let mid = (low + high) / 2;
            if t < self.knots[mid] {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        low - 1
    }

    /// de Boor's algorithm for B-spline evaluation.
    fn de_boor(&self, t: F, k: usize) -> Point2<F> {
        let p = self.degree;

        // Copy the relevant control points
        let mut d: Vec<Point2<F>> = (0..=p).map(|j| self.control_points[k - p + j]).collect();

        // de Boor recursion
        for r in 1..=p {
            for j in (r..=p).rev() {
                let i = k - p + j;
                let denom = self.knots[i + p - r + 1] - self.knots[i];

                let alpha = if denom.abs() < F::epsilon() {
                    F::zero()
                } else {
                    (t - self.knots[i]) / denom
                };

                d[j] = Point2::new(
                    (F::one() - alpha) * d[j - 1].x + alpha * d[j].x,
                    (F::one() - alpha) * d[j - 1].y + alpha * d[j].y,
                );
            }
        }

        d[p]
    }

    /// Converts the B-spline to a polyline using adaptive sampling.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum allowed deviation from the true curve
    pub fn to_polyline(&self, tolerance: F) -> Vec<Point2<F>> {
        let (t_min, t_max) = self.domain();
        let mut points = Vec::new();
        points.push(self.eval(t_min));

        self.subdivide_recursive(t_min, t_max, tolerance, &mut points);
        points
    }

    fn subdivide_recursive(&self, t0: F, t1: F, tolerance: F, points: &mut Vec<Point2<F>>) {
        let two = F::one() + F::one();
        let t_mid = (t0 + t1) / two;

        let p0 = self.eval(t0);
        let p1 = self.eval(t1);
        let p_mid = self.eval(t_mid);

        // Check if the midpoint is close enough to the line segment
        let line_mid = Point2::new((p0.x + p1.x) / two, (p0.y + p1.y) / two);
        let deviation = p_mid.distance(line_mid);

        if deviation <= tolerance {
            points.push(p1);
        } else {
            self.subdivide_recursive(t0, t_mid, tolerance, points);
            self.subdivide_recursive(t_mid, t1, tolerance, points);
        }
    }

    /// Returns the number of Bézier segments this B-spline can be decomposed into.
    pub fn num_bezier_segments(&self) -> usize {
        let n = self.control_points.len();
        let p = self.degree;
        n.saturating_sub(p)
    }

    /// Computes the derivative B-spline.
    ///
    /// The derivative of a degree-p B-spline is a degree-(p-1) B-spline.
    ///
    /// # Panics
    ///
    /// Panics if the spline has degree 0.
    pub fn derivative(&self) -> BSpline2<F> {
        assert!(self.degree > 0, "Cannot take derivative of degree-0 spline");

        let p = self.degree;
        let n = self.control_points.len();

        // New control points: Q_i = p * (P_{i+1} - P_i) / (knots[i+p+1] - knots[i+1])
        let mut new_points = Vec::with_capacity(n - 1);
        let p_f = F::from(p).unwrap();

        for i in 0..n - 1 {
            let denom = self.knots[i + p + 1] - self.knots[i + 1];
            let factor = if denom.abs() < F::epsilon() {
                F::zero()
            } else {
                p_f / denom
            };

            new_points.push(Point2::new(
                factor * (self.control_points[i + 1].x - self.control_points[i].x),
                factor * (self.control_points[i + 1].y - self.control_points[i].y),
            ));
        }

        // New knot vector: remove first and last knot
        let new_knots: Vec<F> = self.knots[1..self.knots.len() - 1].to_vec();

        BSpline2 {
            control_points: new_points,
            degree: p - 1,
            knots: new_knots,
        }
    }

    /// Evaluates the derivative at parameter `t`.
    pub fn derivative_at(&self, t: F) -> Point2<F> {
        if self.degree == 0 {
            return Point2::new(F::zero(), F::zero());
        }
        self.derivative().eval(t)
    }

    /// Returns the approximate arc length of the curve.
    pub fn arc_length(&self, tolerance: F) -> F {
        let points = self.to_polyline(tolerance);
        let mut length = F::zero();
        for i in 1..points.len() {
            length = length + points[i - 1].distance(points[i]);
        }
        length
    }

    /// Inserts a knot at parameter `t`, returning a new B-spline with one more control point.
    ///
    /// Knot insertion does not change the shape of the curve.
    pub fn insert_knot(&self, t: F) -> Self {
        let (t_min, t_max) = self.domain();
        assert!(
            t >= t_min && t <= t_max,
            "Knot must be within the valid domain"
        );

        let p = self.degree;
        let n = self.control_points.len();

        // Find knot span
        let k = self.find_knot_span(t);

        // New knot vector
        let mut new_knots = Vec::with_capacity(self.knots.len() + 1);
        new_knots.extend_from_slice(&self.knots[..=k]);
        new_knots.push(t);
        new_knots.extend_from_slice(&self.knots[k + 1..]);

        // New control points
        let mut new_points = Vec::with_capacity(n + 1);

        for i in 0..=n {
            if i <= k - p {
                new_points.push(self.control_points[i]);
            } else if i > k {
                new_points.push(self.control_points[i - 1]);
            } else {
                // Compute blended point
                let denom = self.knots[i + p] - self.knots[i];
                let alpha = if denom.abs() < F::epsilon() {
                    F::zero()
                } else {
                    (t - self.knots[i]) / denom
                };

                let one_minus_alpha = F::one() - alpha;
                new_points.push(Point2::new(
                    one_minus_alpha * self.control_points[i - 1].x
                        + alpha * self.control_points[i].x,
                    one_minus_alpha * self.control_points[i - 1].y
                        + alpha * self.control_points[i].y,
                ));
            }
        }

        Self {
            control_points: new_points,
            degree: p,
            knots: new_knots,
        }
    }

    /// Returns the bounding box of the control points.
    ///
    /// Note: This is a conservative bound; the actual curve is contained within.
    pub fn control_bounds(&self) -> (Point2<F>, Point2<F>) {
        let mut min_x = self.control_points[0].x;
        let mut min_y = self.control_points[0].y;
        let mut max_x = min_x;
        let mut max_y = min_y;

        for p in &self.control_points[1..] {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_knots_creation() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points.clone(), 2);

        assert_eq!(spline.degree, 2);
        assert_eq!(spline.control_points.len(), 4);
        assert_eq!(spline.knots.len(), 7); // n + p + 1 = 4 + 2 + 1

        // Clamped knots: [0, 0, 0, 1, 2, 2, 2]
        assert_relative_eq!(spline.knots[0], 0.0);
        assert_relative_eq!(spline.knots[1], 0.0);
        assert_relative_eq!(spline.knots[2], 0.0);
    }

    #[test]
    fn test_eval_endpoints() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 3);

        // With clamped knots, curve passes through first and last control points
        let (t_min, t_max) = spline.domain();
        let start = spline.eval(t_min);
        let end = spline.eval(t_max);

        assert_relative_eq!(start.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(start.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(end.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(end.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_degree_1_is_polyline() {
        // Degree 1 B-spline should be a polyline through control points
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 1);
        let (t_min, t_max) = spline.domain();

        // Should pass through middle point at t=0.5
        let t_mid = (t_min + t_max) / 2.0;
        let mid = spline.eval(t_mid);

        assert_relative_eq!(mid.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mid.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_spline_smooth() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 0.0),
            Point2::new(4.0, 1.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 3);
        let (t_min, t_max) = spline.domain();

        // Evaluate at several points - should be smooth
        let num_samples = 50;
        let mut prev_point = spline.eval(t_min);

        for i in 1..=num_samples {
            let t = t_min + (t_max - t_min) * (i as f64) / (num_samples as f64);
            let point = spline.eval(t);

            // Points should be reasonably close (no large jumps)
            // With 50 samples over ~4 units, each step should be < 0.5 units
            let dist = prev_point.distance(point);
            assert!(dist < 0.5, "Large jump detected at t={}: dist={}", t, dist);

            prev_point = point;
        }
    }

    #[test]
    fn test_to_polyline() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 3);
        let polyline = spline.to_polyline(0.1);

        assert!(polyline.len() >= 2);

        // First and last should match curve endpoints
        let (t_min, t_max) = spline.domain();
        let start = spline.eval(t_min);
        let end = spline.eval(t_max);

        assert_relative_eq!(polyline.first().unwrap().x, start.x, epsilon = 1e-10);
        assert_relative_eq!(polyline.last().unwrap().x, end.x, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 2);
        let deriv = spline.derivative();

        // Derivative of degree-2 is degree-1
        assert_eq!(deriv.degree, 1);
        assert_eq!(deriv.control_points.len(), 3);
    }

    #[test]
    fn test_derivative_at() {
        // Straight line should have constant derivative
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 1);
        let (t_min, t_max) = spline.domain();

        let d1 = spline.derivative_at(t_min + 0.1);
        let d2 = spline.derivative_at((t_min + t_max) / 2.0);

        // For a straight line, derivative should be constant
        assert_relative_eq!(d1.x, d2.x, epsilon = 1e-6);
        assert_relative_eq!(d1.y, d2.y, epsilon = 1e-6);
    }

    #[test]
    fn test_knot_insertion() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 3);
        let (t_min, t_max) = spline.domain();
        let t_insert = (t_min + t_max) / 2.0;

        let new_spline = spline.insert_knot(t_insert);

        // Should have one more control point
        assert_eq!(new_spline.control_points.len(), 5);
        assert_eq!(new_spline.knots.len(), 9);

        // Curve shape should be unchanged
        for i in 0..=10 {
            let t = t_min + (t_max - t_min) * (i as f64) / 10.0;
            let p1 = spline.eval(t);
            let p2 = new_spline.eval(t);

            assert_relative_eq!(p1.x, p2.x, epsilon = 1e-10);
            assert_relative_eq!(p1.y, p2.y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_arc_length() {
        // Straight line
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 1);
        let length = spline.arc_length(0.01);

        assert_relative_eq!(length, 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_control_bounds() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, -1.0),
            Point2::new(3.0, 1.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 2);
        let (min, max) = spline.control_bounds();

        assert_relative_eq!(min.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(min.y, -1.0, epsilon = 1e-10);
        assert_relative_eq!(max.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(max.y, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_domain() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 2);
        let (t_min, t_max) = spline.domain();

        // Clamped uniform knots should have domain [0, 1]
        assert_relative_eq!(t_min, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t_max, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_try_eval() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 2);

        // Valid parameter
        assert!(spline.try_eval(0.5).is_some());

        // Invalid parameters
        assert!(spline.try_eval(-1.0).is_none());
        assert!(spline.try_eval(100.0).is_none());
    }

    #[test]
    fn test_quadratic_bezier_equivalent() {
        // A B-spline with n=p+1 control points and clamped knots is a Bezier curve
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points.clone(), 2);

        // At t=0.5, quadratic Bezier with these points gives (1, 1)
        let mid = spline.eval(0.5);
        assert_relative_eq!(mid.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mid.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline = BSpline2::with_uniform_knots(points, 2);
        let mid = spline.eval(0.5);

        assert!((mid.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_num_bezier_segments() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.0),
            Point2::new(4.0, 0.0),
        ];

        let spline: BSpline2<f64> = BSpline2::with_uniform_knots(points, 3);
        assert_eq!(spline.num_bezier_segments(), 2); // n - p = 5 - 3 = 2
    }
}
