//! Bézier curve types and discretization.
//!
//! Provides quadratic and cubic Bézier curves with adaptive subdivision
//! for converting curves to polylines.

use crate::primitives::{Point2, Segment2};
use num_traits::Float;

/// A quadratic Bézier curve defined by 3 control points.
///
/// The curve starts at `p0`, is influenced by `p1`, and ends at `p2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraticBezier2<F> {
    /// Start point
    pub p0: Point2<F>,
    /// Control point
    pub p1: Point2<F>,
    /// End point
    pub p2: Point2<F>,
}

impl<F: Float> QuadraticBezier2<F> {
    /// Creates a new quadratic Bézier curve.
    #[inline]
    pub fn new(p0: Point2<F>, p1: Point2<F>, p2: Point2<F>) -> Self {
        Self { p0, p1, p2 }
    }

    /// Evaluates the curve at parameter `t` (0 to 1).
    #[inline]
    pub fn eval(&self, t: F) -> Point2<F> {
        let one = F::one();
        let mt = one - t;
        let mt2 = mt * mt;
        let t2 = t * t;
        let two = one + one;

        Point2::new(
            mt2 * self.p0.x + two * mt * t * self.p1.x + t2 * self.p2.x,
            mt2 * self.p0.y + two * mt * t * self.p1.y + t2 * self.p2.y,
        )
    }

    /// Splits the curve at parameter `t`, returning two new curves.
    pub fn split(&self, t: F) -> (Self, Self) {
        let one = F::one();
        let mt = one - t;

        // de Casteljau's algorithm
        let p01 = Point2::new(
            mt * self.p0.x + t * self.p1.x,
            mt * self.p0.y + t * self.p1.y,
        );
        let p12 = Point2::new(
            mt * self.p1.x + t * self.p2.x,
            mt * self.p1.y + t * self.p2.y,
        );
        let p012 = Point2::new(mt * p01.x + t * p12.x, mt * p01.y + t * p12.y);

        (
            Self::new(self.p0, p01, p012),
            Self::new(p012, p12, self.p2),
        )
    }

    /// Returns the maximum distance from the control point to the baseline.
    ///
    /// This is used as a flatness measure for adaptive subdivision.
    #[inline]
    pub fn flatness(&self) -> F {
        let seg = Segment2::new(self.p0, self.p2);
        seg.distance_to_point(self.p1)
    }

    /// Converts the curve to a polyline using adaptive subdivision.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum allowed deviation from the true curve
    ///
    /// # Returns
    ///
    /// A vector of points approximating the curve. Always includes the
    /// start point; the end point is included as the last point.
    pub fn to_polyline(&self, tolerance: F) -> Vec<Point2<F>> {
        let mut points = Vec::new();
        points.push(self.p0);
        self.subdivide_recursive(tolerance, &mut points);
        points
    }

    fn subdivide_recursive(&self, tolerance: F, points: &mut Vec<Point2<F>>) {
        if self.flatness() <= tolerance {
            points.push(self.p2);
        } else {
            let half = F::from(0.5).unwrap();
            let (left, right) = self.split(half);
            left.subdivide_recursive(tolerance, points);
            right.subdivide_recursive(tolerance, points);
        }
    }
}

/// A cubic Bézier curve defined by 4 control points.
///
/// The curve starts at `p0`, is influenced by `p1` and `p2`, and ends at `p3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicBezier2<F> {
    /// Start point
    pub p0: Point2<F>,
    /// First control point
    pub p1: Point2<F>,
    /// Second control point
    pub p2: Point2<F>,
    /// End point
    pub p3: Point2<F>,
}

impl<F: Float> CubicBezier2<F> {
    /// Creates a new cubic Bézier curve.
    #[inline]
    pub fn new(p0: Point2<F>, p1: Point2<F>, p2: Point2<F>, p3: Point2<F>) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Evaluates the curve at parameter `t` (0 to 1).
    #[inline]
    pub fn eval(&self, t: F) -> Point2<F> {
        let one = F::one();
        let mt = one - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        let t2 = t * t;
        let t3 = t2 * t;
        let three = one + one + one;

        Point2::new(
            mt3 * self.p0.x
                + three * mt2 * t * self.p1.x
                + three * mt * t2 * self.p2.x
                + t3 * self.p3.x,
            mt3 * self.p0.y
                + three * mt2 * t * self.p1.y
                + three * mt * t2 * self.p2.y
                + t3 * self.p3.y,
        )
    }

    /// Splits the curve at parameter `t`, returning two new curves.
    pub fn split(&self, t: F) -> (Self, Self) {
        // de Casteljau's algorithm
        let p01 = lerp_point(self.p0, self.p1, t);
        let p12 = lerp_point(self.p1, self.p2, t);
        let p23 = lerp_point(self.p2, self.p3, t);

        let p012 = lerp_point(p01, p12, t);
        let p123 = lerp_point(p12, p23, t);

        let p0123 = lerp_point(p012, p123, t);

        (
            Self::new(self.p0, p01, p012, p0123),
            Self::new(p0123, p123, p23, self.p3),
        )
    }

    /// Returns the maximum distance from control points to the baseline.
    ///
    /// This is used as a flatness measure for adaptive subdivision.
    /// Uses the maximum of distances from p1 and p2 to the line p0-p3.
    #[inline]
    pub fn flatness(&self) -> F {
        let seg = Segment2::new(self.p0, self.p3);
        let d1 = seg.distance_to_point(self.p1);
        let d2 = seg.distance_to_point(self.p2);
        d1.max(d2)
    }

    /// Converts the curve to a polyline using adaptive subdivision.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum allowed deviation from the true curve
    ///
    /// # Returns
    ///
    /// A vector of points approximating the curve. Always includes the
    /// start point; the end point is included as the last point.
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, curves::CubicBezier2};
    ///
    /// let curve = CubicBezier2::new(
    ///     Point2::new(0.0, 0.0),
    ///     Point2::new(1.0, 2.0),
    ///     Point2::new(3.0, 2.0),
    ///     Point2::new(4.0, 0.0),
    /// );
    ///
    /// let polyline = curve.to_polyline(0.1);
    /// assert!(polyline.len() >= 2);
    /// assert_eq!(polyline.first().unwrap().x, 0.0);
    /// assert_eq!(polyline.last().unwrap().x, 4.0);
    /// ```
    pub fn to_polyline(&self, tolerance: F) -> Vec<Point2<F>> {
        let mut points = Vec::new();
        points.push(self.p0);
        self.subdivide_recursive(tolerance, &mut points);
        points
    }

    fn subdivide_recursive(&self, tolerance: F, points: &mut Vec<Point2<F>>) {
        if self.flatness() <= tolerance {
            points.push(self.p3);
        } else {
            let half = F::from(0.5).unwrap();
            let (left, right) = self.split(half);
            left.subdivide_recursive(tolerance, points);
            right.subdivide_recursive(tolerance, points);
        }
    }

    /// Returns the approximate arc length of the curve.
    ///
    /// Uses adaptive subdivision to compute a reasonably accurate length.
    pub fn arc_length(&self, tolerance: F) -> F {
        let points = self.to_polyline(tolerance);
        let mut length = F::zero();
        for i in 1..points.len() {
            length = length + points[i - 1].distance(points[i]);
        }
        length
    }

    /// Returns the bounding box of the curve's control points.
    ///
    /// Note: This is a conservative bound; the actual curve may be tighter.
    pub fn control_bounds(&self) -> (Point2<F>, Point2<F>) {
        let min_x = self.p0.x.min(self.p1.x).min(self.p2.x).min(self.p3.x);
        let min_y = self.p0.y.min(self.p1.y).min(self.p2.y).min(self.p3.y);
        let max_x = self.p0.x.max(self.p1.x).max(self.p2.x).max(self.p3.x);
        let max_y = self.p0.y.max(self.p1.y).max(self.p2.y).max(self.p3.y);
        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }

    /// Returns the derivative curve (a quadratic Bézier).
    ///
    /// The derivative represents the velocity along the curve.
    pub fn derivative(&self) -> QuadraticBezier2<F> {
        let three = F::one() + F::one() + F::one();
        QuadraticBezier2::new(
            Point2::new(
                three * (self.p1.x - self.p0.x),
                three * (self.p1.y - self.p0.y),
            ),
            Point2::new(
                three * (self.p2.x - self.p1.x),
                three * (self.p2.y - self.p1.y),
            ),
            Point2::new(
                three * (self.p3.x - self.p2.x),
                three * (self.p3.y - self.p2.y),
            ),
        )
    }
}

/// Linear interpolation between two points.
#[inline]
fn lerp_point<F: Float>(a: Point2<F>, b: Point2<F>, t: F) -> Point2<F> {
    a.lerp(b, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // QuadraticBezier2 tests

    #[test]
    fn test_quadratic_eval_endpoints() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        let start = curve.eval(0.0);
        assert_relative_eq!(start.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(start.y, 0.0, epsilon = 1e-10);

        let end = curve.eval(1.0);
        assert_relative_eq!(end.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(end.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_eval_midpoint() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        let mid = curve.eval(0.5);
        assert_relative_eq!(mid.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mid.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_split() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        let (left, right) = curve.split(0.5);

        // Left curve should start at p0 and end at midpoint
        assert_relative_eq!(left.p0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(left.p2.x, 1.0, epsilon = 1e-10);

        // Right curve should start at midpoint and end at p2
        assert_relative_eq!(right.p0.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(right.p2.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_to_polyline() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        let polyline = curve.to_polyline(0.1);

        // Should have at least start and end
        assert!(polyline.len() >= 2);
        assert_relative_eq!(polyline.first().unwrap().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(polyline.last().unwrap().x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_straight_line() {
        // Quadratic with collinear control points is a straight line
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        );

        let polyline = curve.to_polyline(0.01);

        // Should be just 2 points (start and end)
        assert_eq!(polyline.len(), 2);
    }

    // CubicBezier2 tests

    #[test]
    fn test_cubic_eval_endpoints() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let start = curve.eval(0.0);
        assert_relative_eq!(start.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(start.y, 0.0, epsilon = 1e-10);

        let end = curve.eval(1.0);
        assert_relative_eq!(end.x, 4.0, epsilon = 1e-10);
        assert_relative_eq!(end.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_eval_midpoint() {
        // Symmetric curve
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 2.0),
            Point2::new(4.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let mid = curve.eval(0.5);
        assert_relative_eq!(mid.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(mid.y, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_split() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let mid_point = curve.eval(0.5);
        let (left, right) = curve.split(0.5);

        // Left curve should end at midpoint
        assert_relative_eq!(left.p3.x, mid_point.x, epsilon = 1e-10);
        assert_relative_eq!(left.p3.y, mid_point.y, epsilon = 1e-10);

        // Right curve should start at midpoint
        assert_relative_eq!(right.p0.x, mid_point.x, epsilon = 1e-10);
        assert_relative_eq!(right.p0.y, mid_point.y, epsilon = 1e-10);

        // Evaluating split curves should match original
        let t = 0.25;
        let orig = curve.eval(t);
        let from_left = left.eval(t * 2.0); // t=0.25 on original = t=0.5 on left half
        assert_relative_eq!(orig.x, from_left.x, epsilon = 1e-10);
        assert_relative_eq!(orig.y, from_left.y, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_to_polyline() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let polyline = curve.to_polyline(0.1);

        assert!(polyline.len() >= 2);
        assert_relative_eq!(polyline.first().unwrap().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(polyline.last().unwrap().x, 4.0, epsilon = 1e-10);

        // All points should be close to the curve
        for p in &polyline {
            // This is a rough check - point should be in bounding box
            let (min, max) = curve.control_bounds();
            assert!(p.x >= min.x - 0.1 && p.x <= max.x + 0.1);
            assert!(p.y >= min.y - 0.1 && p.y <= max.y + 0.1);
        }
    }

    #[test]
    fn test_cubic_straight_line() {
        // Cubic with collinear control points
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        );

        let polyline = curve.to_polyline(0.01);
        assert_eq!(polyline.len(), 2);
    }

    #[test]
    fn test_cubic_tolerance_affects_subdivision() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 10.0),
            Point2::new(3.0, 10.0),
            Point2::new(4.0, 0.0),
        );

        let coarse = curve.to_polyline(1.0);
        let fine = curve.to_polyline(0.01);

        // Finer tolerance should produce more points
        assert!(fine.len() > coarse.len());
    }

    #[test]
    fn test_cubic_arc_length() {
        // Straight line should have arc length = distance between endpoints
        let line: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        );

        let length = line.arc_length(0.001);
        assert_relative_eq!(length, 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_cubic_derivative() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(3.0, 0.0),
        );

        let deriv = curve.derivative();

        // Derivative at t=0 should be 3*(p1-p0)
        let d0 = deriv.eval(0.0);
        assert_relative_eq!(d0.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(d0.y, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_f32() {
        let curve: CubicBezier2<f32> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        let polyline = curve.to_polyline(0.1);
        assert!(polyline.len() >= 2);
    }
}
