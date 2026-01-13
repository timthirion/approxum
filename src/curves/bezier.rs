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

    /// Returns the tight bounding box of the curve.
    ///
    /// Unlike `control_bounds()`, this computes the exact bounding box by
    /// finding the curve's extrema (where the derivative is zero).
    pub fn tight_bounds(&self) -> (Point2<F>, Point2<F>) {
        let mut min_x = self.p0.x.min(self.p2.x);
        let mut max_x = self.p0.x.max(self.p2.x);
        let mut min_y = self.p0.y.min(self.p2.y);
        let mut max_y = self.p0.y.max(self.p2.y);

        // Derivative of quadratic Bezier: B'(t) = 2(1-t)(p1-p0) + 2t(p2-p1)
        // Setting to zero: t = (p0 - p1) / (p0 - 2*p1 + p2)

        let two = F::one() + F::one();

        // X extremum
        let denom_x = self.p0.x - two * self.p1.x + self.p2.x;
        if denom_x.abs() > F::epsilon() {
            let t_x = (self.p0.x - self.p1.x) / denom_x;
            if t_x > F::zero() && t_x < F::one() {
                let p = self.eval(t_x);
                min_x = min_x.min(p.x);
                max_x = max_x.max(p.x);
            }
        }

        // Y extremum
        let denom_y = self.p0.y - two * self.p1.y + self.p2.y;
        if denom_y.abs() > F::epsilon() {
            let t_y = (self.p0.y - self.p1.y) / denom_y;
            if t_y > F::zero() && t_y < F::one() {
                let p = self.eval(t_y);
                min_y = min_y.min(p.y);
                max_y = max_y.max(p.y);
            }
        }

        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }

    /// Returns the bounding box of the curve's control points.
    ///
    /// Note: This is a conservative bound; the actual curve may be tighter.
    pub fn control_bounds(&self) -> (Point2<F>, Point2<F>) {
        let min_x = self.p0.x.min(self.p1.x).min(self.p2.x);
        let min_y = self.p0.y.min(self.p1.y).min(self.p2.y);
        let max_x = self.p0.x.max(self.p1.x).max(self.p2.x);
        let max_y = self.p0.y.max(self.p1.y).max(self.p2.y);
        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }

    /// Returns the derivative at parameter `t`.
    ///
    /// The derivative represents the velocity along the curve.
    #[inline]
    pub fn derivative_at(&self, t: F) -> Point2<F> {
        let one = F::one();
        let two = one + one;
        let mt = one - t;

        Point2::new(
            two * (mt * (self.p1.x - self.p0.x) + t * (self.p2.x - self.p1.x)),
            two * (mt * (self.p1.y - self.p0.y) + t * (self.p2.y - self.p1.y)),
        )
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

    /// Finds the nearest point on the curve to a given point.
    ///
    /// Returns `(t, point)` where `t` is the parameter and `point` is the
    /// closest point on the curve.
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, curves::QuadraticBezier2};
    ///
    /// let curve = QuadraticBezier2::new(
    ///     Point2::new(0.0, 0.0),
    ///     Point2::new(1.0, 2.0),
    ///     Point2::new(2.0, 0.0),
    /// );
    ///
    /// let (t, nearest) = curve.nearest_point(Point2::new(1.0, 0.0));
    /// assert!(t >= 0.0 && t <= 1.0);
    /// ```
    pub fn nearest_point(&self, point: Point2<F>) -> (F, Point2<F>) {
        // Use sampling followed by Newton-Raphson refinement
        let samples = 16usize;
        let mut best_t = F::zero();
        let mut best_dist_sq = self.p0.distance_squared(point);

        // Sample the curve to find initial guess
        for i in 0..=samples {
            let t = F::from(i).unwrap() / F::from(samples).unwrap();
            let p = self.eval(t);
            let dist_sq = p.distance_squared(point);
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_t = t;
            }
        }

        // Refine with Newton-Raphson
        best_t = self.refine_nearest(point, best_t);

        (best_t, self.eval(best_t))
    }

    fn refine_nearest(&self, point: Point2<F>, mut t: F) -> F {
        let iterations = 5;
        let zero = F::zero();
        let one = F::one();

        for _ in 0..iterations {
            let p = self.eval(t);
            let d = self.derivative_at(t);

            // f(t) = (C(t) - point) · C'(t)
            let diff = Point2::new(p.x - point.x, p.y - point.y);
            let f = diff.x * d.x + diff.y * d.y;

            // f'(t) = C'(t) · C'(t) + (C(t) - point) · C''(t)
            // For quadratic, C''(t) is constant: 2*(p0 - 2*p1 + p2)
            let two = one + one;
            let d2x = two * (self.p0.x - two * self.p1.x + self.p2.x);
            let d2y = two * (self.p0.y - two * self.p1.y + self.p2.y);
            let f_prime = d.x * d.x + d.y * d.y + diff.x * d2x + diff.y * d2y;

            if f_prime.abs() < F::epsilon() {
                break;
            }

            t = t - f / f_prime;
            t = t.max(zero).min(one);
        }

        t
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

    /// Returns the derivative at parameter `t`.
    #[inline]
    pub fn derivative_at(&self, t: F) -> Point2<F> {
        self.derivative().eval(t)
    }

    /// Returns the tight bounding box of the curve.
    ///
    /// Unlike `control_bounds()`, this computes the exact bounding box by
    /// finding the curve's extrema (where the derivative is zero).
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, curves::CubicBezier2};
    ///
    /// // S-curve that extends beyond control points
    /// let curve = CubicBezier2::new(
    ///     Point2::new(0.0, 0.0),
    ///     Point2::new(0.0, 2.0),
    ///     Point2::new(2.0, -1.0),
    ///     Point2::new(2.0, 1.0),
    /// );
    ///
    /// let (min, max) = curve.tight_bounds();
    /// // Tight bounds will be smaller than control bounds
    /// let (ctrl_min, ctrl_max) = curve.control_bounds();
    /// assert!(max.y <= ctrl_max.y);
    /// ```
    pub fn tight_bounds(&self) -> (Point2<F>, Point2<F>) {
        let mut min_x = self.p0.x.min(self.p3.x);
        let mut max_x = self.p0.x.max(self.p3.x);
        let mut min_y = self.p0.y.min(self.p3.y);
        let mut max_y = self.p0.y.max(self.p3.y);

        // Derivative of cubic: B'(t) = 3[(1-t)²(p1-p0) + 2(1-t)t(p2-p1) + t²(p3-p2)]
        // This is a quadratic, so we solve at² + bt + c = 0 for each axis

        // For X: compute coefficients of the derivative
        let one = F::one();
        let two = one + one;
        let three = two + one;

        // d/dt [B(t)] for x: quadratic with coefficients:
        // a = 3(-p0 + 3p1 - 3p2 + p3)
        // b = 6(p0 - 2p1 + p2)
        // c = 3(p1 - p0)
        let ax = three * (-self.p0.x + three * self.p1.x - three * self.p2.x + self.p3.x);
        let bx = (three + three) * (self.p0.x - two * self.p1.x + self.p2.x);
        let cx = three * (self.p1.x - self.p0.x);

        for t in solve_quadratic(ax, bx, cx) {
            if t > F::zero() && t < one {
                let p = self.eval(t);
                min_x = min_x.min(p.x);
                max_x = max_x.max(p.x);
            }
        }

        let ay = three * (-self.p0.y + three * self.p1.y - three * self.p2.y + self.p3.y);
        let by = (three + three) * (self.p0.y - two * self.p1.y + self.p2.y);
        let cy = three * (self.p1.y - self.p0.y);

        for t in solve_quadratic(ay, by, cy) {
            if t > F::zero() && t < one {
                let p = self.eval(t);
                min_y = min_y.min(p.y);
                max_y = max_y.max(p.y);
            }
        }

        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }

    /// Finds the nearest point on the curve to a given point.
    ///
    /// Returns `(t, point)` where `t` is the parameter and `point` is the
    /// closest point on the curve.
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, curves::CubicBezier2};
    ///
    /// let curve: CubicBezier2<f64> = CubicBezier2::new(
    ///     Point2::new(0.0, 0.0),
    ///     Point2::new(1.0, 2.0),
    ///     Point2::new(3.0, 2.0),
    ///     Point2::new(4.0, 0.0),
    /// );
    ///
    /// // Point below the curve
    /// let (t, nearest) = curve.nearest_point(Point2::new(2.0, -1.0));
    /// assert!(t >= 0.0 && t <= 1.0);
    /// // Nearest point should be on the curve
    /// let on_curve = curve.eval(t);
    /// assert!((nearest.x - on_curve.x).abs() < 1e-10);
    /// ```
    pub fn nearest_point(&self, point: Point2<F>) -> (F, Point2<F>) {
        // Use sampling followed by Newton-Raphson refinement
        let samples = 16usize;
        let mut best_t = F::zero();
        let mut best_dist_sq = self.p0.distance_squared(point);

        // Sample the curve to find initial guess
        for i in 0..=samples {
            let t = F::from(i).unwrap() / F::from(samples).unwrap();
            let p = self.eval(t);
            let dist_sq = p.distance_squared(point);
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_t = t;
            }
        }

        // Refine with Newton-Raphson
        best_t = self.refine_nearest(point, best_t);

        (best_t, self.eval(best_t))
    }

    fn refine_nearest(&self, point: Point2<F>, mut t: F) -> F {
        let iterations = 5;
        let zero = F::zero();
        let one = F::one();

        let deriv = self.derivative();

        for _ in 0..iterations {
            let p = self.eval(t);
            let d = deriv.eval(t);

            // f(t) = (C(t) - point) · C'(t)
            let diff = Point2::new(p.x - point.x, p.y - point.y);
            let f = diff.x * d.x + diff.y * d.y;

            // f'(t) = C'(t) · C'(t) + (C(t) - point) · C''(t)
            let d2 = deriv.derivative_at(t);
            let f_prime = d.x * d.x + d.y * d.y + diff.x * d2.x + diff.y * d2.y;

            if f_prime.abs() < F::epsilon() {
                break;
            }

            t = t - f / f_prime;
            t = t.max(zero).min(one);
        }

        t
    }
}

/// Linear interpolation between two points.
#[inline]
fn lerp_point<F: Float>(a: Point2<F>, b: Point2<F>, t: F) -> Point2<F> {
    a.lerp(b, t)
}

/// Solves the quadratic equation ax² + bx + c = 0.
/// Returns 0, 1, or 2 real roots.
fn solve_quadratic<F: Float>(a: F, b: F, c: F) -> Vec<F> {
    let eps = F::epsilon() * F::from(1000.0).unwrap();

    if a.abs() < eps {
        // Linear equation: bx + c = 0
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

    // Tight bounds tests

    #[test]
    fn test_quadratic_tight_bounds_simple() {
        // Simple curve where tight bounds equal control bounds
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        );

        let (min, max) = curve.tight_bounds();
        assert_relative_eq!(min.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max.x, 2.0, epsilon = 1e-10);
        // The curve peaks at t=0.5 with y=0.5 (midpoint between p0.y, p1.y, p2.y weighted)
        assert!(min.y >= -0.01);
        assert!(max.y <= 1.01);
    }

    #[test]
    fn test_quadratic_tight_bounds_with_extrema() {
        // Curve where control point extends beyond curve
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        let (tight_min, tight_max) = curve.tight_bounds();
        let (ctrl_min, ctrl_max) = curve.control_bounds();

        // X bounds should be same (endpoints are extrema)
        assert_relative_eq!(tight_min.x, ctrl_min.x, epsilon = 1e-10);
        assert_relative_eq!(tight_max.x, ctrl_max.x, epsilon = 1e-10);

        // Tight bounds should not exceed control bounds
        assert!(tight_min.y >= ctrl_min.y - 1e-10);
        assert!(tight_max.y <= ctrl_max.y + 1e-10);

        // The curve reaches max y=1.0 at t=0.5, not y=2.0 (control point)
        assert_relative_eq!(tight_max.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_tight_bounds_simple() {
        // Straight line
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        );

        let (min, max) = curve.tight_bounds();
        assert_relative_eq!(min.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(min.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_tight_bounds_with_extrema() {
        // S-curve that extends beyond endpoints in Y
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 2.0),
            Point2::new(2.0, -1.0),
            Point2::new(2.0, 1.0),
        );

        let (tight_min, tight_max) = curve.tight_bounds();
        let (ctrl_min, ctrl_max) = curve.control_bounds();

        // Tight Y bounds should be within control bounds
        assert!(tight_min.y >= ctrl_min.y - 1e-10);
        assert!(tight_max.y <= ctrl_max.y + 1e-10);

        // X bounds should match control bounds (endpoints are x extrema)
        assert_relative_eq!(tight_min.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(tight_max.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_tight_bounds_symmetric() {
        // Symmetric arch curve
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 3.0),
            Point2::new(3.0, 3.0),
            Point2::new(4.0, 0.0),
        );

        let (min, max) = curve.tight_bounds();

        // X bounds are endpoints
        assert_relative_eq!(min.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max.x, 4.0, epsilon = 1e-10);

        // Y min is 0, max should be less than control point y=3
        assert_relative_eq!(min.y, 0.0, epsilon = 1e-10);
        assert!(max.y < 3.0);
        assert!(max.y > 1.5); // But curve does rise significantly
    }

    // Nearest point tests

    #[test]
    fn test_quadratic_nearest_point_on_curve() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        // Test a point on the curve
        let point_on_curve = curve.eval(0.5);
        let (t, nearest) = curve.nearest_point(point_on_curve);

        assert_relative_eq!(t, 0.5, epsilon = 1e-6);
        assert_relative_eq!(nearest.x, point_on_curve.x, epsilon = 1e-6);
        assert_relative_eq!(nearest.y, point_on_curve.y, epsilon = 1e-6);
    }

    #[test]
    fn test_quadratic_nearest_point_off_curve() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        // Point above the curve's peak - closer to curve interior than endpoints
        let query = Point2::new(1.0, 2.0);
        let (t, nearest) = curve.nearest_point(query);

        // Should find a point near t=0.5 (curve peak)
        assert!(t > 0.3 && t < 0.7);

        // Nearest point should be closer to query than endpoints
        let dist_nearest = nearest.distance(query);
        let dist_start = curve.p0.distance(query);
        let dist_end = curve.p2.distance(query);
        assert!(dist_nearest < dist_start);
        assert!(dist_nearest < dist_end);
    }

    #[test]
    fn test_quadratic_nearest_point_at_endpoint() {
        let curve: QuadraticBezier2<f64> = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        );

        // Point closest to start
        let (t, _nearest) = curve.nearest_point(Point2::new(-1.0, 0.0));
        assert!(t < 0.1);

        // Point closest to end
        let (t, _nearest) = curve.nearest_point(Point2::new(3.0, 0.0));
        assert!(t > 0.9);
    }

    #[test]
    fn test_cubic_nearest_point_on_curve() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        // Test a point on the curve
        let point_on_curve = curve.eval(0.3);
        let (t, nearest) = curve.nearest_point(point_on_curve);

        assert_relative_eq!(t, 0.3, epsilon = 1e-5);
        assert_relative_eq!(nearest.x, point_on_curve.x, epsilon = 1e-5);
        assert_relative_eq!(nearest.y, point_on_curve.y, epsilon = 1e-5);
    }

    #[test]
    fn test_cubic_nearest_point_off_curve() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        // Point above the curve
        let (t, nearest) = curve.nearest_point(Point2::new(2.0, 3.0));

        // Should be roughly in the middle
        assert!(t > 0.3 && t < 0.7);

        // The nearest point should be on the curve
        let expected = curve.eval(t);
        assert_relative_eq!(nearest.x, expected.x, epsilon = 1e-10);
        assert_relative_eq!(nearest.y, expected.y, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_nearest_point_at_endpoints() {
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );

        // Point closest to start
        let (t, _) = curve.nearest_point(Point2::new(-2.0, 0.0));
        assert!(t < 0.1);

        // Point closest to end
        let (t, _) = curve.nearest_point(Point2::new(6.0, 0.0));
        assert!(t > 0.9);
    }

    #[test]
    fn test_cubic_nearest_point_perpendicular() {
        // Straight horizontal line
        let curve: CubicBezier2<f64> = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        );

        // Point directly above middle of line
        let (t, nearest) = curve.nearest_point(Point2::new(1.5, 5.0));

        assert_relative_eq!(t, 0.5, epsilon = 1e-5);
        assert_relative_eq!(nearest.x, 1.5, epsilon = 1e-5);
        assert_relative_eq!(nearest.y, 0.0, epsilon = 1e-5);
    }
}
