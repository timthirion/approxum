//! Cubic Hermite splines.
//!
//! Hermite splines are defined by points and their tangent vectors, giving
//! explicit control over the curve direction at each point.
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, Vec2, curves::HermiteSpline2};
//!
//! let points = vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(2.0, 0.0),
//! ];
//! let tangents = vec![
//!     Vec2::new(1.0, 2.0),  // Start tangent points up-right
//!     Vec2::new(1.0, -2.0), // End tangent points down-right
//! ];
//!
//! let spline = HermiteSpline2::new(points, tangents);
//! let mid = spline.eval(0.5); // S-curve shape
//! ```

use crate::primitives::{Point2, Vec2};
use num_traits::Float;

/// A cubic Hermite spline in 2D.
///
/// Hermite splines are defined by points and their tangent vectors.
/// This gives explicit control over the curve direction at each point,
/// unlike Catmull-Rom which computes tangents automatically.
#[derive(Debug, Clone, PartialEq)]
pub struct HermiteSpline2<F> {
    /// Points the spline passes through
    pub points: Vec<Point2<F>>,
    /// Tangent vectors at each point
    pub tangents: Vec<Vec2<F>>,
}

impl<F: Float> HermiteSpline2<F> {
    /// Creates a new Hermite spline with the given points and tangents.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Fewer than 2 points
    /// - Number of tangents doesn't match number of points
    pub fn new(points: Vec<Point2<F>>, tangents: Vec<Vec2<F>>) -> Self {
        assert!(points.len() >= 2, "Need at least 2 points");
        assert!(
            tangents.len() == points.len(),
            "Number of tangents must equal number of points"
        );
        Self { points, tangents }
    }

    /// Creates a Hermite spline with automatically computed tangents.
    ///
    /// Tangents are computed using finite differences (similar to Catmull-Rom).
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, curves::HermiteSpline2};
    ///
    /// let points = vec![
    ///     Point2::new(0.0_f64, 0.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(2.0, 0.0),
    /// ];
    ///
    /// let spline = HermiteSpline2::with_auto_tangents(points);
    /// // Tangents are computed automatically from neighboring points
    /// ```
    pub fn with_auto_tangents(points: Vec<Point2<F>>) -> Self {
        assert!(points.len() >= 2, "Need at least 2 points");

        let n = points.len();
        let mut tangents = Vec::with_capacity(n);
        let half = F::from(0.5).unwrap();

        for i in 0..n {
            let tangent = if i == 0 {
                // Forward difference
                Vec2::new(points[1].x - points[0].x, points[1].y - points[0].y)
            } else if i == n - 1 {
                // Backward difference
                Vec2::new(
                    points[n - 1].x - points[n - 2].x,
                    points[n - 1].y - points[n - 2].y,
                )
            } else {
                // Central difference
                Vec2::new(
                    half * (points[i + 1].x - points[i - 1].x),
                    half * (points[i + 1].y - points[i - 1].y),
                )
            };
            tangents.push(tangent);
        }

        Self { points, tangents }
    }

    /// Creates a Hermite spline with zero tangents (cardinal spline with tension = 1).
    ///
    /// The curve will have sharp corners at control points.
    pub fn with_zero_tangents(points: Vec<Point2<F>>) -> Self {
        assert!(points.len() >= 2, "Need at least 2 points");
        let tangents = vec![Vec2::new(F::zero(), F::zero()); points.len()];
        Self { points, tangents }
    }

    /// Creates a cardinal spline with the given tension.
    ///
    /// Cardinal splines are Hermite splines where tangents are computed as:
    /// `m_i = (1 - tension) * (p_{i+1} - p_{i-1}) / 2`
    ///
    /// - tension = 0: Catmull-Rom spline
    /// - tension = 1: Zero tangents (sharp corners)
    pub fn cardinal(points: Vec<Point2<F>>, tension: F) -> Self {
        assert!(points.len() >= 2, "Need at least 2 points");

        let n = points.len();
        let mut tangents = Vec::with_capacity(n);
        let half = F::from(0.5).unwrap();
        let scale = (F::one() - tension) * half;

        for i in 0..n {
            let tangent = if i == 0 {
                let dx = points[1].x - points[0].x;
                let dy = points[1].y - points[0].y;
                Vec2::new(scale * F::from(2.0).unwrap() * dx, scale * F::from(2.0).unwrap() * dy)
            } else if i == n - 1 {
                let dx = points[n - 1].x - points[n - 2].x;
                let dy = points[n - 1].y - points[n - 2].y;
                Vec2::new(scale * F::from(2.0).unwrap() * dx, scale * F::from(2.0).unwrap() * dy)
            } else {
                Vec2::new(
                    scale * (points[i + 1].x - points[i - 1].x),
                    scale * (points[i + 1].y - points[i - 1].y),
                )
            };
            tangents.push(tangent);
        }

        Self { points, tangents }
    }

    /// Returns the number of spline segments.
    pub fn num_segments(&self) -> usize {
        if self.points.len() < 2 {
            0
        } else {
            self.points.len() - 1
        }
    }

    /// Returns the valid parameter range.
    pub fn domain(&self) -> (F, F) {
        (F::zero(), F::from(self.num_segments()).unwrap())
    }

    /// Evaluates the spline at parameter `t`.
    ///
    /// Parameter `t` ranges from 0 to `num_segments()`. Integer values
    /// correspond to control points.
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, Vec2, curves::HermiteSpline2};
    ///
    /// let points = vec![Point2::new(0.0_f64, 0.0), Point2::new(1.0, 0.0)];
    /// let tangents = vec![Vec2::new(1.0, 1.0), Vec2::new(1.0, -1.0)];
    ///
    /// let spline = HermiteSpline2::new(points, tangents);
    ///
    /// // At t=0, returns first point
    /// let start = spline.eval(0.0);
    /// assert!((start.x - 0.0).abs() < 1e-10);
    ///
    /// // At t=1, returns second point
    /// let end = spline.eval(1.0);
    /// assert!((end.x - 1.0).abs() < 1e-10);
    /// ```
    pub fn eval(&self, t: F) -> Point2<F> {
        let n = self.points.len();
        if n < 2 {
            return self.points[0];
        }

        let (t_min, t_max) = self.domain();
        let t = t.max(t_min).min(t_max);

        // Find segment
        let segment = t.floor().to_usize().unwrap().min(n - 2);
        let local_t = t - F::from(segment).unwrap();

        let p0 = self.points[segment];
        let p1 = self.points[segment + 1];
        let m0 = self.tangents[segment];
        let m1 = self.tangents[segment + 1];

        self.eval_segment(p0, m0, p1, m1, local_t)
    }

    /// Evaluates a single cubic Hermite segment.
    fn eval_segment(
        &self,
        p0: Point2<F>,
        m0: Vec2<F>,
        p1: Point2<F>,
        m1: Vec2<F>,
        t: F,
    ) -> Point2<F> {
        let t2 = t * t;
        let t3 = t2 * t;

        let one = F::one();
        let two = one + one;
        let three = two + one;

        // Hermite basis functions
        let h00 = two * t3 - three * t2 + one;  // 2t³ - 3t² + 1
        let h10 = t3 - two * t2 + t;            // t³ - 2t² + t
        let h01 = -two * t3 + three * t2;       // -2t³ + 3t²
        let h11 = t3 - t2;                      // t³ - t²

        Point2::new(
            h00 * p0.x + h10 * m0.x + h01 * p1.x + h11 * m1.x,
            h00 * p0.y + h10 * m0.y + h01 * p1.y + h11 * m1.y,
        )
    }

    /// Evaluates the tangent (derivative) at parameter `t`.
    pub fn tangent_at(&self, t: F) -> Vec2<F> {
        let n = self.points.len();
        if n < 2 {
            return Vec2::new(F::zero(), F::zero());
        }

        let (t_min, t_max) = self.domain();
        let t = t.max(t_min).min(t_max);

        let segment = t.floor().to_usize().unwrap().min(n - 2);
        let local_t = t - F::from(segment).unwrap();

        let p0 = self.points[segment];
        let p1 = self.points[segment + 1];
        let m0 = self.tangents[segment];
        let m1 = self.tangents[segment + 1];

        self.derivative_segment(p0, m0, p1, m1, local_t)
    }

    /// Evaluates derivative of a single Hermite segment.
    fn derivative_segment(
        &self,
        p0: Point2<F>,
        m0: Vec2<F>,
        p1: Point2<F>,
        m1: Vec2<F>,
        t: F,
    ) -> Vec2<F> {
        let t2 = t * t;

        let one = F::one();
        let two = one + one;
        let three = two + one;
        let four = two + two;
        let six = three + three;

        // Derivatives of Hermite basis functions
        let dh00 = six * t2 - six * t;           // 6t² - 6t
        let dh10 = three * t2 - four * t + one;  // 3t² - 4t + 1
        let dh01 = -six * t2 + six * t;          // -6t² + 6t
        let dh11 = three * t2 - two * t;         // 3t² - 2t

        Vec2::new(
            dh00 * p0.x + dh10 * m0.x + dh01 * p1.x + dh11 * m1.x,
            dh00 * p0.y + dh10 * m0.y + dh01 * p1.y + dh11 * m1.y,
        )
    }

    /// Converts the spline to a polyline using adaptive sampling.
    pub fn to_polyline(&self, tolerance: F) -> Vec<Point2<F>> {
        if self.points.len() < 2 {
            return self.points.clone();
        }

        let (t_min, t_max) = self.domain();
        let mut result = Vec::new();
        result.push(self.eval(t_min));

        self.subdivide_recursive(t_min, t_max, tolerance, &mut result);
        result
    }

    fn subdivide_recursive(&self, t0: F, t1: F, tolerance: F, points: &mut Vec<Point2<F>>) {
        let two = F::one() + F::one();
        let t_mid = (t0 + t1) / two;

        let p0 = self.eval(t0);
        let p1 = self.eval(t1);
        let p_mid = self.eval(t_mid);

        let line_mid = Point2::new((p0.x + p1.x) / two, (p0.y + p1.y) / two);
        let deviation = p_mid.distance(line_mid);

        if deviation <= tolerance {
            points.push(p1);
        } else {
            self.subdivide_recursive(t0, t_mid, tolerance, points);
            self.subdivide_recursive(t_mid, t1, tolerance, points);
        }
    }

    /// Returns the approximate arc length of the spline.
    pub fn arc_length(&self, tolerance: F) -> F {
        let points = self.to_polyline(tolerance);
        let mut length = F::zero();
        for i in 1..points.len() {
            length = length + points[i - 1].distance(points[i]);
        }
        length
    }

    /// Returns the bounding box of the control points.
    ///
    /// Note: The actual curve may extend beyond this box due to tangents.
    pub fn control_bounds(&self) -> (Point2<F>, Point2<F>) {
        let mut min_x = self.points[0].x;
        let mut min_y = self.points[0].y;
        let mut max_x = min_x;
        let mut max_y = min_y;

        for p in &self.points[1..] {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        (Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }

    /// Scales all tangent vectors by a factor.
    ///
    /// Larger tangents create more pronounced curves between points.
    pub fn scale_tangents(&mut self, factor: F) {
        for t in &mut self.tangents {
            t.x = t.x * factor;
            t.y = t.y * factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_creation() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let tangents = vec![Vec2::new(1.0, 1.0), Vec2::new(1.0, -1.0)];

        let spline: HermiteSpline2<f64> = HermiteSpline2::new(points, tangents);
        assert_eq!(spline.points.len(), 2);
        assert_eq!(spline.tangents.len(), 2);
        assert_eq!(spline.num_segments(), 1);
    }

    #[test]
    fn test_interpolation() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];
        let tangents = vec![
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, -1.0),
        ];

        let spline: HermiteSpline2<f64> = HermiteSpline2::new(points.clone(), tangents);

        // Should pass through control points
        for (i, p) in points.iter().enumerate() {
            let t = i as f64;
            let eval_p = spline.eval(t);
            assert_relative_eq!(eval_p.x, p.x, epsilon = 1e-10);
            assert_relative_eq!(eval_p.y, p.y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_auto_tangents() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: HermiteSpline2<f64> = HermiteSpline2::with_auto_tangents(points.clone());

        // Should pass through control points
        for (i, p) in points.iter().enumerate() {
            let t = i as f64;
            let eval_p = spline.eval(t);
            assert_relative_eq!(eval_p.x, p.x, epsilon = 1e-10);
            assert_relative_eq!(eval_p.y, p.y, epsilon = 1e-10);
        }

        // Tangents should be computed
        assert!(spline.tangents[0].x != 0.0 || spline.tangents[0].y != 0.0);
    }

    #[test]
    fn test_zero_tangents() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)];

        let spline: HermiteSpline2<f64> = HermiteSpline2::with_zero_tangents(points);

        assert_relative_eq!(spline.tangents[0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(spline.tangents[0].y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cardinal() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        // Tension 0 should give Catmull-Rom-like tangents
        let spline: HermiteSpline2<f64> = HermiteSpline2::cardinal(points.clone(), 0.0);
        assert!(spline.tangents[1].x.abs() > 0.0);

        // Tension 1 should give zero tangents
        let spline_tight: HermiteSpline2<f64> = HermiteSpline2::cardinal(points, 1.0);
        assert_relative_eq!(spline_tight.tangents[1].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(spline_tight.tangents[1].y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tangent_direction() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];
        let tangents = vec![Vec2::new(1.0, 2.0), Vec2::new(1.0, -2.0)];

        let spline: HermiteSpline2<f64> = HermiteSpline2::new(points, tangents);

        // Tangent at start should match specified direction
        let t_start = spline.tangent_at(0.0);
        let len = (t_start.x * t_start.x + t_start.y * t_start.y).sqrt();
        let norm_x = t_start.x / len;
        let norm_y = t_start.y / len;

        let expected_len = (1.0_f64.powi(2) + 2.0_f64.powi(2)).sqrt();
        assert_relative_eq!(norm_x, 1.0 / expected_len, epsilon = 1e-6);
        assert_relative_eq!(norm_y, 2.0 / expected_len, epsilon = 1e-6);
    }

    #[test]
    fn test_s_curve() {
        // S-curve using horizontal tangents
        let points = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 2.0)];
        let tangents = vec![Vec2::new(2.0, 0.0), Vec2::new(2.0, 0.0)];

        let spline: HermiteSpline2<f64> = HermiteSpline2::new(points, tangents);

        // Midpoint should form S-shape
        let mid = spline.eval(0.5);
        assert_relative_eq!(mid.x, 1.0, epsilon = 1e-10);
        // y should be around 1.0 for smooth S-curve
        assert!(mid.y > 0.5 && mid.y < 1.5);
    }

    #[test]
    fn test_to_polyline() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];
        let tangents = vec![Vec2::new(1.0, 2.0), Vec2::new(1.0, -2.0)];

        let spline: HermiteSpline2<f64> = HermiteSpline2::new(points, tangents);
        let polyline = spline.to_polyline(0.1);

        assert!(polyline.len() >= 2);

        // First and last should match endpoints
        assert_relative_eq!(polyline.first().unwrap().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(polyline.last().unwrap().x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_length_straight() {
        // Straight line with matching tangents
        let points = vec![Point2::new(0.0, 0.0), Point2::new(3.0, 0.0)];
        let tangents = vec![Vec2::new(3.0, 0.0), Vec2::new(3.0, 0.0)];

        let spline: HermiteSpline2<f64> = HermiteSpline2::new(points, tangents);
        let length = spline.arc_length(0.01);

        assert_relative_eq!(length, 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_domain() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: HermiteSpline2<f64> = HermiteSpline2::with_auto_tangents(points);
        let (t_min, t_max) = spline.domain();

        assert_relative_eq!(t_min, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t_max, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_control_bounds() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, -1.0),
        ];

        let spline: HermiteSpline2<f64> = HermiteSpline2::with_auto_tangents(points);
        let (min, max) = spline.control_bounds();

        assert_relative_eq!(min.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(min.y, -1.0, epsilon = 1e-10);
        assert_relative_eq!(max.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(max.y, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scale_tangents() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let tangents = vec![Vec2::new(1.0, 1.0), Vec2::new(1.0, -1.0)];

        let mut spline: HermiteSpline2<f64> = HermiteSpline2::new(points, tangents);
        let mid_before = spline.eval(0.5);

        spline.scale_tangents(2.0);
        let mid_after = spline.eval(0.5);

        // Scaling tangents should change the curve shape
        assert!((mid_before.y - mid_after.y).abs() > 0.01);
    }

    #[test]
    fn test_f32_support() {
        let points: Vec<Point2<f32>> = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let tangents: Vec<Vec2<f32>> = vec![Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0)];

        let spline = HermiteSpline2::new(points, tangents);
        let mid = spline.eval(0.5);
        assert!((mid.x - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_multi_segment() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.0),
        ];

        let spline: HermiteSpline2<f64> = HermiteSpline2::with_auto_tangents(points.clone());

        // Should pass through all points
        for (i, p) in points.iter().enumerate() {
            let eval_p = spline.eval(i as f64);
            assert_relative_eq!(eval_p.x, p.x, epsilon = 1e-10);
            assert_relative_eq!(eval_p.y, p.y, epsilon = 1e-10);
        }
    }
}
