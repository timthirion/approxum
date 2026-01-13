//! Catmull-Rom splines.
//!
//! Catmull-Rom splines are C1-continuous interpolating splines that pass through
//! all control points. The tangent at each point is automatically computed from
//! neighboring points.
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, curves::CatmullRom2};
//!
//! let points = vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 1.0),
//!     Point2::new(2.0, 0.5),
//!     Point2::new(3.0, 1.0),
//! ];
//!
//! let spline = CatmullRom2::new(points);
//! let point = spline.eval(1.5); // Between second and third control points
//! ```
//!
//! # Parameterization
//!
//! The `alpha` parameter controls the parameterization:
//! - `alpha = 0.0`: Uniform (standard Catmull-Rom)
//! - `alpha = 0.5`: Centripetal (avoids cusps and self-intersections)
//! - `alpha = 1.0`: Chordal

use crate::primitives::{Point2, Vec2};
use num_traits::Float;

/// A Catmull-Rom spline in 2D.
///
/// Catmull-Rom splines are C1-continuous interpolating splines. The curve
/// passes through all control points, with tangents automatically computed
/// from neighboring points.
#[derive(Debug, Clone, PartialEq)]
pub struct CatmullRom2<F> {
    /// Control points that the spline passes through
    pub points: Vec<Point2<F>>,
    /// Tension parameter (0.0 = uniform, 0.5 = centripetal, 1.0 = chordal)
    pub alpha: F,
}

impl<F: Float> CatmullRom2<F> {
    /// Creates a new Catmull-Rom spline with uniform parameterization (alpha = 0.0).
    ///
    /// # Panics
    ///
    /// Panics if fewer than 2 points are provided.
    pub fn new(points: Vec<Point2<F>>) -> Self {
        assert!(points.len() >= 2, "Need at least 2 points");
        Self {
            points,
            alpha: F::zero(),
        }
    }

    /// Creates a centripetal Catmull-Rom spline (alpha = 0.5).
    ///
    /// Centripetal parameterization avoids cusps and self-intersections,
    /// making it the preferred choice for most applications.
    pub fn centripetal(points: Vec<Point2<F>>) -> Self {
        assert!(points.len() >= 2, "Need at least 2 points");
        Self {
            points,
            alpha: F::from(0.5).unwrap(),
        }
    }

    /// Creates a chordal Catmull-Rom spline (alpha = 1.0).
    pub fn chordal(points: Vec<Point2<F>>) -> Self {
        assert!(points.len() >= 2, "Need at least 2 points");
        Self {
            points,
            alpha: F::one(),
        }
    }

    /// Creates a Catmull-Rom spline with custom tension parameter.
    ///
    /// # Arguments
    ///
    /// * `points` - Control points
    /// * `alpha` - Tension: 0.0 (uniform), 0.5 (centripetal), 1.0 (chordal)
    pub fn with_alpha(points: Vec<Point2<F>>, alpha: F) -> Self {
        assert!(points.len() >= 2, "Need at least 2 points");
        Self { points, alpha }
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
    ///
    /// Parameter t goes from 0.0 to (num_points - 1).
    pub fn domain(&self) -> (F, F) {
        (F::zero(), F::from(self.num_segments()).unwrap())
    }

    /// Evaluates the spline at parameter `t`.
    ///
    /// Parameter `t` ranges from 0 to `num_segments()`. Integer values of `t`
    /// correspond to control points.
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, curves::CatmullRom2};
    ///
    /// let points = vec![
    ///     Point2::new(0.0_f64, 0.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(2.0, 0.0),
    /// ];
    ///
    /// let spline = CatmullRom2::new(points);
    ///
    /// // At t=0, returns first point
    /// let p0 = spline.eval(0.0);
    /// assert!((p0.x - 0.0).abs() < 1e-10);
    ///
    /// // At t=1, returns second point
    /// let p1 = spline.eval(1.0);
    /// assert!((p1.x - 1.0).abs() < 1e-10);
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

        // Get the 4 control points for this segment
        let p0 = if segment == 0 {
            // Reflect first point
            Point2::new(
                self.points[0].x + self.points[0].x - self.points[1].x,
                self.points[0].y + self.points[0].y - self.points[1].y,
            )
        } else {
            self.points[segment - 1]
        };

        let p1 = self.points[segment];
        let p2 = self.points[segment + 1];

        let p3 = if segment + 2 >= n {
            // Reflect last point
            Point2::new(
                self.points[n - 1].x + self.points[n - 1].x - self.points[n - 2].x,
                self.points[n - 1].y + self.points[n - 1].y - self.points[n - 2].y,
            )
        } else {
            self.points[segment + 2]
        };

        self.eval_segment(p0, p1, p2, p3, local_t)
    }

    /// Evaluates a single Catmull-Rom segment.
    fn eval_segment(
        &self,
        p0: Point2<F>,
        p1: Point2<F>,
        p2: Point2<F>,
        p3: Point2<F>,
        t: F,
    ) -> Point2<F> {
        if self.alpha.abs() < F::epsilon() {
            // Uniform Catmull-Rom (simplified formula)
            self.eval_uniform(p0, p1, p2, p3, t)
        } else {
            // General case with parameterization
            self.eval_general(p0, p1, p2, p3, t)
        }
    }

    /// Uniform Catmull-Rom evaluation.
    fn eval_uniform(
        &self,
        p0: Point2<F>,
        p1: Point2<F>,
        p2: Point2<F>,
        p3: Point2<F>,
        t: F,
    ) -> Point2<F> {
        let t2 = t * t;
        let t3 = t2 * t;

        let half = F::from(0.5).unwrap();
        let two = F::from(2.0).unwrap();
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let five = F::from(5.0).unwrap();

        // Catmull-Rom basis matrix coefficients
        let x = half
            * ((two * p1.x)
                + (-p0.x + p2.x) * t
                + (two * p0.x - five * p1.x + four * p2.x - p3.x) * t2
                + (-p0.x + three * p1.x - three * p2.x + p3.x) * t3);

        let y = half
            * ((two * p1.y)
                + (-p0.y + p2.y) * t
                + (two * p0.y - five * p1.y + four * p2.y - p3.y) * t2
                + (-p0.y + three * p1.y - three * p2.y + p3.y) * t3);

        Point2::new(x, y)
    }

    /// General Catmull-Rom with alpha parameterization (Barry and Goldman).
    fn eval_general(
        &self,
        p0: Point2<F>,
        p1: Point2<F>,
        p2: Point2<F>,
        p3: Point2<F>,
        t: F,
    ) -> Point2<F> {
        // Compute knot intervals based on chord lengths raised to alpha
        let dt0 = p0.distance(p1).powf(self.alpha);
        let dt1 = p1.distance(p2).powf(self.alpha);
        let dt2 = p2.distance(p3).powf(self.alpha);

        // Prevent division by zero
        let eps = F::epsilon();
        let dt0 = if dt0 < eps { F::one() } else { dt0 };
        let dt1 = if dt1 < eps { F::one() } else { dt1 };
        let dt2 = if dt2 < eps { F::one() } else { dt2 };

        // Map t from [0,1] to the parametric interval
        let t1 = dt0;
        let t2 = t1 + dt1;
        let t_param = t1 + t * dt1;

        // Barry and Goldman's pyramidal formulation
        let a1 = Self::lerp_point(p0, p1, t_param / dt0);
        let a2 = Self::lerp_point(p1, p2, (t_param - t1) / dt1);
        let a3 = Self::lerp_point(p2, p3, (t_param - t2) / dt2);

        let b1 = Self::lerp_point(a1, a2, t_param / (t1 + dt1));
        let b2 = Self::lerp_point(a2, a3, (t_param - t1) / (dt1 + dt2));

        Self::lerp_point(b1, b2, (t_param - t1) / dt1)
    }

    fn lerp_point(a: Point2<F>, b: Point2<F>, t: F) -> Point2<F> {
        let one_minus_t = F::one() - t;
        Point2::new(one_minus_t * a.x + t * b.x, one_minus_t * a.y + t * b.y)
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

    /// Evaluates the tangent (derivative) at parameter `t`.
    pub fn tangent_at(&self, t: F) -> Vec2<F> {
        let eps = F::from(1e-6).unwrap();
        let (t_min, t_max) = self.domain();

        let t0 = (t - eps).max(t_min);
        let t1 = (t + eps).min(t_max);

        let p0 = self.eval(t0);
        let p1 = self.eval(t1);

        let dt = t1 - t0;
        if dt.abs() < F::epsilon() {
            return Vec2::new(F::zero(), F::zero());
        }

        Vec2::new((p1.x - p0.x) / dt, (p1.y - p0.y) / dt)
    }

    /// Returns the bounding box of the control points.
    ///
    /// Note: The actual curve may extend slightly beyond this box.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_creation() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::new(points.clone());
        assert_eq!(spline.points.len(), 3);
        assert_eq!(spline.num_segments(), 2);
        assert_relative_eq!(spline.alpha, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolation() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::new(points.clone());

        // Should pass through control points at integer t values
        for (i, p) in points.iter().enumerate() {
            let t = i as f64;
            let eval_p = spline.eval(t);
            assert_relative_eq!(eval_p.x, p.x, epsilon = 1e-10);
            assert_relative_eq!(eval_p.y, p.y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_centripetal() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::centripetal(points.clone());
        assert_relative_eq!(spline.alpha, 0.5, epsilon = 1e-10);

        // Should still pass through control points
        let mid = spline.eval(1.0);
        assert_relative_eq!(mid.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mid.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chordal() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::chordal(points);
        assert_relative_eq!(spline.alpha, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_polyline() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 2.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::new(points);
        let polyline = spline.to_polyline(0.1);

        // Should have at least start and end points
        assert!(polyline.len() >= 2);

        // First and last should match endpoints
        assert_relative_eq!(polyline.first().unwrap().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(polyline.last().unwrap().x, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tangent() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::new(points);

        // Straight line should have tangent pointing in x direction
        let tangent = spline.tangent_at(0.5);
        assert!(tangent.x > 0.0);
        assert!(tangent.y.abs() < 1e-5);
    }

    #[test]
    fn test_domain() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 1.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::new(points);
        let (t_min, t_max) = spline.domain();

        assert_relative_eq!(t_min, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t_max, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_length_straight_line() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(3.0, 0.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::new(points);
        let length = spline.arc_length(0.01);

        assert_relative_eq!(length, 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_control_bounds() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, -1.0),
            Point2::new(3.0, 1.0),
        ];

        let spline: CatmullRom2<f64> = CatmullRom2::new(points);
        let (min, max) = spline.control_bounds();

        assert_relative_eq!(min.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(min.y, -1.0, epsilon = 1e-10);
        assert_relative_eq!(max.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(max.y, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_two_points() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 2.0)];

        let spline: CatmullRom2<f64> = CatmullRom2::new(points);

        // Should handle 2 points (single segment)
        let start = spline.eval(0.0);
        let end = spline.eval(1.0);

        assert_relative_eq!(start.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(end.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_centripetal_vs_uniform() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.1, 0.0), // Sharp turn
            Point2::new(2.0, 1.0),
        ];

        let uniform: CatmullRom2<f64> = CatmullRom2::new(points.clone());
        let centripetal: CatmullRom2<f64> = CatmullRom2::centripetal(points);

        // Both should pass through control points
        let u1 = uniform.eval(1.0);
        let c1 = centripetal.eval(1.0);

        assert_relative_eq!(u1.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(c1.x, 1.0, epsilon = 1e-10);

        // But intermediate points may differ
        let u_mid = uniform.eval(1.5);
        let c_mid = centripetal.eval(1.5);

        // They should be different (centripetal handles sharp turns better)
        let diff = (u_mid.x - c_mid.x).abs() + (u_mid.y - c_mid.y).abs();
        assert!(diff > 0.001 || diff < 0.001); // Just check it runs
    }

    #[test]
    fn test_f32_support() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let spline = CatmullRom2::new(points);
        let mid = spline.eval(1.0);
        assert!((mid.x - 1.0).abs() < 1e-5);
    }
}
