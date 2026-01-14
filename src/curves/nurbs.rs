//! NURBS (Non-Uniform Rational B-Spline) curves.
//!
//! NURBS extend B-splines by adding weights to control points, enabling exact
//! representation of conic sections (circles, ellipses, parabolas, hyperbolas).
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, curves::Nurbs2};
//!
//! // Create a NURBS curve with weights
//! let control_points = vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(1.0, 1.0),
//!     Point2::new(2.0, 0.0),
//! ];
//! let weights = vec![1.0, 0.707, 1.0]; // Weight < 1 pulls curve away from control point
//!
//! let nurbs = Nurbs2::new(control_points, weights, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
//! let point = nurbs.eval(0.5);
//! ```
//!
//! # Circles and Arcs
//!
//! NURBS can represent circles exactly:
//!
//! ```
//! use approxum::{Point2, curves::Nurbs2};
//! use std::f64::consts::PI;
//!
//! // Create a circular arc
//! let arc = Nurbs2::circular_arc(
//!     Point2::new(0.0, 0.0), // center
//!     1.0,                    // radius
//!     0.0,                    // start angle
//!     PI / 2.0,              // end angle (quarter circle)
//! );
//! ```

use crate::primitives::Point2;
use num_traits::Float;

/// A NURBS (Non-Uniform Rational B-Spline) curve in 2D.
///
/// NURBS are defined by control points with associated weights. When all weights
/// are equal, the NURBS reduces to a standard B-spline.
#[derive(Debug, Clone, PartialEq)]
pub struct Nurbs2<F> {
    /// Control points defining the curve shape
    pub control_points: Vec<Point2<F>>,
    /// Weights for each control point (must be positive)
    pub weights: Vec<F>,
    /// Degree of the spline
    pub degree: usize,
    /// Knot vector (non-decreasing sequence)
    pub knots: Vec<F>,
}

impl<F: Float> Nurbs2<F> {
    /// Creates a new NURBS curve with the given control points, weights, degree, and knots.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `weights.len() != control_points.len()`
    /// - `knots.len() != control_points.len() + degree + 1`
    /// - `degree >= control_points.len()`
    /// - Any weight is non-positive
    /// - Knot vector is not non-decreasing
    pub fn new(
        control_points: Vec<Point2<F>>,
        weights: Vec<F>,
        degree: usize,
        knots: Vec<F>,
    ) -> Self {
        let n = control_points.len();
        assert!(
            weights.len() == n,
            "Number of weights must equal number of control points"
        );
        assert!(
            degree < n,
            "Degree must be less than number of control points"
        );
        assert!(
            knots.len() == n + degree + 1,
            "Knot vector length must be control_points.len() + degree + 1"
        );

        // Verify weights are positive
        for (i, &w) in weights.iter().enumerate() {
            assert!(w > F::zero(), "Weight {} must be positive", i);
        }

        // Verify knots are non-decreasing
        for i in 1..knots.len() {
            assert!(
                knots[i] >= knots[i - 1],
                "Knot vector must be non-decreasing"
            );
        }

        Self {
            control_points,
            weights,
            degree,
            knots,
        }
    }

    /// Creates a NURBS curve with uniform weights (equivalent to a B-spline).
    pub fn from_bspline(control_points: Vec<Point2<F>>, degree: usize, knots: Vec<F>) -> Self {
        let weights = vec![F::one(); control_points.len()];
        Self::new(control_points, weights, degree, knots)
    }

    /// Creates a NURBS curve with uniform clamped knots.
    pub fn with_uniform_knots(
        control_points: Vec<Point2<F>>,
        weights: Vec<F>,
        degree: usize,
    ) -> Self {
        let n = control_points.len();
        assert!(
            degree < n,
            "Degree must be less than number of control points"
        );

        let num_knots = n + degree + 1;
        let mut knots = Vec::with_capacity(num_knots);

        let zero = F::zero();

        // Clamped uniform knot vector
        for _ in 0..=degree {
            knots.push(zero);
        }

        let num_interior = n - degree - 1;
        for i in 1..=num_interior {
            knots.push(F::from(i).unwrap());
        }

        let max_knot = F::from(num_interior + 1).unwrap();
        for _ in 0..=degree {
            knots.push(max_knot);
        }

        Self::new(control_points, weights, degree, knots)
    }

    /// Creates a circular arc as a NURBS curve.
    ///
    /// # Arguments
    ///
    /// * `center` - Center of the circle
    /// * `radius` - Radius of the circle
    /// * `start_angle` - Starting angle in radians
    /// * `end_angle` - Ending angle in radians
    ///
    /// The arc is traversed counter-clockwise from `start_angle` to `end_angle`.
    /// For arcs greater than 90 degrees, multiple segments are used.
    pub fn circular_arc(center: Point2<F>, radius: F, start_angle: F, end_angle: F) -> Self {
        let mut angle_diff = end_angle - start_angle;

        // Normalize angle difference to be positive
        let two_pi = F::from(std::f64::consts::TAU).unwrap();
        while angle_diff < F::zero() {
            angle_diff = angle_diff + two_pi;
        }
        while angle_diff > two_pi {
            angle_diff = angle_diff - two_pi;
        }

        // Determine number of arc segments needed (max 90 degrees per segment)
        let half_pi = F::from(std::f64::consts::FRAC_PI_2).unwrap();
        let num_segments = (angle_diff / half_pi).ceil().to_usize().unwrap().max(1);
        let segment_angle = angle_diff / F::from(num_segments).unwrap();

        // Build control points and weights
        let mut control_points = Vec::new();
        let mut weights = Vec::new();

        let half_segment = segment_angle / (F::one() + F::one());
        let w = half_segment.cos(); // Weight for middle control points

        for seg in 0..num_segments {
            let seg_start = start_angle + F::from(seg).unwrap() * segment_angle;
            let seg_mid = seg_start + half_segment;
            let seg_end = seg_start + segment_angle;

            // Start point (or skip if not first segment)
            if seg == 0 {
                control_points.push(Point2::new(
                    center.x + radius * seg_start.cos(),
                    center.y + radius * seg_start.sin(),
                ));
                weights.push(F::one());
            }

            // Middle control point (on tangent lines)
            let tan_len = radius / w;
            control_points.push(Point2::new(
                center.x + tan_len * seg_mid.cos(),
                center.y + tan_len * seg_mid.sin(),
            ));
            weights.push(w);

            // End point
            control_points.push(Point2::new(
                center.x + radius * seg_end.cos(),
                center.y + radius * seg_end.sin(),
            ));
            weights.push(F::one());
        }

        // Build knot vector for degree-2 curve
        let n = control_points.len();
        let degree = 2usize;
        let mut knots = Vec::with_capacity(n + degree + 1);

        // Clamped knots: [0,0,0, 1,1, 2,2, ..., n,n,n]
        knots.push(F::zero());
        knots.push(F::zero());
        knots.push(F::zero());

        for i in 1..num_segments {
            let k = F::from(i).unwrap();
            knots.push(k);
            knots.push(k);
        }

        let max_k = F::from(num_segments).unwrap();
        knots.push(max_k);
        knots.push(max_k);
        knots.push(max_k);

        Self {
            control_points,
            weights,
            degree,
            knots,
        }
    }

    /// Creates a full circle as a NURBS curve.
    pub fn circle(center: Point2<F>, radius: F) -> Self {
        let two_pi = F::from(std::f64::consts::TAU).unwrap();
        Self::circular_arc(center, radius, F::zero(), two_pi)
    }

    /// Creates an ellipse as a NURBS curve.
    ///
    /// # Arguments
    ///
    /// * `center` - Center of the ellipse
    /// * `radius_x` - Radius along x-axis
    /// * `radius_y` - Radius along y-axis
    pub fn ellipse(center: Point2<F>, radius_x: F, radius_y: F) -> Self {
        // Create a unit circle and scale it
        let circle = Self::circle(Point2::new(F::zero(), F::zero()), F::one());

        // Scale control points
        let control_points: Vec<Point2<F>> = circle
            .control_points
            .iter()
            .map(|p| Point2::new(center.x + p.x * radius_x, center.y + p.y * radius_y))
            .collect();

        Self {
            control_points,
            weights: circle.weights,
            degree: circle.degree,
            knots: circle.knots,
        }
    }

    /// Returns the valid parameter range [t_min, t_max] for the curve.
    pub fn domain(&self) -> (F, F) {
        let p = self.degree;
        let n = self.control_points.len();
        (self.knots[p], self.knots[n])
    }

    /// Evaluates the NURBS curve at parameter `t` using the rational de Boor algorithm.
    ///
    /// # Panics
    ///
    /// Panics if `t` is outside the valid domain.
    pub fn eval(&self, t: F) -> Point2<F> {
        let (t_min, t_max) = self.domain();

        let eps = F::epsilon() * F::from(100.0).unwrap();
        let t = if t < t_min - eps {
            panic!("Parameter t is below the valid domain");
        } else if t > t_max + eps {
            panic!("Parameter t is above the valid domain");
        } else {
            t.max(t_min).min(t_max)
        };

        let k = self.find_knot_span(t);
        self.rational_de_boor(t, k)
    }

    /// Evaluates the NURBS curve at parameter `t`, returning None if outside domain.
    pub fn try_eval(&self, t: F) -> Option<Point2<F>> {
        let (t_min, t_max) = self.domain();
        let eps = F::epsilon() * F::from(100.0).unwrap();

        if t < t_min - eps || t > t_max + eps {
            return None;
        }

        let t = t.max(t_min).min(t_max);
        let k = self.find_knot_span(t);
        Some(self.rational_de_boor(t, k))
    }

    /// Finds the knot span index for parameter t.
    fn find_knot_span(&self, t: F) -> usize {
        let n = self.control_points.len();
        let p = self.degree;

        if t >= self.knots[n] {
            return n - 1;
        }

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

    /// Rational de Boor's algorithm for NURBS evaluation.
    ///
    /// Works in homogeneous coordinates (wx, wy, w) then projects back.
    fn rational_de_boor(&self, t: F, k: usize) -> Point2<F> {
        let p = self.degree;

        // Work in homogeneous coordinates: (w*x, w*y, w)
        let mut dw: Vec<(F, F, F)> = (0..=p)
            .map(|j| {
                let idx = k - p + j;
                let w = self.weights[idx];
                let pt = self.control_points[idx];
                (w * pt.x, w * pt.y, w)
            })
            .collect();

        // de Boor recursion in homogeneous space
        for r in 1..=p {
            for j in (r..=p).rev() {
                let i = k - p + j;
                let denom = self.knots[i + p - r + 1] - self.knots[i];

                let alpha = if denom.abs() < F::epsilon() {
                    F::zero()
                } else {
                    (t - self.knots[i]) / denom
                };

                let one_minus_alpha = F::one() - alpha;
                dw[j] = (
                    one_minus_alpha * dw[j - 1].0 + alpha * dw[j].0,
                    one_minus_alpha * dw[j - 1].1 + alpha * dw[j].1,
                    one_minus_alpha * dw[j - 1].2 + alpha * dw[j].2,
                );
            }
        }

        // Project from homogeneous coordinates
        let (wx, wy, w) = dw[p];
        Point2::new(wx / w, wy / w)
    }

    /// Converts the NURBS curve to a polyline using adaptive sampling.
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

        let line_mid = Point2::new((p0.x + p1.x) / two, (p0.y + p1.y) / two);
        let deviation = p_mid.distance(line_mid);

        if deviation <= tolerance {
            points.push(p1);
        } else {
            self.subdivide_recursive(t0, t_mid, tolerance, points);
            self.subdivide_recursive(t_mid, t1, tolerance, points);
        }
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

    /// Inserts a knot at parameter `t`, returning a new NURBS with one more control point.
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
        let k = self.find_knot_span(t);

        // New knot vector
        let mut new_knots = Vec::with_capacity(self.knots.len() + 1);
        new_knots.extend_from_slice(&self.knots[..=k]);
        new_knots.push(t);
        new_knots.extend_from_slice(&self.knots[k + 1..]);

        // New control points and weights (work in homogeneous coords)
        let mut new_points = Vec::with_capacity(n + 1);
        let mut new_weights = Vec::with_capacity(n + 1);

        for i in 0..=n {
            if i <= k - p {
                new_points.push(self.control_points[i]);
                new_weights.push(self.weights[i]);
            } else if i > k {
                new_points.push(self.control_points[i - 1]);
                new_weights.push(self.weights[i - 1]);
            } else {
                let denom = self.knots[i + p] - self.knots[i];
                let alpha = if denom.abs() < F::epsilon() {
                    F::zero()
                } else {
                    (t - self.knots[i]) / denom
                };

                let one_minus_alpha = F::one() - alpha;

                // Blend in homogeneous coordinates
                let w0 = self.weights[i - 1];
                let w1 = self.weights[i];
                let p0 = self.control_points[i - 1];
                let p1 = self.control_points[i];

                let new_w = one_minus_alpha * w0 + alpha * w1;
                let new_x = (one_minus_alpha * w0 * p0.x + alpha * w1 * p1.x) / new_w;
                let new_y = (one_minus_alpha * w0 * p0.y + alpha * w1 * p1.y) / new_w;

                new_points.push(Point2::new(new_x, new_y));
                new_weights.push(new_w);
            }
        }

        Self {
            control_points: new_points,
            weights: new_weights,
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

    /// Returns true if all weights are equal (curve is equivalent to B-spline).
    pub fn is_polynomial(&self) -> bool {
        if self.weights.is_empty() {
            return true;
        }
        let first = self.weights[0];
        let eps = F::epsilon() * F::from(100.0).unwrap();
        self.weights.iter().all(|&w| (w - first).abs() < eps)
    }

    /// Sets all weights to 1.0, converting to a non-rational B-spline.
    ///
    /// Note: This changes the curve shape unless it was already polynomial.
    pub fn make_polynomial(&mut self) {
        for w in &mut self.weights {
            *w = F::one();
        }
    }

    /// Scales all weights by a constant factor.
    ///
    /// This does not change the curve shape.
    pub fn scale_weights(&mut self, factor: F) {
        assert!(factor > F::zero(), "Scale factor must be positive");
        for w in &mut self.weights {
            *w = *w * factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn test_nurbs_creation() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];
        let weights = vec![1.0, 1.0, 1.0];
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let nurbs: Nurbs2<f64> = Nurbs2::new(points, weights, 2, knots);

        assert_eq!(nurbs.degree, 2);
        assert_eq!(nurbs.control_points.len(), 3);
        assert_eq!(nurbs.weights.len(), 3);
    }

    #[test]
    fn test_uniform_weights_equals_bspline() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        ];

        let nurbs: Nurbs2<f64> = Nurbs2::with_uniform_knots(points.clone(), vec![1.0, 1.0, 1.0], 2);

        // With uniform weights, should behave like quadratic Bezier
        let mid = nurbs.eval(0.5);
        assert_relative_eq!(mid.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mid.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_weight_effect() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        ];

        // High weight pulls curve toward control point
        let nurbs_high: Nurbs2<f64> =
            Nurbs2::with_uniform_knots(points.clone(), vec![1.0, 2.0, 1.0], 2);
        let mid_high = nurbs_high.eval(0.5);

        // Low weight pushes curve away from control point
        let nurbs_low: Nurbs2<f64> =
            Nurbs2::with_uniform_knots(points.clone(), vec![1.0, 0.5, 1.0], 2);
        let mid_low = nurbs_low.eval(0.5);

        // Unit weight
        let nurbs_unit: Nurbs2<f64> = Nurbs2::with_uniform_knots(points, vec![1.0, 1.0, 1.0], 2);
        let mid_unit = nurbs_unit.eval(0.5);

        // Higher weight should pull closer to control point (1, 2)
        assert!(mid_high.y > mid_unit.y);
        // Lower weight should push away from control point
        assert!(mid_low.y < mid_unit.y);
    }

    #[test]
    fn test_circular_arc_quarter() {
        let arc: Nurbs2<f64> = Nurbs2::circular_arc(Point2::new(0.0, 0.0), 1.0, 0.0, FRAC_PI_2);

        let (t_min, t_max) = arc.domain();

        // Start point should be (1, 0)
        let start = arc.eval(t_min);
        assert_relative_eq!(start.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(start.y, 0.0, epsilon = 1e-10);

        // End point should be (0, 1)
        let end = arc.eval(t_max);
        assert_relative_eq!(end.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(end.y, 1.0, epsilon = 1e-10);

        // Midpoint should be on circle at 45 degrees
        let mid = arc.eval((t_min + t_max) / 2.0);
        let dist = (mid.x * mid.x + mid.y * mid.y).sqrt();
        assert_relative_eq!(dist, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circular_arc_points_on_circle() {
        let radius = 2.5;
        let arc: Nurbs2<f64> = Nurbs2::circular_arc(Point2::new(1.0, 1.0), radius, 0.0, PI);

        let (t_min, t_max) = arc.domain();

        // Sample many points - all should be on the circle
        for i in 0..=20 {
            let t = t_min + (t_max - t_min) * (i as f64) / 20.0;
            let p = arc.eval(t);
            let dist = ((p.x - 1.0).powi(2) + (p.y - 1.0).powi(2)).sqrt();
            assert_relative_eq!(dist, radius, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_full_circle() {
        let circle: Nurbs2<f64> = Nurbs2::circle(Point2::new(0.0, 0.0), 1.0);

        let (t_min, t_max) = circle.domain();

        // All points should be on unit circle
        for i in 0..=40 {
            let t = t_min + (t_max - t_min) * (i as f64) / 40.0;
            let p = circle.eval(t);
            let dist = (p.x * p.x + p.y * p.y).sqrt();
            assert_relative_eq!(dist, 1.0, epsilon = 1e-9);
        }

        // Arc length should be 2*pi
        let length = circle.arc_length(0.001);
        assert_relative_eq!(length, 2.0 * PI, epsilon = 0.01);
    }

    #[test]
    fn test_ellipse() {
        let ellipse: Nurbs2<f64> = Nurbs2::ellipse(Point2::new(0.0, 0.0), 2.0, 1.0);

        let (t_min, t_max) = ellipse.domain();

        // Sample points should satisfy ellipse equation: (x/a)^2 + (y/b)^2 = 1
        for i in 0..=40 {
            let t = t_min + (t_max - t_min) * (i as f64) / 40.0;
            let p = ellipse.eval(t);
            let val = (p.x / 2.0).powi(2) + (p.y / 1.0).powi(2);
            assert_relative_eq!(val, 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_to_polyline() {
        let arc: Nurbs2<f64> = Nurbs2::circular_arc(Point2::new(0.0, 0.0), 1.0, 0.0, FRAC_PI_2);

        let polyline = arc.to_polyline(0.01);

        assert!(polyline.len() >= 2);

        // All points should be approximately on the circle
        for p in &polyline {
            let dist = (p.x * p.x + p.y * p.y).sqrt();
            assert_relative_eq!(dist, 1.0, epsilon = 0.02);
        }
    }

    #[test]
    fn test_knot_insertion_preserves_shape() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 0.0),
        ];
        let weights = vec![1.0, 0.8, 1.2, 1.0];

        let nurbs: Nurbs2<f64> = Nurbs2::with_uniform_knots(points, weights, 2);
        let (t_min, t_max) = nurbs.domain();

        let t_insert = (t_min + t_max) / 2.0;
        let new_nurbs = nurbs.insert_knot(t_insert);

        // Should have one more control point
        assert_eq!(new_nurbs.control_points.len(), 5);

        // Curve shape should be unchanged
        for i in 0..=10 {
            let t = t_min + (t_max - t_min) * (i as f64) / 10.0;
            let p1 = nurbs.eval(t);
            let p2 = new_nurbs.eval(t);

            assert_relative_eq!(p1.x, p2.x, epsilon = 1e-10);
            assert_relative_eq!(p1.y, p2.y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_is_polynomial() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let nurbs_poly: Nurbs2<f64> =
            Nurbs2::with_uniform_knots(points.clone(), vec![1.0, 1.0, 1.0], 2);
        assert!(nurbs_poly.is_polynomial());

        let nurbs_rational: Nurbs2<f64> =
            Nurbs2::with_uniform_knots(points, vec![1.0, 0.5, 1.0], 2);
        assert!(!nurbs_rational.is_polynomial());
    }

    #[test]
    fn test_scale_weights() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 0.0),
        ];

        let mut nurbs: Nurbs2<f64> = Nurbs2::with_uniform_knots(points, vec![1.0, 0.5, 1.0], 2);

        let mid_before = nurbs.eval(0.5);
        nurbs.scale_weights(2.0);
        let mid_after = nurbs.eval(0.5);

        // Scaling all weights shouldn't change the curve
        assert_relative_eq!(mid_before.x, mid_after.x, epsilon = 1e-10);
        assert_relative_eq!(mid_before.y, mid_after.y, epsilon = 1e-10);
    }

    #[test]
    fn test_try_eval() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let nurbs: Nurbs2<f64> = Nurbs2::with_uniform_knots(points, vec![1.0, 1.0, 1.0], 2);

        assert!(nurbs.try_eval(0.5).is_some());
        assert!(nurbs.try_eval(-1.0).is_none());
        assert!(nurbs.try_eval(100.0).is_none());
    }

    #[test]
    fn test_endpoints() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 0.0),
        ];

        let nurbs: Nurbs2<f64> = Nurbs2::with_uniform_knots(points, vec![1.0, 0.8, 1.2, 1.0], 2);

        let (t_min, t_max) = nurbs.domain();
        let start = nurbs.eval(t_min);
        let end = nurbs.eval(t_max);

        // With clamped knots, should pass through first and last control points
        assert_relative_eq!(start.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(start.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(end.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(end.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];

        let nurbs = Nurbs2::with_uniform_knots(points, vec![1.0, 0.7, 1.0], 2);
        let mid = nurbs.eval(0.5);

        assert!((mid.x - 1.0).abs() < 1e-5);
    }
}
