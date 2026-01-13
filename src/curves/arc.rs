//! Circular arc type and discretization.
//!
//! Provides circular arcs defined by center, radius, and angular range,
//! with conversion to polylines.

use crate::primitives::Point2;
use num_traits::Float;

/// A 2D circular arc defined by center, radius, and angular range.
///
/// Angles are in radians, measured counter-clockwise from the positive x-axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc2<F> {
    /// Center of the arc's circle.
    pub center: Point2<F>,
    /// Radius of the arc.
    pub radius: F,
    /// Start angle in radians.
    pub start_angle: F,
    /// End angle in radians.
    pub end_angle: F,
}

impl<F: Float> Arc2<F> {
    /// Creates a new arc.
    ///
    /// Angles are in radians, measured counter-clockwise from positive x-axis.
    /// The arc goes from `start_angle` to `end_angle` in the counter-clockwise
    /// direction. For clockwise arcs, use `start_angle > end_angle`.
    #[inline]
    pub fn new(center: Point2<F>, radius: F, start_angle: F, end_angle: F) -> Self {
        Self {
            center,
            radius,
            start_angle,
            end_angle,
        }
    }

    /// Creates an arc from three points on the circle.
    ///
    /// Returns `None` if the points are collinear.
    pub fn from_three_points(p1: Point2<F>, p2: Point2<F>, p3: Point2<F>) -> Option<Self> {
        // Find circumcenter and radius
        let d = (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
            * (F::one() + F::one());

        if d.abs() <= F::epsilon() {
            return None; // Collinear
        }

        let p1_sq = p1.x * p1.x + p1.y * p1.y;
        let p2_sq = p2.x * p2.x + p2.y * p2.y;
        let p3_sq = p3.x * p3.x + p3.y * p3.y;

        let cx = (p1_sq * (p2.y - p3.y) + p2_sq * (p3.y - p1.y) + p3_sq * (p1.y - p2.y)) / d;
        let cy = (p1_sq * (p3.x - p2.x) + p2_sq * (p1.x - p3.x) + p3_sq * (p2.x - p1.x)) / d;

        let center = Point2::new(cx, cy);
        let radius = center.distance(p1);

        // Compute angles for each point
        let angle1 = (p1.y - cy).atan2(p1.x - cx);
        let angle3 = (p3.y - cy).atan2(p3.x - cx);

        Some(Self::new(center, radius, angle1, angle3))
    }

    /// Creates an arc from start point, end point, and bulge factor.
    ///
    /// Bulge is the tangent of 1/4 of the arc's subtended angle.
    /// Positive bulge = counter-clockwise arc, negative = clockwise.
    /// Bulge of 1 = semicircle, 0 = straight line (returns None).
    pub fn from_bulge(start: Point2<F>, end: Point2<F>, bulge: F) -> Option<Self> {
        if bulge.abs() <= F::epsilon() {
            return None; // Straight line
        }

        let one = F::one();
        let two = one + one;
        let four = two + two;

        // Chord vector and length
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let chord_len = (dx * dx + dy * dy).sqrt();

        if chord_len <= F::epsilon() {
            return None;
        }

        // Sagitta (arc height) from bulge
        let sagitta = bulge * chord_len / two;

        // Radius from chord and sagitta
        let radius = (chord_len * chord_len / four + sagitta * sagitta) / (two * sagitta.abs());

        // Center is perpendicular to chord at midpoint
        let mid = start.midpoint(end);
        let perp_dist = radius - sagitta.abs();

        // Perpendicular direction (normalized)
        let perp_x = -dy / chord_len;
        let perp_y = dx / chord_len;

        // Center position depends on bulge sign
        let center = if bulge > F::zero() {
            Point2::new(mid.x + perp_x * perp_dist, mid.y + perp_y * perp_dist)
        } else {
            Point2::new(mid.x - perp_x * perp_dist, mid.y - perp_y * perp_dist)
        };

        let start_angle = (start.y - center.y).atan2(start.x - center.x);
        let end_angle = (end.y - center.y).atan2(end.x - center.x);

        Some(Self::new(center, radius, start_angle, end_angle))
    }

    /// Returns the point at the given angle on the arc's circle.
    #[inline]
    pub fn point_at_angle(&self, angle: F) -> Point2<F> {
        Point2::new(
            self.center.x + self.radius * angle.cos(),
            self.center.y + self.radius * angle.sin(),
        )
    }

    /// Returns the start point of the arc.
    #[inline]
    pub fn start_point(&self) -> Point2<F> {
        self.point_at_angle(self.start_angle)
    }

    /// Returns the end point of the arc.
    #[inline]
    pub fn end_point(&self) -> Point2<F> {
        self.point_at_angle(self.end_angle)
    }

    /// Returns the signed sweep angle (positive = counter-clockwise).
    #[inline]
    pub fn sweep_angle(&self) -> F {
        self.end_angle - self.start_angle
    }

    /// Returns the arc length.
    #[inline]
    pub fn arc_length(&self) -> F {
        self.radius * self.sweep_angle().abs()
    }

    /// Evaluates the arc at parameter `t` (0 = start, 1 = end).
    #[inline]
    pub fn eval(&self, t: F) -> Point2<F> {
        let angle = self.start_angle + t * self.sweep_angle();
        self.point_at_angle(angle)
    }

    /// Converts the arc to a polyline.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum allowed deviation from the true arc (sagitta).
    ///
    /// # Returns
    ///
    /// A vector of points approximating the arc.
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::{Point2, curves::Arc2};
    /// use std::f64::consts::PI;
    ///
    /// // Quarter circle
    /// let arc = Arc2::new(
    ///     Point2::new(0.0, 0.0),
    ///     1.0,
    ///     0.0,
    ///     PI / 2.0,
    /// );
    ///
    /// let polyline = arc.to_polyline(0.01);
    /// assert!(polyline.len() >= 2);
    /// ```
    pub fn to_polyline(&self, tolerance: F) -> Vec<Point2<F>> {
        let sweep = self.sweep_angle().abs();

        if sweep <= F::epsilon() {
            return vec![self.start_point()];
        }

        // Calculate number of segments needed
        // Sagitta formula: s = r * (1 - cos(θ/2))
        // Solving for θ: θ = 2 * acos(1 - s/r)
        let num_segments = self.segments_for_tolerance(tolerance);

        let mut points = Vec::with_capacity(num_segments + 1);
        let direction = if self.sweep_angle() >= F::zero() {
            F::one()
        } else {
            -F::one()
        };

        let step = sweep / F::from(num_segments).unwrap() * direction;

        for i in 0..=num_segments {
            let angle = self.start_angle + step * F::from(i).unwrap();
            points.push(self.point_at_angle(angle));
        }

        points
    }

    /// Calculates the number of segments needed for a given tolerance.
    fn segments_for_tolerance(&self, tolerance: F) -> usize {
        let one = F::one();
        let two = one + one;

        let sweep = self.sweep_angle().abs();

        // Handle degenerate cases
        if self.radius <= F::epsilon() || sweep <= F::epsilon() {
            return 1;
        }

        // If tolerance is larger than radius, one segment suffices
        if tolerance >= self.radius {
            return 1;
        }

        // Maximum angle per segment: θ = 2 * acos(1 - tolerance/radius)
        let ratio = (one - tolerance / self.radius).max(-one).min(one);
        let max_angle = two * ratio.acos();

        if max_angle <= F::epsilon() {
            return 1;
        }

        // Number of segments = ceil(sweep / max_angle)
        let n = (sweep / max_angle).ceil();
        n.to_usize().unwrap_or(1).max(1)
    }

    /// Returns `true` if this is a full circle (sweep >= 2π).
    pub fn is_full_circle(&self) -> bool {
        let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();
        self.sweep_angle().abs() >= two_pi - F::epsilon()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_arc_new() {
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, PI / 2.0);
        assert_eq!(arc.center.x, 0.0);
        assert_eq!(arc.radius, 1.0);
        assert_eq!(arc.start_angle, 0.0);
        assert_relative_eq!(arc.end_angle, PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_points() {
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, PI / 2.0);

        let start = arc.start_point();
        assert_relative_eq!(start.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(start.y, 0.0, epsilon = 1e-10);

        let end = arc.end_point();
        assert_relative_eq!(end.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(end.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_eval() {
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, PI);

        let mid = arc.eval(0.5);
        assert_relative_eq!(mid.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(mid.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_sweep_angle() {
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, PI);
        assert_relative_eq!(arc.sweep_angle(), PI, epsilon = 1e-10);

        // Clockwise arc
        let arc_cw: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, PI, 0.0);
        assert_relative_eq!(arc_cw.sweep_angle(), -PI, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_length() {
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, PI);
        assert_relative_eq!(arc.arc_length(), PI, epsilon = 1e-10);

        let arc2: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 2.0, 0.0, PI);
        assert_relative_eq!(arc2.arc_length(), 2.0 * PI, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_to_polyline() {
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, PI / 2.0);
        let polyline = arc.to_polyline(0.01);

        // Should have multiple points
        assert!(polyline.len() >= 2);

        // First point should be start
        assert_relative_eq!(polyline[0].x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(polyline[0].y, 0.0, epsilon = 1e-10);

        // Last point should be end
        let last = polyline.last().unwrap();
        assert_relative_eq!(last.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(last.y, 1.0, epsilon = 1e-10);

        // All points should be on the circle (within tolerance)
        for p in &polyline {
            let dist = p.distance(arc.center);
            assert_relative_eq!(dist, 1.0, epsilon = 0.02);
        }
    }

    #[test]
    fn test_arc_to_polyline_tolerance() {
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 10.0, 0.0, PI);

        let coarse = arc.to_polyline(1.0);
        let fine = arc.to_polyline(0.01);

        // Finer tolerance should produce more points
        assert!(fine.len() > coarse.len());
    }

    #[test]
    fn test_arc_to_polyline_full_circle() {
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, 2.0 * PI);
        let polyline = arc.to_polyline(0.01);

        // Should be a closed circle
        assert!(polyline.len() > 4);

        // First and last should be the same point
        let first = polyline.first().unwrap();
        let last = polyline.last().unwrap();
        assert_relative_eq!(first.x, last.x, epsilon = 1e-10);
        assert_relative_eq!(first.y, last.y, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_clockwise() {
        // Clockwise quarter arc
        let arc: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, PI / 2.0, 0.0);
        let polyline = arc.to_polyline(0.01);

        // Start should be at (0, 1)
        assert_relative_eq!(polyline[0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(polyline[0].y, 1.0, epsilon = 1e-10);

        // End should be at (1, 0)
        let last = polyline.last().unwrap();
        assert_relative_eq!(last.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(last.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_from_bulge() {
        // Semicircle (bulge = 1)
        let arc: Arc2<f64> = Arc2::from_bulge(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            1.0,
        )
        .unwrap();

        assert_relative_eq!(arc.radius, 1.0, epsilon = 1e-10);
        assert_relative_eq!(arc.center.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(arc.center.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_from_bulge_zero() {
        // Zero bulge = straight line = None
        let result: Option<Arc2<f64>> = Arc2::from_bulge(
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            0.0,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_arc_from_three_points() {
        // Three points on a unit circle
        let p1: Point2<f64> = Point2::new(1.0, 0.0);
        let p2 = Point2::new(0.0, 1.0);
        let p3 = Point2::new(-1.0, 0.0);

        let arc = Arc2::from_three_points(p1, p2, p3).unwrap();

        assert_relative_eq!(arc.center.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(arc.center.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(arc.radius, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_arc_from_three_points_collinear() {
        let p1: Point2<f64> = Point2::new(0.0, 0.0);
        let p2 = Point2::new(1.0, 0.0);
        let p3 = Point2::new(2.0, 0.0);

        let result = Arc2::from_three_points(p1, p2, p3);
        assert!(result.is_none());
    }

    #[test]
    fn test_is_full_circle() {
        let full: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, 2.0 * PI);
        assert!(full.is_full_circle());

        let half: Arc2<f64> = Arc2::new(Point2::new(0.0, 0.0), 1.0, 0.0, PI);
        assert!(!half.is_full_circle());
    }

    #[test]
    fn test_arc_f32() {
        let arc: Arc2<f32> = Arc2::new(
            Point2::new(0.0, 0.0),
            1.0,
            0.0,
            std::f32::consts::PI / 2.0,
        );
        let polyline = arc.to_polyline(0.01);
        assert!(polyline.len() >= 2);
    }
}
