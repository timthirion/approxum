//! 3D line segment type.

use super::{Point3, Vec3};
use num_traits::Float;

/// A 3D line segment defined by two endpoints.
///
/// Generic over floating-point types (`f32` or `f64`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment3<F> {
    pub start: Point3<F>,
    pub end: Point3<F>,
}

impl<F: Float> Segment3<F> {
    /// Creates a new segment from two points.
    #[inline]
    pub fn new(start: Point3<F>, end: Point3<F>) -> Self {
        Self { start, end }
    }

    /// Creates a segment from coordinate tuples.
    #[inline]
    pub fn from_coords(x1: F, y1: F, z1: F, x2: F, y2: F, z2: F) -> Self {
        Self {
            start: Point3::new(x1, y1, z1),
            end: Point3::new(x2, y2, z2),
        }
    }

    /// Returns the direction vector from start to end.
    #[inline]
    pub fn direction(self) -> Vec3<F> {
        self.end - self.start
    }

    /// Returns the squared length of the segment.
    #[inline]
    pub fn length_squared(self) -> F {
        self.start.distance_squared(self.end)
    }

    /// Returns the length of the segment.
    #[inline]
    pub fn length(self) -> F {
        self.start.distance(self.end)
    }

    /// Returns the midpoint of the segment.
    #[inline]
    pub fn midpoint(self) -> Point3<F> {
        self.start.midpoint(self.end)
    }

    /// Returns the point at parameter `t` along the segment.
    ///
    /// - `t = 0` returns `start`
    /// - `t = 1` returns `end`
    /// - Values outside [0, 1] extrapolate beyond the segment
    #[inline]
    pub fn point_at(self, t: F) -> Point3<F> {
        self.start.lerp(self.end, t)
    }

    /// Returns the reversed segment (start and end swapped).
    #[inline]
    pub fn reversed(self) -> Self {
        Self {
            start: self.end,
            end: self.start,
        }
    }

    /// Computes the closest point on the segment to the given point.
    ///
    /// Returns a tuple of (closest_point, parameter_t) where t is in [0, 1].
    pub fn closest_point(self, p: Point3<F>) -> (Point3<F>, F) {
        let v = self.direction();
        let len_sq = v.magnitude_squared();

        // Degenerate segment (start == end)
        if len_sq <= F::epsilon() {
            return (self.start, F::zero());
        }

        // Project p onto the line, clamping to [0, 1]
        let t = (p - self.start).dot(v) / len_sq;
        let t_clamped = t.max(F::zero()).min(F::one());

        (self.point_at(t_clamped), t_clamped)
    }

    /// Computes the squared distance from a point to this segment.
    #[inline]
    pub fn distance_squared_to_point(self, p: Point3<F>) -> F {
        let (closest, _) = self.closest_point(p);
        p.distance_squared(closest)
    }

    /// Computes the distance from a point to this segment.
    #[inline]
    pub fn distance_to_point(self, p: Point3<F>) -> F {
        self.distance_squared_to_point(p).sqrt()
    }

    /// Returns `true` if the segment is degenerate (start equals end within epsilon).
    #[inline]
    pub fn is_degenerate(self, eps: F) -> bool {
        self.length_squared() <= eps * eps
    }
}

impl<F: Float> From<(Point3<F>, Point3<F>)> for Segment3<F> {
    fn from((start, end): (Point3<F>, Point3<F>)) -> Self {
        Self::new(start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new() {
        let s: Segment3<f64> =
            Segment3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 3.0, 6.0));
        assert_eq!(s.start.x, 0.0);
        assert_eq!(s.end.x, 2.0);
    }

    #[test]
    fn test_from_coords() {
        let s: Segment3<f64> = Segment3::from_coords(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(s.start.x, 1.0);
        assert_eq!(s.start.y, 2.0);
        assert_eq!(s.start.z, 3.0);
        assert_eq!(s.end.x, 4.0);
        assert_eq!(s.end.y, 5.0);
        assert_eq!(s.end.z, 6.0);
    }

    #[test]
    fn test_direction() {
        let s: Segment3<f64> = Segment3::from_coords(1.0, 1.0, 1.0, 4.0, 5.0, 7.0);
        let d = s.direction();
        assert_eq!(d.x, 3.0);
        assert_eq!(d.y, 4.0);
        assert_eq!(d.z, 6.0);
    }

    #[test]
    fn test_length() {
        let s: Segment3<f64> = Segment3::from_coords(0.0, 0.0, 0.0, 2.0, 3.0, 6.0);
        assert_eq!(s.length_squared(), 49.0);
        assert_eq!(s.length(), 7.0);
    }

    #[test]
    fn test_midpoint() {
        let s: Segment3<f64> = Segment3::from_coords(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        let m = s.midpoint();
        assert_eq!(m.x, 5.0);
        assert_eq!(m.y, 10.0);
        assert_eq!(m.z, 15.0);
    }

    #[test]
    fn test_point_at() {
        let s: Segment3<f64> = Segment3::from_coords(0.0, 0.0, 0.0, 10.0, 0.0, 0.0);

        let p0 = s.point_at(0.0);
        assert_eq!(p0.x, 0.0);

        let p1 = s.point_at(1.0);
        assert_eq!(p1.x, 10.0);

        let p_mid = s.point_at(0.5);
        assert_eq!(p_mid.x, 5.0);
    }

    #[test]
    fn test_reversed() {
        let s: Segment3<f64> = Segment3::from_coords(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let r = s.reversed();
        assert_eq!(r.start.x, 4.0);
        assert_eq!(r.start.y, 5.0);
        assert_eq!(r.start.z, 6.0);
        assert_eq!(r.end.x, 1.0);
        assert_eq!(r.end.y, 2.0);
        assert_eq!(r.end.z, 3.0);
    }

    #[test]
    fn test_closest_point_on_segment() {
        // Segment along x-axis from 0 to 10
        let s: Segment3<f64> = Segment3::from_coords(0.0, 0.0, 0.0, 10.0, 0.0, 0.0);

        // Point above midpoint
        let p1 = Point3::new(5.0, 5.0, 0.0);
        let (closest1, t1) = s.closest_point(p1);
        assert_relative_eq!(closest1.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(closest1.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(closest1.z, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t1, 0.5, epsilon = 1e-10);

        // Point beyond start
        let p2 = Point3::new(-5.0, 0.0, 0.0);
        let (closest2, t2) = s.closest_point(p2);
        assert_relative_eq!(closest2.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t2, 0.0, epsilon = 1e-10);

        // Point beyond end
        let p3 = Point3::new(15.0, 0.0, 0.0);
        let (closest3, t3) = s.closest_point(p3);
        assert_relative_eq!(closest3.x, 10.0, epsilon = 1e-10);
        assert_relative_eq!(t3, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_to_point() {
        let s: Segment3<f64> = Segment3::from_coords(0.0, 0.0, 0.0, 10.0, 0.0, 0.0);

        // Point above segment (perpendicular distance)
        let p = Point3::new(5.0, 3.0, 4.0);
        assert_relative_eq!(s.distance_to_point(p), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_degenerate() {
        let degen: Segment3<f64> = Segment3::from_coords(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(degen.is_degenerate(1e-10));

        let normal: Segment3<f64> = Segment3::from_coords(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert!(!normal.is_degenerate(1e-10));
    }

    #[test]
    fn test_degenerate_closest_point() {
        // Degenerate segment (point)
        let s: Segment3<f64> = Segment3::from_coords(5.0, 5.0, 5.0, 5.0, 5.0, 5.0);
        let p = Point3::new(0.0, 0.0, 0.0);
        let (closest, t) = s.closest_point(p);
        assert_eq!(closest.x, 5.0);
        assert_eq!(closest.y, 5.0);
        assert_eq!(closest.z, 5.0);
        assert_eq!(t, 0.0);
    }
}
