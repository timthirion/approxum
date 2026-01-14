//! 2D circle type.

use super::{Point2, Segment2, Vec2};
use num_traits::Float;

/// A 2D circle defined by center and radius.
///
/// # Example
///
/// ```
/// use approxum::primitives::{Circle2, Point2};
///
/// let circle: Circle2<f64> = Circle2::new(Point2::new(0.0, 0.0), 1.0);
/// assert!(circle.contains(Point2::new(0.5, 0.0)));
/// assert!(!circle.contains(Point2::new(2.0, 0.0)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle2<F> {
    /// Center point of the circle
    pub center: Point2<F>,
    /// Radius of the circle (must be non-negative)
    pub radius: F,
}

impl<F: Float> Circle2<F> {
    /// Creates a new circle from center and radius.
    #[inline]
    pub fn new(center: Point2<F>, radius: F) -> Self {
        Self { center, radius }
    }

    /// Creates a unit circle centered at the origin.
    #[inline]
    pub fn unit() -> Self {
        Self {
            center: Point2::origin(),
            radius: F::one(),
        }
    }

    /// Creates a circle from center coordinates and radius.
    #[inline]
    pub fn from_coords(cx: F, cy: F, radius: F) -> Self {
        Self {
            center: Point2::new(cx, cy),
            radius,
        }
    }

    /// Returns the diameter of the circle.
    #[inline]
    pub fn diameter(&self) -> F {
        self.radius + self.radius
    }

    /// Returns the area of the circle.
    #[inline]
    pub fn area(&self) -> F {
        F::from(std::f64::consts::PI).unwrap() * self.radius * self.radius
    }

    /// Returns the circumference of the circle.
    #[inline]
    pub fn circumference(&self) -> F {
        F::from(std::f64::consts::TAU).unwrap() * self.radius
    }

    /// Checks if a point is inside the circle (including boundary).
    #[inline]
    pub fn contains(&self, point: Point2<F>) -> bool {
        self.center.distance_squared(point) <= self.radius * self.radius
    }

    /// Checks if a point is strictly inside the circle (excluding boundary).
    #[inline]
    pub fn contains_strict(&self, point: Point2<F>) -> bool {
        self.center.distance_squared(point) < self.radius * self.radius
    }

    /// Returns the signed distance from a point to the circle boundary.
    ///
    /// Negative inside, positive outside.
    #[inline]
    pub fn signed_distance(&self, point: Point2<F>) -> F {
        self.center.distance(point) - self.radius
    }

    /// Returns the point on the circle at the given angle (in radians).
    ///
    /// Angle 0 is at (center.x + radius, center.y), increasing counter-clockwise.
    #[inline]
    pub fn point_at(&self, angle: F) -> Point2<F> {
        Point2::new(
            self.center.x + self.radius * angle.cos(),
            self.center.y + self.radius * angle.sin(),
        )
    }

    /// Returns the tangent vector at the given angle (in radians).
    ///
    /// The tangent points counter-clockwise.
    #[inline]
    pub fn tangent_at(&self, angle: F) -> Vec2<F> {
        Vec2::new(-angle.sin(), angle.cos())
    }

    /// Returns the closest point on the circle to the given point.
    ///
    /// If the point is at the center, returns the point at angle 0.
    #[inline]
    pub fn nearest_point(&self, point: Point2<F>) -> Point2<F> {
        let dx = point.x - self.center.x;
        let dy = point.y - self.center.y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < F::epsilon() {
            // Point is at center, return point at angle 0
            Point2::new(self.center.x + self.radius, self.center.y)
        } else {
            Point2::new(
                self.center.x + self.radius * dx / dist,
                self.center.y + self.radius * dy / dist,
            )
        }
    }

    /// Returns the axis-aligned bounding box as (min, max) points.
    #[inline]
    pub fn bounding_box(&self) -> (Point2<F>, Point2<F>) {
        (
            Point2::new(self.center.x - self.radius, self.center.y - self.radius),
            Point2::new(self.center.x + self.radius, self.center.y + self.radius),
        )
    }

    /// Converts the circle to a polygon with the given number of segments.
    pub fn to_polygon(&self, num_segments: usize) -> Vec<Point2<F>> {
        let n = num_segments.max(3);
        let tau = F::from(std::f64::consts::TAU).unwrap();

        (0..n)
            .map(|i| {
                let angle = tau * F::from(i).unwrap() / F::from(n).unwrap();
                self.point_at(angle)
            })
            .collect()
    }

    /// Finds intersection points between two circles.
    ///
    /// Returns:
    /// - `None` if circles don't intersect
    /// - `Some(vec)` with 1 point if circles are tangent
    /// - `Some(vec)` with 2 points if circles intersect at two points
    ///
    /// Note: Returns `None` if circles are identical (infinite intersections).
    pub fn intersect_circle(&self, other: &Circle2<F>) -> Option<Vec<Point2<F>>> {
        let d = self.center.distance(other.center);
        let r1 = self.radius;
        let r2 = other.radius;

        // No intersection if too far apart or one contains the other
        if d > r1 + r2 || d < (r1 - r2).abs() {
            return None;
        }

        // Coincident circles (infinite intersections)
        if d < F::epsilon() && (r1 - r2).abs() < F::epsilon() {
            return None;
        }

        // Distance from center1 to the line connecting intersection points
        let a = (r1 * r1 - r2 * r2 + d * d) / (d + d);
        let h_sq = r1 * r1 - a * a;

        // Direction from center1 to center2
        let dx = (other.center.x - self.center.x) / d;
        let dy = (other.center.y - self.center.y) / d;

        // Point on the line between centers
        let px = self.center.x + a * dx;
        let py = self.center.y + a * dy;

        if h_sq < F::epsilon() {
            // Tangent (one intersection point)
            return Some(vec![Point2::new(px, py)]);
        }

        let h = h_sq.sqrt();

        // Two intersection points
        Some(vec![
            Point2::new(px + h * dy, py - h * dx),
            Point2::new(px - h * dy, py + h * dx),
        ])
    }

    /// Finds intersection points between the circle and a line segment.
    ///
    /// Returns intersection points that lie on the segment.
    pub fn intersect_segment(&self, segment: &Segment2<F>) -> Vec<Point2<F>> {
        let d = segment.end - segment.start;
        let f = segment.start - self.center;

        let a = d.dot(d);
        let b = F::from(2.0).unwrap() * f.dot(d);
        let c = f.dot(f) - self.radius * self.radius;

        let discriminant = b * b - F::from(4.0).unwrap() * a * c;

        if discriminant < F::zero() {
            return Vec::new();
        }

        let mut results = Vec::new();
        let sqrt_disc = discriminant.sqrt();

        let t1 = (-b - sqrt_disc) / (F::from(2.0).unwrap() * a);
        let t2 = (-b + sqrt_disc) / (F::from(2.0).unwrap() * a);

        if t1 >= F::zero() && t1 <= F::one() {
            results.push(segment.start.lerp(segment.end, t1));
        }

        if discriminant > F::epsilon() && t2 >= F::zero() && t2 <= F::one() {
            results.push(segment.start.lerp(segment.end, t2));
        }

        results
    }

    /// Checks if this circle intersects another circle.
    #[inline]
    pub fn intersects_circle(&self, other: &Circle2<F>) -> bool {
        let d = self.center.distance(other.center);
        let r_sum = self.radius + other.radius;
        let r_diff = (self.radius - other.radius).abs();
        d <= r_sum && d >= r_diff
    }

    /// Checks if this circle intersects a line segment.
    #[inline]
    pub fn intersects_segment(&self, segment: &Segment2<F>) -> bool {
        segment.distance_to_point(self.center) <= self.radius
    }

    /// Returns a circle scaled by the given factor around its center.
    #[inline]
    pub fn scaled(&self, factor: F) -> Self {
        Self {
            center: self.center,
            radius: self.radius * factor,
        }
    }

    /// Returns a circle translated by the given vector.
    #[inline]
    pub fn translated(&self, offset: Vec2<F>) -> Self {
        Self {
            center: self.center + offset,
            radius: self.radius,
        }
    }
}

impl<F: Float> Default for Circle2<F> {
    fn default() -> Self {
        Self::unit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new() {
        let c: Circle2<f64> = Circle2::new(Point2::new(1.0, 2.0), 3.0);
        assert_eq!(c.center.x, 1.0);
        assert_eq!(c.center.y, 2.0);
        assert_eq!(c.radius, 3.0);
    }

    #[test]
    fn test_unit() {
        let c: Circle2<f64> = Circle2::unit();
        assert_eq!(c.center.x, 0.0);
        assert_eq!(c.center.y, 0.0);
        assert_eq!(c.radius, 1.0);
    }

    #[test]
    fn test_diameter() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 5.0);
        assert_eq!(c.diameter(), 10.0);
    }

    #[test]
    fn test_area() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);
        assert_relative_eq!(c.area(), std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_circumference() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);
        assert_relative_eq!(c.circumference(), std::f64::consts::TAU, epsilon = 1e-10);
    }

    #[test]
    fn test_contains() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);

        assert!(c.contains(Point2::new(0.0, 0.0))); // Center
        assert!(c.contains(Point2::new(1.0, 0.0))); // On boundary
        assert!(c.contains(Point2::new(0.5, 0.5))); // Inside
        assert!(!c.contains(Point2::new(1.0, 1.0))); // Outside
    }

    #[test]
    fn test_contains_strict() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);

        assert!(c.contains_strict(Point2::new(0.0, 0.0)));
        assert!(!c.contains_strict(Point2::new(1.0, 0.0))); // On boundary
        assert!(c.contains_strict(Point2::new(0.5, 0.0)));
    }

    #[test]
    fn test_signed_distance() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);

        assert_relative_eq!(c.signed_distance(Point2::new(0.0, 0.0)), -1.0);
        assert_relative_eq!(
            c.signed_distance(Point2::new(1.0, 0.0)),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(c.signed_distance(Point2::new(2.0, 0.0)), 1.0);
    }

    #[test]
    fn test_point_at() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);

        let p0 = c.point_at(0.0);
        assert_relative_eq!(p0.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 0.0, epsilon = 1e-10);

        let p90 = c.point_at(std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(p90.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p90.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_point() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);

        // Point outside
        let nearest = c.nearest_point(Point2::new(2.0, 0.0));
        assert_relative_eq!(nearest.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(nearest.y, 0.0, epsilon = 1e-10);

        // Point inside
        let nearest = c.nearest_point(Point2::new(0.5, 0.0));
        assert_relative_eq!(nearest.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(nearest.y, 0.0, epsilon = 1e-10);

        // Point at center
        let nearest = c.nearest_point(Point2::new(0.0, 0.0));
        assert_relative_eq!(nearest.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(nearest.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bounding_box() {
        let c: Circle2<f64> = Circle2::new(Point2::new(1.0, 2.0), 3.0);
        let (min, max) = c.bounding_box();
        assert_eq!(min.x, -2.0);
        assert_eq!(min.y, -1.0);
        assert_eq!(max.x, 4.0);
        assert_eq!(max.y, 5.0);
    }

    #[test]
    fn test_to_polygon() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);
        let poly = c.to_polygon(4);

        assert_eq!(poly.len(), 4);
        assert_relative_eq!(poly[0].x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(poly[0].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(poly[1].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(poly[1].y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_circle_no_intersection() {
        let c1: Circle2<f64> = Circle2::new(Point2::new(0.0, 0.0), 1.0);
        let c2: Circle2<f64> = Circle2::new(Point2::new(5.0, 0.0), 1.0);

        assert!(c1.intersect_circle(&c2).is_none());
    }

    #[test]
    fn test_intersect_circle_tangent() {
        let c1: Circle2<f64> = Circle2::new(Point2::new(0.0, 0.0), 1.0);
        let c2: Circle2<f64> = Circle2::new(Point2::new(2.0, 0.0), 1.0);

        let result = c1.intersect_circle(&c2).unwrap();
        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0].x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[0].y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_circle_two_points() {
        let c1: Circle2<f64> = Circle2::new(Point2::new(0.0, 0.0), 1.0);
        let c2: Circle2<f64> = Circle2::new(Point2::new(1.0, 0.0), 1.0);

        let result = c1.intersect_circle(&c2).unwrap();
        assert_eq!(result.len(), 2);

        // Points should be at x = 0.5, y = ±sqrt(3)/2
        let sqrt3_2 = (3.0_f64).sqrt() / 2.0;
        assert_relative_eq!(result[0].x, 0.5, epsilon = 1e-10);
        assert_relative_eq!(result[0].y.abs(), sqrt3_2, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_segment() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);

        // Segment through center
        let seg = Segment2::new(Point2::new(-2.0, 0.0), Point2::new(2.0, 0.0));
        let result = c.intersect_segment(&seg);
        assert_eq!(result.len(), 2);

        // Segment tangent to circle
        let seg = Segment2::new(Point2::new(1.0, -2.0), Point2::new(1.0, 2.0));
        let result = c.intersect_segment(&seg);
        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0].x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[0].y, 0.0, epsilon = 1e-10);

        // Segment outside circle
        let seg = Segment2::new(Point2::new(2.0, 0.0), Point2::new(3.0, 0.0));
        let result = c.intersect_segment(&seg);
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersects_circle() {
        let c1: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);
        let c2: Circle2<f64> = Circle2::new(Point2::new(1.5, 0.0), 1.0);
        let c3: Circle2<f64> = Circle2::new(Point2::new(5.0, 0.0), 1.0);

        assert!(c1.intersects_circle(&c2));
        assert!(!c1.intersects_circle(&c3));
    }

    #[test]
    fn test_scaled() {
        let c: Circle2<f64> = Circle2::new(Point2::new(1.0, 1.0), 2.0);
        let scaled = c.scaled(2.0);
        assert_eq!(scaled.center, c.center);
        assert_eq!(scaled.radius, 4.0);
    }

    #[test]
    fn test_translated() {
        let c: Circle2<f64> = Circle2::new(Point2::origin(), 1.0);
        let translated = c.translated(Vec2::new(3.0, 4.0));
        assert_eq!(translated.center.x, 3.0);
        assert_eq!(translated.center.y, 4.0);
        assert_eq!(translated.radius, 1.0);
    }

    #[test]
    fn test_f32_support() {
        let c: Circle2<f32> = Circle2::new(Point2::new(1.0, 2.0), 3.0);
        assert!(c.contains(Point2::new(1.0, 2.0)));
    }
}
