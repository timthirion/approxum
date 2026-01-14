//! 2D infinite line type.

use super::{Circle2, Point2, Ray2, Segment2, Vec2};
use num_traits::Float;

/// A 2D infinite line defined by a point and direction.
///
/// The line extends infinitely in both directions through the origin point.
///
/// # Example
///
/// ```
/// use approxum::primitives::{Line2, Point2, Vec2};
///
/// // Horizontal line through y=1
/// let line: Line2<f64> = Line2::new(Point2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
/// assert_eq!(line.signed_distance(Point2::new(5.0, 3.0)), 2.0);
/// assert_eq!(line.signed_distance(Point2::new(5.0, -1.0)), -2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line2<F> {
    /// A point on the line
    pub origin: Point2<F>,
    /// Direction vector of the line (not necessarily normalized)
    pub direction: Vec2<F>,
}

impl<F: Float> Line2<F> {
    /// Creates a new line from a point and direction.
    #[inline]
    pub fn new(origin: Point2<F>, direction: Vec2<F>) -> Self {
        Self { origin, direction }
    }

    /// Creates a line passing through two points.
    #[inline]
    pub fn from_points(p1: Point2<F>, p2: Point2<F>) -> Self {
        Self {
            origin: p1,
            direction: p2 - p1,
        }
    }

    /// Creates a line from a segment (extending it infinitely).
    #[inline]
    pub fn from_segment(segment: &Segment2<F>) -> Self {
        Self {
            origin: segment.start,
            direction: segment.direction(),
        }
    }

    /// Creates a line from a ray (extending it in both directions).
    #[inline]
    pub fn from_ray(ray: &Ray2<F>) -> Self {
        Self {
            origin: ray.origin,
            direction: ray.direction,
        }
    }

    /// Creates a horizontal line at the given y-coordinate.
    #[inline]
    pub fn horizontal(y: F) -> Self {
        Self {
            origin: Point2::new(F::zero(), y),
            direction: Vec2::new(F::one(), F::zero()),
        }
    }

    /// Creates a vertical line at the given x-coordinate.
    #[inline]
    pub fn vertical(x: F) -> Self {
        Self {
            origin: Point2::new(x, F::zero()),
            direction: Vec2::new(F::zero(), F::one()),
        }
    }

    /// Creates a line from implicit form coefficients: ax + by + c = 0.
    ///
    /// The coefficients don't need to be normalized.
    pub fn from_implicit(a: F, b: F, c: F) -> Option<Self> {
        let len_sq = a * a + b * b;
        if len_sq < F::epsilon() {
            return None;
        }

        // Normal vector is (a, b), direction is perpendicular: (-b, a)
        // Find a point on the line: if a != 0, use (−c/a, 0); else use (0, −c/b)
        let origin = if a.abs() > b.abs() {
            Point2::new(-c / a, F::zero())
        } else {
            Point2::new(F::zero(), -c / b)
        };

        Some(Self {
            origin,
            direction: Vec2::new(-b, a),
        })
    }

    /// Returns the implicit form coefficients (a, b, c) where ax + by + c = 0.
    ///
    /// The coefficients are normalized so that a² + b² = 1.
    pub fn to_implicit(&self) -> (F, F, F) {
        // Normal is perpendicular to direction
        let normal = Vec2::new(-self.direction.y, self.direction.x);
        let len = normal.magnitude();

        if len < F::epsilon() {
            return (F::zero(), F::zero(), F::zero());
        }

        let a = normal.x / len;
        let b = normal.y / len;
        let c = -(a * self.origin.x + b * self.origin.y);

        (a, b, c)
    }

    /// Returns the normalized direction vector.
    #[inline]
    pub fn normalized_direction(&self) -> Option<Vec2<F>> {
        self.direction.normalize()
    }

    /// Returns the normal vector (perpendicular to direction, pointing "left").
    #[inline]
    pub fn normal(&self) -> Vec2<F> {
        Vec2::new(-self.direction.y, self.direction.x)
    }

    /// Returns the unit normal vector.
    #[inline]
    pub fn unit_normal(&self) -> Option<Vec2<F>> {
        self.normal().normalize()
    }

    /// Returns the point on the line at parameter t.
    ///
    /// - `t = 0` returns the origin
    /// - Positive t is in the direction of the direction vector
    /// - Negative t is in the opposite direction
    #[inline]
    pub fn point_at(&self, t: F) -> Point2<F> {
        Point2::new(
            self.origin.x + t * self.direction.x,
            self.origin.y + t * self.direction.y,
        )
    }

    /// Projects a point onto the line and returns the closest point.
    pub fn closest_point(&self, point: Point2<F>) -> Point2<F> {
        let to_point = point - self.origin;
        let len_sq = self.direction.magnitude_squared();

        if len_sq < F::epsilon() {
            return self.origin;
        }

        let t = to_point.dot(self.direction) / len_sq;
        self.point_at(t)
    }

    /// Projects a point onto the line and returns (closest_point, parameter_t).
    pub fn project(&self, point: Point2<F>) -> (Point2<F>, F) {
        let to_point = point - self.origin;
        let len_sq = self.direction.magnitude_squared();

        if len_sq < F::epsilon() {
            return (self.origin, F::zero());
        }

        let t = to_point.dot(self.direction) / len_sq;
        (self.point_at(t), t)
    }

    /// Returns the signed distance from a point to the line.
    ///
    /// Positive on one side, negative on the other (based on normal direction).
    pub fn signed_distance(&self, point: Point2<F>) -> F {
        let (a, b, c) = self.to_implicit();
        a * point.x + b * point.y + c
    }

    /// Returns the unsigned distance from a point to the line.
    #[inline]
    pub fn distance(&self, point: Point2<F>) -> F {
        self.signed_distance(point).abs()
    }

    /// Returns which side of the line a point is on.
    ///
    /// Returns positive, negative, or zero (on the line).
    #[inline]
    pub fn side(&self, point: Point2<F>) -> F {
        self.signed_distance(point)
    }

    /// Checks if a point is on the line (within tolerance).
    #[inline]
    pub fn contains(&self, point: Point2<F>, tolerance: F) -> bool {
        self.distance(point) <= tolerance
    }

    /// Intersects this line with another line.
    ///
    /// Returns `Some((point, t_self, t_other))` if they intersect,
    /// where the parameters give the position along each line.
    ///
    /// Returns `None` if lines are parallel.
    pub fn intersect_line(&self, other: &Line2<F>) -> Option<(Point2<F>, F, F)> {
        let cross = self.direction.x * other.direction.y - self.direction.y * other.direction.x;

        if cross.abs() < F::epsilon() {
            return None;
        }

        let delta = other.origin - self.origin;
        let t_self = (delta.x * other.direction.y - delta.y * other.direction.x) / cross;
        let t_other = (delta.x * self.direction.y - delta.y * self.direction.x) / cross;

        Some((self.point_at(t_self), t_self, t_other))
    }

    /// Intersects this line with a ray.
    ///
    /// Returns the intersection point and parameters if the ray hits the line.
    /// The ray parameter must be >= 0 for a valid intersection.
    pub fn intersect_ray(&self, ray: &Ray2<F>) -> Option<(Point2<F>, F, F)> {
        let cross = self.direction.x * ray.direction.y - self.direction.y * ray.direction.x;

        if cross.abs() < F::epsilon() {
            return None;
        }

        let delta = ray.origin - self.origin;
        let t_line = (delta.x * ray.direction.y - delta.y * ray.direction.x) / cross;
        let t_ray = (delta.x * self.direction.y - delta.y * self.direction.x) / cross;

        if t_ray >= F::zero() {
            Some((self.point_at(t_line), t_line, t_ray))
        } else {
            None
        }
    }

    /// Intersects this line with a segment.
    ///
    /// Returns the intersection point and parameters if they intersect.
    /// The segment parameter must be in [0, 1] for a valid intersection.
    pub fn intersect_segment(&self, segment: &Segment2<F>) -> Option<(Point2<F>, F, F)> {
        let seg_dir = segment.direction();
        let cross = self.direction.x * seg_dir.y - self.direction.y * seg_dir.x;

        if cross.abs() < F::epsilon() {
            return None;
        }

        let delta = segment.start - self.origin;
        let t_line = (delta.x * seg_dir.y - delta.y * seg_dir.x) / cross;
        let t_seg = (delta.x * self.direction.y - delta.y * self.direction.x) / cross;

        if t_seg >= F::zero() && t_seg <= F::one() {
            Some((self.point_at(t_line), t_line, t_seg))
        } else {
            None
        }
    }

    /// Intersects this line with a circle.
    ///
    /// Returns 0, 1, or 2 intersection points with their line parameters.
    pub fn intersect_circle(&self, circle: &Circle2<F>) -> Vec<(Point2<F>, F)> {
        let oc = self.origin - circle.center;

        let a = self.direction.dot(self.direction);
        let b = F::from(2.0).unwrap() * oc.dot(self.direction);
        let c = oc.dot(oc) - circle.radius * circle.radius;

        let discriminant = b * b - F::from(4.0).unwrap() * a * c;

        if discriminant < F::zero() {
            return Vec::new();
        }

        let sqrt_disc = discriminant.sqrt();
        let two_a = F::from(2.0).unwrap() * a;

        let t1 = (-b - sqrt_disc) / two_a;
        let t2 = (-b + sqrt_disc) / two_a;

        if discriminant < F::epsilon() {
            // Tangent - one intersection
            vec![(self.point_at(t1), t1)]
        } else {
            // Two intersections
            vec![(self.point_at(t1), t1), (self.point_at(t2), t2)]
        }
    }

    /// Checks if this line is parallel to another line.
    #[inline]
    pub fn is_parallel(&self, other: &Line2<F>) -> bool {
        let cross = self.direction.x * other.direction.y - self.direction.y * other.direction.x;
        cross.abs() < F::epsilon()
    }

    /// Checks if this line is perpendicular to another line.
    #[inline]
    pub fn is_perpendicular(&self, other: &Line2<F>) -> bool {
        let dot = self.direction.dot(other.direction);
        dot.abs() < F::epsilon()
    }

    /// Returns the angle between this line and another line (in radians).
    ///
    /// Returns a value in [0, π/2] (lines don't have a preferred direction).
    pub fn angle_to(&self, other: &Line2<F>) -> F {
        let len1 = self.direction.magnitude();
        let len2 = other.direction.magnitude();

        if len1 < F::epsilon() || len2 < F::epsilon() {
            return F::zero();
        }

        let cos_angle = self.direction.dot(other.direction).abs() / (len1 * len2);
        cos_angle.min(F::one()).acos()
    }

    /// Returns a line perpendicular to this one, passing through the given point.
    pub fn perpendicular_at(&self, point: Point2<F>) -> Self {
        Self {
            origin: point,
            direction: self.normal(),
        }
    }

    /// Returns a parallel line at the given signed distance.
    ///
    /// Positive distance is in the direction of the normal.
    pub fn parallel_at_distance(&self, distance: F) -> Option<Self> {
        let unit_normal = self.unit_normal()?;
        Some(Self {
            origin: Point2::new(
                self.origin.x + distance * unit_normal.x,
                self.origin.y + distance * unit_normal.y,
            ),
            direction: self.direction,
        })
    }

    /// Returns a line translated by the given offset.
    #[inline]
    pub fn translated(&self, offset: Vec2<F>) -> Self {
        Self {
            origin: self.origin + offset,
            direction: self.direction,
        }
    }

    /// Returns a line rotated by the given angle (in radians) around the origin point.
    pub fn rotated(&self, angle: F) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Self {
            origin: self.origin,
            direction: Vec2::new(
                self.direction.x * cos_a - self.direction.y * sin_a,
                self.direction.x * sin_a + self.direction.y * cos_a,
            ),
        }
    }

    /// Returns a line rotated around a specific pivot point.
    pub fn rotated_around(&self, angle: F, pivot: Point2<F>) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Rotate origin around pivot
        let dx = self.origin.x - pivot.x;
        let dy = self.origin.y - pivot.y;
        let new_origin = Point2::new(
            pivot.x + dx * cos_a - dy * sin_a,
            pivot.y + dx * sin_a + dy * cos_a,
        );

        // Rotate direction
        let new_direction = Vec2::new(
            self.direction.x * cos_a - self.direction.y * sin_a,
            self.direction.x * sin_a + self.direction.y * cos_a,
        );

        Self {
            origin: new_origin,
            direction: new_direction,
        }
    }

    /// Reflects a point across this line.
    pub fn reflect_point(&self, point: Point2<F>) -> Point2<F> {
        let closest = self.closest_point(point);
        Point2::new(
            F::from(2.0).unwrap() * closest.x - point.x,
            F::from(2.0).unwrap() * closest.y - point.y,
        )
    }
}

impl<F: Float> Default for Line2<F> {
    fn default() -> Self {
        Self::horizontal(F::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new() {
        let line: Line2<f64> = Line2::new(Point2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(line.origin.x, 1.0);
        assert_eq!(line.origin.y, 2.0);
        assert_eq!(line.direction.x, 3.0);
        assert_eq!(line.direction.y, 4.0);
    }

    #[test]
    fn test_from_points() {
        let line: Line2<f64> = Line2::from_points(Point2::new(1.0, 1.0), Point2::new(4.0, 5.0));
        assert_eq!(line.direction.x, 3.0);
        assert_eq!(line.direction.y, 4.0);
    }

    #[test]
    fn test_horizontal() {
        let line: Line2<f64> = Line2::horizontal(5.0);
        assert_eq!(line.origin.y, 5.0);
        assert_eq!(line.direction.y, 0.0);
    }

    #[test]
    fn test_vertical() {
        let line: Line2<f64> = Line2::vertical(5.0);
        assert_eq!(line.origin.x, 5.0);
        assert_eq!(line.direction.x, 0.0);
    }

    #[test]
    fn test_implicit_form() {
        // Line y = 2 (horizontal) => 0x + 1y - 2 = 0
        let line: Line2<f64> = Line2::horizontal(2.0);
        let (a, b, c) = line.to_implicit();

        // Should be approximately (0, 1, -2) or (0, -1, 2)
        assert_relative_eq!(a.abs(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(b.abs(), 1.0, epsilon = 1e-10);
        assert_relative_eq!((a * 0.0 + b * 2.0 + c).abs(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_implicit() {
        // x + y - 1 = 0 (diagonal line through (1,0) and (0,1))
        let line = Line2::<f64>::from_implicit(1.0, 1.0, -1.0).unwrap();

        // Check that (1,0) and (0,1) are on the line
        assert!(line.distance(Point2::new(1.0, 0.0)) < 1e-10);
        assert!(line.distance(Point2::new(0.0, 1.0)) < 1e-10);
    }

    #[test]
    fn test_closest_point() {
        let line: Line2<f64> = Line2::horizontal(0.0);

        let closest = line.closest_point(Point2::new(5.0, 3.0));
        assert_relative_eq!(closest.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(closest.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_project() {
        let line: Line2<f64> = Line2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        let (point, t) = line.project(Point2::new(5.0, 3.0));
        assert_relative_eq!(point.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(point.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t, 5.0, epsilon = 1e-10);

        // Point behind origin still projects correctly (unlike ray)
        let (point, t) = line.project(Point2::new(-5.0, 3.0));
        assert_relative_eq!(point.x, -5.0, epsilon = 1e-10);
        assert_relative_eq!(t, -5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_signed_distance() {
        let line: Line2<f64> = Line2::horizontal(0.0);

        assert_relative_eq!(
            line.signed_distance(Point2::new(0.0, 5.0)),
            5.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            line.signed_distance(Point2::new(0.0, -5.0)),
            -5.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            line.signed_distance(Point2::new(100.0, 0.0)),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_distance() {
        let line: Line2<f64> = Line2::horizontal(0.0);

        assert_relative_eq!(line.distance(Point2::new(0.0, 5.0)), 5.0, epsilon = 1e-10);
        assert_relative_eq!(line.distance(Point2::new(0.0, -5.0)), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_contains() {
        let line: Line2<f64> = Line2::horizontal(0.0);

        assert!(line.contains(Point2::new(100.0, 0.0), 1e-10));
        assert!(line.contains(Point2::new(-100.0, 0.0), 1e-10));
        assert!(!line.contains(Point2::new(0.0, 1.0), 1e-10));
    }

    #[test]
    fn test_intersect_line() {
        let line1: Line2<f64> = Line2::horizontal(0.0);
        let line2: Line2<f64> = Line2::vertical(5.0);

        let result = line1.intersect_line(&line2);
        assert!(result.is_some());

        let (point, _, _) = result.unwrap();
        assert_relative_eq!(point.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(point.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_line_parallel() {
        let line1: Line2<f64> = Line2::horizontal(0.0);
        let line2: Line2<f64> = Line2::horizontal(5.0);

        assert!(line1.intersect_line(&line2).is_none());
    }

    #[test]
    fn test_intersect_ray() {
        let line: Line2<f64> = Line2::vertical(5.0);
        let ray = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        let result = line.intersect_ray(&ray);
        assert!(result.is_some());

        let (point, _, t_ray) = result.unwrap();
        assert_relative_eq!(point.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(t_ray, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_ray_behind() {
        let line: Line2<f64> = Line2::vertical(-5.0);
        let ray = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        // Ray points away from line
        assert!(line.intersect_ray(&ray).is_none());
    }

    #[test]
    fn test_intersect_segment() {
        let line: Line2<f64> = Line2::horizontal(0.5);
        let seg = Segment2::new(Point2::new(0.0, 0.0), Point2::new(0.0, 1.0));

        let result = line.intersect_segment(&seg);
        assert!(result.is_some());

        let (point, _, t_seg) = result.unwrap();
        assert_relative_eq!(point.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(point.y, 0.5, epsilon = 1e-10);
        assert_relative_eq!(t_seg, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_segment_miss() {
        let line: Line2<f64> = Line2::horizontal(5.0);
        let seg = Segment2::new(Point2::new(0.0, 0.0), Point2::new(0.0, 1.0));

        assert!(line.intersect_segment(&seg).is_none());
    }

    #[test]
    fn test_intersect_circle() {
        let line: Line2<f64> = Line2::horizontal(0.0);
        let circle = Circle2::new(Point2::origin(), 2.0);

        let hits = line.intersect_circle(&circle);
        assert_eq!(hits.len(), 2);

        assert_relative_eq!(hits[0].0.x, -2.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_circle_tangent() {
        let line: Line2<f64> = Line2::horizontal(2.0);
        let circle = Circle2::new(Point2::origin(), 2.0);

        let hits = line.intersect_circle(&circle);
        assert_eq!(hits.len(), 1);

        assert_relative_eq!(hits[0].0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(hits[0].0.y, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_circle_miss() {
        let line: Line2<f64> = Line2::horizontal(5.0);
        let circle = Circle2::new(Point2::origin(), 2.0);

        let hits = line.intersect_circle(&circle);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_is_parallel() {
        let line1: Line2<f64> = Line2::horizontal(0.0);
        let line2: Line2<f64> = Line2::horizontal(5.0);
        let line3: Line2<f64> = Line2::vertical(0.0);

        assert!(line1.is_parallel(&line2));
        assert!(!line1.is_parallel(&line3));
    }

    #[test]
    fn test_is_perpendicular() {
        let line1: Line2<f64> = Line2::horizontal(0.0);
        let line2: Line2<f64> = Line2::vertical(0.0);
        let line3: Line2<f64> = Line2::horizontal(5.0);

        assert!(line1.is_perpendicular(&line2));
        assert!(!line1.is_perpendicular(&line3));
    }

    #[test]
    fn test_angle_to() {
        let line1: Line2<f64> = Line2::horizontal(0.0);
        let line2: Line2<f64> = Line2::vertical(0.0);

        let angle = line1.angle_to(&line2);
        assert_relative_eq!(angle, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);

        // Angle to self should be 0
        assert_relative_eq!(line1.angle_to(&line1), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_perpendicular_at() {
        let line: Line2<f64> = Line2::horizontal(0.0);
        let perp = line.perpendicular_at(Point2::new(5.0, 0.0));

        assert!(line.is_perpendicular(&perp));
        assert!(perp.contains(Point2::new(5.0, 0.0), 1e-10));
    }

    #[test]
    fn test_parallel_at_distance() {
        let line: Line2<f64> = Line2::horizontal(0.0);
        let parallel = line.parallel_at_distance(3.0).unwrap();

        assert!(line.is_parallel(&parallel));
        assert_relative_eq!(line.origin.y + 3.0, parallel.origin.y, epsilon = 1e-10);
    }

    #[test]
    fn test_translated() {
        let line: Line2<f64> = Line2::horizontal(0.0);
        let translated = line.translated(Vec2::new(1.0, 5.0));

        assert_eq!(translated.origin.y, 5.0);
        assert!(line.is_parallel(&translated));
    }

    #[test]
    fn test_rotated() {
        let line: Line2<f64> = Line2::horizontal(0.0);
        let rotated = line.rotated(std::f64::consts::FRAC_PI_2);

        assert_relative_eq!(rotated.direction.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated.direction.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reflect_point() {
        let line: Line2<f64> = Line2::horizontal(0.0);

        let reflected = line.reflect_point(Point2::new(3.0, 5.0));
        assert_relative_eq!(reflected.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(reflected.y, -5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let line: Line2<f32> = Line2::horizontal(0.0);
        assert!(line.contains(Point2::new(100.0, 0.0), 1e-6));
    }

    #[test]
    fn test_diagonal_line() {
        // y = x line
        let line: Line2<f64> = Line2::from_points(Point2::origin(), Point2::new(1.0, 1.0));

        // Distance from (1, 0) to y=x should be 1/sqrt(2)
        let expected_dist = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(
            line.distance(Point2::new(1.0, 0.0)),
            expected_dist,
            epsilon = 1e-10
        );

        // Point on the line
        assert!(line.contains(Point2::new(5.0, 5.0), 1e-10));
        assert!(line.contains(Point2::new(-3.0, -3.0), 1e-10));
    }
}
