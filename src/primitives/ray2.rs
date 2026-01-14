//! 2D ray type.

use super::{Circle2, Point2, Segment2, Vec2};
use num_traits::Float;

/// A 2D ray defined by an origin point and direction.
///
/// A ray extends infinitely from its origin in the direction specified.
/// The direction is stored as-is (not necessarily normalized).
///
/// # Example
///
/// ```
/// use approxum::primitives::{Ray2, Point2, Vec2, Segment2};
///
/// let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));
/// let segment = Segment2::new(Point2::new(5.0, -1.0), Point2::new(5.0, 1.0));
///
/// let hit = ray.intersect_segment(&segment);
/// assert!(hit.is_some());
/// assert_eq!(hit.unwrap().0.x, 5.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray2<F> {
    /// Origin point of the ray
    pub origin: Point2<F>,
    /// Direction vector (not necessarily normalized)
    pub direction: Vec2<F>,
}

impl<F: Float> Ray2<F> {
    /// Creates a new ray from origin and direction.
    #[inline]
    pub fn new(origin: Point2<F>, direction: Vec2<F>) -> Self {
        Self { origin, direction }
    }

    /// Creates a ray from an origin point through a target point.
    #[inline]
    pub fn from_points(origin: Point2<F>, through: Point2<F>) -> Self {
        Self {
            origin,
            direction: through - origin,
        }
    }

    /// Creates a ray with a normalized direction vector.
    ///
    /// Returns `None` if the direction vector is zero.
    pub fn normalized(origin: Point2<F>, direction: Vec2<F>) -> Option<Self> {
        direction.normalize().map(|d| Self {
            origin,
            direction: d,
        })
    }

    /// Returns the point along the ray at parameter t.
    ///
    /// - `t = 0` returns the origin
    /// - `t > 0` returns points along the ray direction
    /// - `t < 0` returns points behind the origin (not on the ray)
    #[inline]
    pub fn point_at(&self, t: F) -> Point2<F> {
        Point2::new(
            self.origin.x + t * self.direction.x,
            self.origin.y + t * self.direction.y,
        )
    }

    /// Returns the normalized direction vector.
    ///
    /// Returns `None` if the direction is zero.
    #[inline]
    pub fn normalized_direction(&self) -> Option<Vec2<F>> {
        self.direction.normalize()
    }

    /// Returns the closest point on the ray to the given point.
    ///
    /// Returns the point and the parameter t (clamped to >= 0).
    pub fn closest_point(&self, point: Point2<F>) -> (Point2<F>, F) {
        let to_point = point - self.origin;
        let len_sq = self.direction.magnitude_squared();

        if len_sq < F::epsilon() {
            return (self.origin, F::zero());
        }

        let t = to_point.dot(self.direction) / len_sq;
        let t_clamped = t.max(F::zero()); // Ray only goes forward

        (self.point_at(t_clamped), t_clamped)
    }

    /// Returns the distance from the ray to the given point.
    #[inline]
    pub fn distance_to_point(&self, point: Point2<F>) -> F {
        let (closest, _) = self.closest_point(point);
        point.distance(closest)
    }

    /// Returns the squared distance from the ray to the given point.
    #[inline]
    pub fn distance_squared_to_point(&self, point: Point2<F>) -> F {
        let (closest, _) = self.closest_point(point);
        point.distance_squared(closest)
    }

    /// Intersects this ray with a line segment.
    ///
    /// Returns `Some((point, t_ray, t_segment))` if they intersect, where:
    /// - `point` is the intersection point
    /// - `t_ray` is the parameter along the ray (>= 0)
    /// - `t_segment` is the parameter along the segment (in [0, 1])
    ///
    /// Returns `None` if no intersection or ray is parallel to segment.
    pub fn intersect_segment(&self, segment: &Segment2<F>) -> Option<(Point2<F>, F, F)> {
        let seg_dir = segment.direction();

        // Cross product of directions (2D "cross" gives scalar)
        let cross = self.direction.x * seg_dir.y - self.direction.y * seg_dir.x;

        // Parallel check
        if cross.abs() < F::epsilon() {
            return None;
        }

        let delta = segment.start - self.origin;
        let t_ray = (delta.x * seg_dir.y - delta.y * seg_dir.x) / cross;
        let t_seg = (delta.x * self.direction.y - delta.y * self.direction.x) / cross;

        // Ray only intersects for t >= 0, segment for t in [0, 1]
        if t_ray >= F::zero() && t_seg >= F::zero() && t_seg <= F::one() {
            Some((self.point_at(t_ray), t_ray, t_seg))
        } else {
            None
        }
    }

    /// Intersects this ray with a circle.
    ///
    /// Returns intersection points along with their ray parameters.
    /// Points are ordered by increasing t (closest first).
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

        let mut results = Vec::new();

        // Only include intersections with t >= 0 (on the ray)
        if t1 >= F::zero() {
            results.push((self.point_at(t1), t1));
        }

        if t2 >= F::zero() && discriminant > F::epsilon() {
            results.push((self.point_at(t2), t2));
        }

        results
    }

    /// Intersects this ray with another ray.
    ///
    /// Returns `Some((point, t_self, t_other))` if they intersect at a single point
    /// where both t values are >= 0.
    ///
    /// Returns `None` if rays are parallel or intersection is behind either origin.
    pub fn intersect_ray(&self, other: &Ray2<F>) -> Option<(Point2<F>, F, F)> {
        let cross = self.direction.x * other.direction.y - self.direction.y * other.direction.x;

        if cross.abs() < F::epsilon() {
            return None;
        }

        let delta = other.origin - self.origin;
        let t_self = (delta.x * other.direction.y - delta.y * other.direction.x) / cross;
        let t_other = (delta.x * self.direction.y - delta.y * self.direction.x) / cross;

        if t_self >= F::zero() && t_other >= F::zero() {
            Some((self.point_at(t_self), t_self, t_other))
        } else {
            None
        }
    }

    /// Intersects this ray with a polygon (given as a slice of vertices).
    ///
    /// Returns all intersection points sorted by distance from origin,
    /// along with their ray parameters and the edge index.
    pub fn intersect_polygon(&self, vertices: &[Point2<F>]) -> Vec<(Point2<F>, F, usize)> {
        let n = vertices.len();
        if n < 3 {
            return Vec::new();
        }

        let mut hits = Vec::new();

        for i in 0..n {
            let segment = Segment2::new(vertices[i], vertices[(i + 1) % n]);

            if let Some((point, t, _)) = self.intersect_segment(&segment) {
                hits.push((point, t, i));
            }
        }

        // Sort by distance (t parameter)
        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        hits
    }

    /// Casts the ray against a polygon and returns the first hit.
    ///
    /// Returns `Some((point, t, edge_index))` for the closest intersection,
    /// or `None` if the ray doesn't hit the polygon.
    pub fn cast_polygon(&self, vertices: &[Point2<F>]) -> Option<(Point2<F>, F, usize)> {
        self.intersect_polygon(vertices).into_iter().next()
    }

    /// Checks if the ray intersects a segment.
    #[inline]
    pub fn hits_segment(&self, segment: &Segment2<F>) -> bool {
        self.intersect_segment(segment).is_some()
    }

    /// Checks if the ray intersects a circle.
    #[inline]
    pub fn hits_circle(&self, circle: &Circle2<F>) -> bool {
        !self.intersect_circle(circle).is_empty()
    }

    /// Returns a ray pointing in the opposite direction from the same origin.
    #[inline]
    pub fn reversed(&self) -> Self {
        Self {
            origin: self.origin,
            direction: Vec2::new(-self.direction.x, -self.direction.y),
        }
    }

    /// Returns a ray translated by the given offset.
    #[inline]
    pub fn translated(&self, offset: Vec2<F>) -> Self {
        Self {
            origin: self.origin + offset,
            direction: self.direction,
        }
    }

    /// Returns a ray rotated by the given angle (in radians) around its origin.
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
}

impl<F: Float> Default for Ray2<F> {
    fn default() -> Self {
        Self {
            origin: Point2::origin(),
            direction: Vec2::new(F::one(), F::zero()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new() {
        let ray: Ray2<f64> = Ray2::new(Point2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        assert_eq!(ray.origin.x, 1.0);
        assert_eq!(ray.origin.y, 2.0);
        assert_eq!(ray.direction.x, 3.0);
        assert_eq!(ray.direction.y, 4.0);
    }

    #[test]
    fn test_from_points() {
        let ray: Ray2<f64> = Ray2::from_points(Point2::new(1.0, 1.0), Point2::new(4.0, 5.0));
        assert_eq!(ray.origin.x, 1.0);
        assert_eq!(ray.origin.y, 1.0);
        assert_eq!(ray.direction.x, 3.0);
        assert_eq!(ray.direction.y, 4.0);
    }

    #[test]
    fn test_point_at() {
        let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        let p0 = ray.point_at(0.0);
        assert_eq!(p0.x, 0.0);
        assert_eq!(p0.y, 0.0);

        let p5 = ray.point_at(5.0);
        assert_eq!(p5.x, 5.0);
        assert_eq!(p5.y, 0.0);
    }

    #[test]
    fn test_closest_point_on_ray() {
        let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        // Point above the ray
        let (closest, t) = ray.closest_point(Point2::new(5.0, 3.0));
        assert_relative_eq!(closest.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(closest.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t, 5.0, epsilon = 1e-10);

        // Point behind the origin
        let (closest, t) = ray.closest_point(Point2::new(-5.0, 3.0));
        assert_eq!(closest.x, 0.0); // Clamped to origin
        assert_eq!(closest.y, 0.0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_distance_to_point() {
        let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        assert_relative_eq!(
            ray.distance_to_point(Point2::new(5.0, 3.0)),
            3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            ray.distance_to_point(Point2::new(5.0, 0.0)),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_intersect_segment_hit() {
        let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));
        let seg = Segment2::new(Point2::new(5.0, -2.0), Point2::new(5.0, 2.0));

        let result = ray.intersect_segment(&seg);
        assert!(result.is_some());

        let (point, t_ray, t_seg) = result.unwrap();
        assert_relative_eq!(point.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(point.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t_ray, 5.0, epsilon = 1e-10);
        assert_relative_eq!(t_seg, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_segment_miss() {
        let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        // Segment behind the ray
        let seg_behind = Segment2::new(Point2::new(-5.0, -1.0), Point2::new(-5.0, 1.0));
        assert!(ray.intersect_segment(&seg_behind).is_none());

        // Segment to the side
        let seg_side = Segment2::new(Point2::new(5.0, 5.0), Point2::new(5.0, 10.0));
        assert!(ray.intersect_segment(&seg_side).is_none());

        // Parallel segment
        let seg_parallel = Segment2::new(Point2::new(0.0, 1.0), Point2::new(10.0, 1.0));
        assert!(ray.intersect_segment(&seg_parallel).is_none());
    }

    #[test]
    fn test_intersect_circle_two_hits() {
        let ray: Ray2<f64> = Ray2::new(Point2::new(-5.0, 0.0), Vec2::new(1.0, 0.0));
        let circle = Circle2::new(Point2::origin(), 2.0);

        let hits = ray.intersect_circle(&circle);
        assert_eq!(hits.len(), 2);

        // First hit (entering)
        assert_relative_eq!(hits[0].0.x, -2.0, epsilon = 1e-10);
        assert_relative_eq!(hits[0].1, 3.0, epsilon = 1e-10);

        // Second hit (exiting)
        assert_relative_eq!(hits[1].0.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].1, 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_circle_one_hit() {
        // Ray tangent to circle
        let ray: Ray2<f64> = Ray2::new(Point2::new(-5.0, 2.0), Vec2::new(1.0, 0.0));
        let circle = Circle2::new(Point2::origin(), 2.0);

        let hits = ray.intersect_circle(&circle);
        assert_eq!(hits.len(), 1);
        assert_relative_eq!(hits[0].0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(hits[0].0.y, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_circle_miss() {
        let ray: Ray2<f64> = Ray2::new(Point2::new(-5.0, 5.0), Vec2::new(1.0, 0.0));
        let circle = Circle2::new(Point2::origin(), 2.0);

        let hits = ray.intersect_circle(&circle);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_intersect_circle_origin_inside() {
        // Ray starts inside circle
        let ray: Ray2<f64> = Ray2::new(Point2::new(0.5, 0.0), Vec2::new(1.0, 0.0));
        let circle = Circle2::new(Point2::origin(), 2.0);

        let hits = ray.intersect_circle(&circle);
        assert_eq!(hits.len(), 1); // Only exit point
        assert_relative_eq!(hits[0].0.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_ray() {
        let ray1: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 1.0));
        let ray2: Ray2<f64> = Ray2::new(Point2::new(0.0, 2.0), Vec2::new(1.0, -1.0));

        let result = ray1.intersect_ray(&ray2);
        assert!(result.is_some());

        let (point, t1, t2) = result.unwrap();
        assert_relative_eq!(point.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(point.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t1, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t2, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_ray_parallel() {
        let ray1: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));
        let ray2: Ray2<f64> = Ray2::new(Point2::new(0.0, 1.0), Vec2::new(1.0, 0.0));

        assert!(ray1.intersect_ray(&ray2).is_none());
    }

    #[test]
    fn test_intersect_ray_behind() {
        // Rays pointing away from each other
        let ray1: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));
        let ray2: Ray2<f64> = Ray2::new(Point2::new(-1.0, 1.0), Vec2::new(0.0, 1.0));

        assert!(ray1.intersect_ray(&ray2).is_none());
    }

    #[test]
    fn test_intersect_polygon() {
        let ray: Ray2<f64> = Ray2::new(Point2::new(-5.0, 0.5), Vec2::new(1.0, 0.0));
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];

        let hits = ray.intersect_polygon(&square);
        assert_eq!(hits.len(), 2);

        // Entry point
        assert_relative_eq!(hits[0].0.x, 0.0, epsilon = 1e-10);
        assert_eq!(hits[0].2, 3); // Left edge (edge from vertex 3 to 0)

        // Exit point
        assert_relative_eq!(hits[1].0.x, 2.0, epsilon = 1e-10);
        assert_eq!(hits[1].2, 1); // Right edge
    }

    #[test]
    fn test_cast_polygon() {
        let ray: Ray2<f64> = Ray2::new(Point2::new(-5.0, 0.5), Vec2::new(1.0, 0.0));
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];

        let hit = ray.cast_polygon(&square);
        assert!(hit.is_some());

        let (point, _, edge) = hit.unwrap();
        assert_relative_eq!(point.x, 0.0, epsilon = 1e-10);
        assert_eq!(edge, 3);
    }

    #[test]
    fn test_hits_segment() {
        let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        let seg_hit = Segment2::new(Point2::new(5.0, -1.0), Point2::new(5.0, 1.0));
        assert!(ray.hits_segment(&seg_hit));

        let seg_miss = Segment2::new(Point2::new(5.0, 5.0), Point2::new(5.0, 10.0));
        assert!(!ray.hits_segment(&seg_miss));
    }

    #[test]
    fn test_hits_circle() {
        let ray: Ray2<f64> = Ray2::new(Point2::new(-5.0, 0.0), Vec2::new(1.0, 0.0));

        let circle_hit = Circle2::new(Point2::origin(), 2.0);
        assert!(ray.hits_circle(&circle_hit));

        let circle_miss = Circle2::new(Point2::new(0.0, 10.0), 2.0);
        assert!(!ray.hits_circle(&circle_miss));
    }

    #[test]
    fn test_reversed() {
        let ray: Ray2<f64> = Ray2::new(Point2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let reversed = ray.reversed();

        assert_eq!(reversed.origin, ray.origin);
        assert_eq!(reversed.direction.x, -3.0);
        assert_eq!(reversed.direction.y, -4.0);
    }

    #[test]
    fn test_translated() {
        let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));
        let translated = ray.translated(Vec2::new(5.0, 3.0));

        assert_eq!(translated.origin.x, 5.0);
        assert_eq!(translated.origin.y, 3.0);
        assert_eq!(translated.direction, ray.direction);
    }

    #[test]
    fn test_rotated() {
        let ray: Ray2<f64> = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));
        let rotated = ray.rotated(std::f64::consts::FRAC_PI_2);

        assert_eq!(rotated.origin, ray.origin);
        assert_relative_eq!(rotated.direction.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated.direction.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let ray: Ray2<f32> = Ray2::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let circle = Circle2::new(Point2::new(5.0, 0.0), 1.0);

        assert!(ray.hits_circle(&circle));
    }

    #[test]
    fn test_diagonal_ray_through_square() {
        // Diagonal ray through a unit square (offset to avoid hitting vertices)
        // Ray from (-1, -0.9) in direction (1, 1):
        // - Enters left edge (x=0) at y=0.1
        // - Exits top edge (y=1) at x=0.9
        let ray: Ray2<f64> = Ray2::new(Point2::new(-1.0, -0.9), Vec2::new(1.0, 1.0));
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        let hits = ray.intersect_polygon(&square);
        assert_eq!(hits.len(), 2);

        // Entry: left edge (x=0, y=0.1)
        assert_relative_eq!(hits[0].0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(hits[0].0.y, 0.1, epsilon = 1e-10);

        // Exit: top edge (x=0.9, y=1)
        assert_relative_eq!(hits[1].0.x, 0.9, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.y, 1.0, epsilon = 1e-10);
    }
}
