//! 2D ellipse type.

use super::{Circle2, Line2, Point2, Ray2, Segment2, Vec2};
use num_traits::Float;

/// A 2D ellipse defined by center, semi-axes, and rotation.
///
/// The ellipse is parameterized by:
/// - `center`: the center point
/// - `semi_major`: half the length of the major axis (a)
/// - `semi_minor`: half the length of the minor axis (b)
/// - `rotation`: angle of the major axis from the x-axis (radians)
///
/// # Example
///
/// ```
/// use approxum::primitives::{Ellipse2, Point2};
///
/// // Axis-aligned ellipse with semi-major=2, semi-minor=1
/// let ellipse: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
/// assert!(ellipse.contains(Point2::new(1.0, 0.5)));
/// assert!(!ellipse.contains(Point2::new(2.5, 0.0)));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse2<F> {
    /// Center point of the ellipse
    pub center: Point2<F>,
    /// Semi-major axis length (half of major axis)
    pub semi_major: F,
    /// Semi-minor axis length (half of minor axis)
    pub semi_minor: F,
    /// Rotation angle in radians (major axis direction from x-axis)
    pub rotation: F,
}

impl<F: Float> Ellipse2<F> {
    /// Creates a new ellipse with the given parameters.
    #[inline]
    pub fn new(center: Point2<F>, semi_major: F, semi_minor: F, rotation: F) -> Self {
        Self {
            center,
            semi_major,
            semi_minor,
            rotation,
        }
    }

    /// Creates an axis-aligned ellipse (no rotation).
    #[inline]
    pub fn axis_aligned(center: Point2<F>, semi_major: F, semi_minor: F) -> Self {
        Self {
            center,
            semi_major,
            semi_minor,
            rotation: F::zero(),
        }
    }

    /// Creates an ellipse from a circle.
    #[inline]
    pub fn from_circle(center: Point2<F>, radius: F) -> Self {
        Self {
            center,
            semi_major: radius,
            semi_minor: radius,
            rotation: F::zero(),
        }
    }

    /// Returns true if this ellipse is actually a circle (semi_major == semi_minor).
    #[inline]
    pub fn is_circle(&self) -> bool {
        (self.semi_major - self.semi_minor).abs() < F::epsilon()
    }

    /// Returns the eccentricity of the ellipse (0 for circle, approaches 1 for very elongated).
    #[inline]
    pub fn eccentricity(&self) -> F {
        let a = self.semi_major;
        let b = self.semi_minor;
        ((a * a - b * b) / (a * a)).sqrt()
    }

    /// Returns the area of the ellipse.
    #[inline]
    pub fn area(&self) -> F {
        F::from(std::f64::consts::PI).unwrap() * self.semi_major * self.semi_minor
    }

    /// Returns an approximation of the circumference using Ramanujan's formula.
    ///
    /// This is accurate to within 0.01% for most ellipses.
    pub fn circumference(&self) -> F {
        let a = self.semi_major;
        let b = self.semi_minor;
        let h = (a - b) * (a - b) / ((a + b) * (a + b));
        let three = F::from(3.0).unwrap();
        let four = F::from(4.0).unwrap();
        let ten = F::from(10.0).unwrap();
        let pi = F::from(std::f64::consts::PI).unwrap();

        // Ramanujan's second approximation
        pi * (a + b) * (F::one() + three * h / (ten + (four - three * h).sqrt()))
    }

    /// Returns the unit vector along the major axis direction.
    #[inline]
    pub fn major_axis_direction(&self) -> Vec2<F> {
        Vec2::new(self.rotation.cos(), self.rotation.sin())
    }

    /// Returns the unit vector along the minor axis direction.
    #[inline]
    pub fn minor_axis_direction(&self) -> Vec2<F> {
        Vec2::new(-self.rotation.sin(), self.rotation.cos())
    }

    /// Transforms a point from world coordinates to the ellipse's local coordinates.
    ///
    /// In local coordinates, the ellipse is axis-aligned and centered at origin.
    #[allow(clippy::wrong_self_convention)]
    fn to_local(&self, point: Point2<F>) -> Point2<F> {
        let dx = point.x - self.center.x;
        let dy = point.y - self.center.y;
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        Point2::new(dx * cos_r + dy * sin_r, -dx * sin_r + dy * cos_r)
    }

    /// Transforms a point from local coordinates back to world coordinates.
    #[allow(clippy::wrong_self_convention)]
    fn from_local(&self, local: Point2<F>) -> Point2<F> {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        Point2::new(
            self.center.x + local.x * cos_r - local.y * sin_r,
            self.center.y + local.x * sin_r + local.y * cos_r,
        )
    }

    /// Checks if a point is inside the ellipse (including boundary).
    pub fn contains(&self, point: Point2<F>) -> bool {
        let local = self.to_local(point);
        let nx = local.x / self.semi_major;
        let ny = local.y / self.semi_minor;
        nx * nx + ny * ny <= F::one()
    }

    /// Checks if a point is strictly inside the ellipse (excluding boundary).
    pub fn contains_strict(&self, point: Point2<F>) -> bool {
        let local = self.to_local(point);
        let nx = local.x / self.semi_major;
        let ny = local.y / self.semi_minor;
        nx * nx + ny * ny < F::one()
    }

    /// Returns the point on the ellipse boundary at parameter t.
    ///
    /// The parameter t represents the angle in the parametric form:
    /// - t = 0: point at (center + semi_major * major_axis_direction)
    /// - t = π/2: point at (center + semi_minor * minor_axis_direction)
    pub fn point_at(&self, t: F) -> Point2<F> {
        let local = Point2::new(self.semi_major * t.cos(), self.semi_minor * t.sin());
        self.from_local(local)
    }

    /// Returns the tangent vector at parameter t.
    ///
    /// The tangent points in the direction of increasing t.
    pub fn tangent_at(&self, t: F) -> Vec2<F> {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();

        // Derivative in local coordinates
        let local_dx = -self.semi_major * t.sin();
        let local_dy = self.semi_minor * t.cos();

        // Rotate to world coordinates
        Vec2::new(
            local_dx * cos_r - local_dy * sin_r,
            local_dx * sin_r + local_dy * cos_r,
        )
    }

    /// Returns the normal vector at parameter t (pointing outward).
    ///
    /// Returns the zero vector if the tangent is degenerate.
    pub fn normal_at(&self, t: F) -> Vec2<F> {
        let tangent = self.tangent_at(t);
        Vec2::new(tangent.y, -tangent.x)
            .normalize()
            .unwrap_or_else(|| Vec2::new(F::one(), F::zero()))
    }

    /// Returns the axis-aligned bounding box as (min, max) points.
    pub fn bounding_box(&self) -> (Point2<F>, Point2<F>) {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();

        // Compute the extent along x and y axes
        let a = self.semi_major;
        let b = self.semi_minor;

        // Half-widths of the bounding box
        let hx = (a * a * cos_r * cos_r + b * b * sin_r * sin_r).sqrt();
        let hy = (a * a * sin_r * sin_r + b * b * cos_r * cos_r).sqrt();

        (
            Point2::new(self.center.x - hx, self.center.y - hy),
            Point2::new(self.center.x + hx, self.center.y + hy),
        )
    }

    /// Finds the approximate closest point on the ellipse to the given point.
    ///
    /// Uses Newton's method iteration. For points at the center,
    /// returns the point at t=0.
    pub fn nearest_point(&self, point: Point2<F>) -> Point2<F> {
        let local = self.to_local(point);
        let a = self.semi_major;
        let b = self.semi_minor;

        // Handle degenerate case (point at center)
        if local.x.abs() < F::epsilon() && local.y.abs() < F::epsilon() {
            return self.from_local(Point2::new(a, F::zero()));
        }

        // Initial guess: angle to the point
        let mut t = local.y.atan2(local.x);

        // Newton iteration to find closest point
        let max_iter = 10;
        for _ in 0..max_iter {
            let cos_t = t.cos();
            let sin_t = t.sin();

            // Point on ellipse
            let ex = a * cos_t;
            let ey = b * sin_t;

            // Vector from ellipse point to target
            let dx = local.x - ex;
            let dy = local.y - ey;

            // Tangent direction
            let tx = -a * sin_t;
            let ty = b * cos_t;

            // Project difference onto tangent
            let dot = dx * tx + dy * ty;
            let tangent_len_sq = tx * tx + ty * ty;

            if tangent_len_sq < F::epsilon() {
                break;
            }

            // Newton step
            let dt = dot / tangent_len_sq;

            if dt.abs() < F::from(1e-10).unwrap() {
                break;
            }

            t = t + dt;
        }

        self.from_local(Point2::new(a * t.cos(), b * t.sin()))
    }

    /// Returns the signed distance from a point to the ellipse boundary.
    ///
    /// Negative inside, positive outside. This is an approximation.
    pub fn signed_distance(&self, point: Point2<F>) -> F {
        let nearest = self.nearest_point(point);
        let dist = point.distance(nearest);

        if self.contains(point) {
            -dist
        } else {
            dist
        }
    }

    /// Converts the ellipse to a polygon with the given number of segments.
    pub fn to_polygon(&self, num_segments: usize) -> Vec<Point2<F>> {
        let n = num_segments.max(3);
        let tau = F::from(std::f64::consts::TAU).unwrap();

        (0..n)
            .map(|i| {
                let t = tau * F::from(i).unwrap() / F::from(n).unwrap();
                self.point_at(t)
            })
            .collect()
    }

    /// Returns the ellipse scaled by the given factor around its center.
    #[inline]
    pub fn scaled(&self, factor: F) -> Self {
        Self {
            center: self.center,
            semi_major: self.semi_major * factor,
            semi_minor: self.semi_minor * factor,
            rotation: self.rotation,
        }
    }

    /// Returns the ellipse translated by the given vector.
    #[inline]
    pub fn translated(&self, offset: Vec2<F>) -> Self {
        Self {
            center: self.center + offset,
            semi_major: self.semi_major,
            semi_minor: self.semi_minor,
            rotation: self.rotation,
        }
    }

    /// Returns the ellipse rotated by the given angle (in radians).
    #[inline]
    pub fn rotated(&self, angle: F) -> Self {
        Self {
            center: self.center,
            semi_major: self.semi_major,
            semi_minor: self.semi_minor,
            rotation: self.rotation + angle,
        }
    }

    /// Returns the foci of the ellipse.
    pub fn foci(&self) -> (Point2<F>, Point2<F>) {
        let c = (self.semi_major * self.semi_major - self.semi_minor * self.semi_minor).sqrt();
        let dir = self.major_axis_direction();
        (
            Point2::new(self.center.x - c * dir.x, self.center.y - c * dir.y),
            Point2::new(self.center.x + c * dir.x, self.center.y + c * dir.y),
        )
    }

    // ==================== Intersection Methods ====================

    /// Transforms a direction vector to local coordinates.
    fn dir_to_local(&self, dir: Vec2<F>) -> Vec2<F> {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        Vec2::new(
            dir.x * cos_r + dir.y * sin_r,
            -dir.x * sin_r + dir.y * cos_r,
        )
    }

    /// Intersects this ellipse with an infinite line.
    ///
    /// Returns 0, 1, or 2 intersection points with their line parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use approxum::primitives::{Ellipse2, Line2, Point2, Vec2};
    ///
    /// let ellipse: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
    /// let line = Line2::horizontal(0.0);
    ///
    /// let hits = ellipse.intersect_line(&line);
    /// assert_eq!(hits.len(), 2);
    /// ```
    pub fn intersect_line(&self, line: &Line2<F>) -> Vec<(Point2<F>, F)> {
        // Transform line to ellipse's local coordinate system
        let local_origin = self.to_local(line.origin);
        let local_dir = self.dir_to_local(line.direction);

        // Solve intersection with axis-aligned ellipse: (x/a)² + (y/b)² = 1
        // Line: P = origin + t * dir
        // Substitute: ((ox + t*dx)/a)² + ((oy + t*dy)/b)² = 1
        let a = self.semi_major;
        let b = self.semi_minor;

        let ox = local_origin.x;
        let oy = local_origin.y;
        let dx = local_dir.x;
        let dy = local_dir.y;

        // Expand to: At² + Bt + C = 0
        let coef_a = (dx * dx) / (a * a) + (dy * dy) / (b * b);
        let coef_b = F::from(2.0).unwrap() * ((ox * dx) / (a * a) + (oy * dy) / (b * b));
        let coef_c = (ox * ox) / (a * a) + (oy * oy) / (b * b) - F::one();

        self.solve_quadratic_intersection(coef_a, coef_b, coef_c, &local_dir, &local_origin)
    }

    /// Intersects this ellipse with a ray.
    ///
    /// Returns intersection points that are on the ray (t >= 0),
    /// sorted by distance from ray origin.
    pub fn intersect_ray(&self, ray: &Ray2<F>) -> Vec<(Point2<F>, F)> {
        let local_origin = self.to_local(ray.origin);
        let local_dir = self.dir_to_local(ray.direction);

        let a = self.semi_major;
        let b = self.semi_minor;

        let ox = local_origin.x;
        let oy = local_origin.y;
        let dx = local_dir.x;
        let dy = local_dir.y;

        let coef_a = (dx * dx) / (a * a) + (dy * dy) / (b * b);
        let coef_b = F::from(2.0).unwrap() * ((ox * dx) / (a * a) + (oy * dy) / (b * b));
        let coef_c = (ox * ox) / (a * a) + (oy * oy) / (b * b) - F::one();

        // Get all intersections then filter for t >= 0
        let all_hits =
            self.solve_quadratic_intersection(coef_a, coef_b, coef_c, &local_dir, &local_origin);
        all_hits
            .into_iter()
            .filter(|(_, t)| *t >= F::zero())
            .collect()
    }

    /// Intersects this ellipse with a line segment.
    ///
    /// Returns intersection points that lie on the segment (t in [0, 1]).
    pub fn intersect_segment(&self, segment: &Segment2<F>) -> Vec<(Point2<F>, F)> {
        let local_start = self.to_local(segment.start);
        let local_dir = self.dir_to_local(segment.direction());

        let a = self.semi_major;
        let b = self.semi_minor;

        let ox = local_start.x;
        let oy = local_start.y;
        let dx = local_dir.x;
        let dy = local_dir.y;

        let coef_a = (dx * dx) / (a * a) + (dy * dy) / (b * b);
        let coef_b = F::from(2.0).unwrap() * ((ox * dx) / (a * a) + (oy * dy) / (b * b));
        let coef_c = (ox * ox) / (a * a) + (oy * oy) / (b * b) - F::one();

        let all_hits =
            self.solve_quadratic_intersection(coef_a, coef_b, coef_c, &local_dir, &local_start);
        all_hits
            .into_iter()
            .filter(|(_, t)| *t >= F::zero() && *t <= F::one())
            .collect()
    }

    /// Intersects this ellipse with a circle.
    ///
    /// Returns 0, 1, 2, 3, or 4 intersection points.
    /// Uses numerical iteration for the general case.
    pub fn intersect_circle(&self, circle: &Circle2<F>) -> Vec<Point2<F>> {
        // Special case: if this is actually a circle
        if self.is_circle() {
            let my_circle = Circle2::new(self.center, self.semi_major);
            return my_circle.intersect_circle(circle).unwrap_or_default();
        }

        // General case: sample the ellipse and find intersections numerically
        self.intersect_implicit(|p| {
            let dx = p.x - circle.center.x;
            let dy = p.y - circle.center.y;
            dx * dx + dy * dy - circle.radius * circle.radius
        })
    }

    /// Intersects this ellipse with another ellipse.
    ///
    /// Returns 0, 1, 2, 3, or 4 intersection points.
    /// Uses numerical methods for the general case.
    pub fn intersect_ellipse(&self, other: &Ellipse2<F>) -> Vec<Point2<F>> {
        // Use the implicit equation of the other ellipse
        self.intersect_implicit(|p| {
            let local = other.to_local(p);
            let nx = local.x / other.semi_major;
            let ny = local.y / other.semi_minor;
            nx * nx + ny * ny - F::one()
        })
    }

    /// Helper: solves quadratic and returns intersection points.
    fn solve_quadratic_intersection(
        &self,
        a: F,
        b: F,
        c: F,
        local_dir: &Vec2<F>,
        local_origin: &Point2<F>,
    ) -> Vec<(Point2<F>, F)> {
        if a.abs() < F::epsilon() {
            // Linear case (degenerate)
            if b.abs() < F::epsilon() {
                return Vec::new();
            }
            let t = -c / b;
            let local_point = Point2::new(
                local_origin.x + t * local_dir.x,
                local_origin.y + t * local_dir.y,
            );
            return vec![(self.from_local(local_point), t)];
        }

        let discriminant = b * b - F::from(4.0).unwrap() * a * c;

        if discriminant < F::zero() {
            return Vec::new();
        }

        let sqrt_disc = discriminant.sqrt();
        let two_a = F::from(2.0).unwrap() * a;

        let t1 = (-b - sqrt_disc) / two_a;
        let t2 = (-b + sqrt_disc) / two_a;

        let mut results = Vec::new();

        let local_p1 = Point2::new(
            local_origin.x + t1 * local_dir.x,
            local_origin.y + t1 * local_dir.y,
        );
        results.push((self.from_local(local_p1), t1));

        if discriminant > F::epsilon() {
            let local_p2 = Point2::new(
                local_origin.x + t2 * local_dir.x,
                local_origin.y + t2 * local_dir.y,
            );
            results.push((self.from_local(local_p2), t2));
        }

        // Sort by t
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Helper: finds intersections with an implicit curve f(x,y) = 0.
    ///
    /// Samples around the ellipse and finds sign changes.
    fn intersect_implicit<G>(&self, f: G) -> Vec<Point2<F>>
    where
        G: Fn(Point2<F>) -> F,
    {
        let num_samples = 64;
        let tau = F::from(std::f64::consts::TAU).unwrap();
        let mut results = Vec::new();
        let tolerance = F::from(1e-8).unwrap();

        // Sample around the ellipse looking for sign changes
        let mut prev_t = F::zero();
        let mut prev_val = f(self.point_at(prev_t));

        for i in 1..=num_samples {
            let t = tau * F::from(i).unwrap() / F::from(num_samples).unwrap();
            let point = self.point_at(t);
            let val = f(point);

            // Sign change indicates intersection
            if prev_val * val < F::zero() {
                // Binary search to refine
                if let Some(intersection) = self.bisect_intersection(&f, prev_t, t, tolerance) {
                    // Avoid duplicates
                    let is_duplicate = results
                        .iter()
                        .any(|p: &Point2<F>| p.distance(intersection) < tolerance);
                    if !is_duplicate {
                        results.push(intersection);
                    }
                }
            } else if val.abs() < tolerance {
                // Point is on the curve
                let is_duplicate = results
                    .iter()
                    .any(|p: &Point2<F>| p.distance(point) < tolerance);
                if !is_duplicate {
                    results.push(point);
                }
            }

            prev_t = t;
            prev_val = val;
        }

        results
    }

    /// Binary search to find exact intersection point.
    fn bisect_intersection<G>(
        &self,
        f: &G,
        mut t_low: F,
        mut t_high: F,
        tolerance: F,
    ) -> Option<Point2<F>>
    where
        G: Fn(Point2<F>) -> F,
    {
        let max_iterations = 50;

        for _ in 0..max_iterations {
            let t_mid = (t_low + t_high) / (F::one() + F::one());
            let p_mid = self.point_at(t_mid);
            let val_mid = f(p_mid);

            if val_mid.abs() < tolerance {
                return Some(p_mid);
            }

            let val_low = f(self.point_at(t_low));
            if val_low * val_mid < F::zero() {
                t_high = t_mid;
            } else {
                t_low = t_mid;
            }

            if (t_high - t_low).abs() < tolerance {
                return Some(p_mid);
            }
        }

        Some(self.point_at((t_low + t_high) / (F::one() + F::one())))
    }

    /// Checks if this ellipse intersects a line.
    #[inline]
    pub fn intersects_line(&self, line: &Line2<F>) -> bool {
        !self.intersect_line(line).is_empty()
    }

    /// Checks if this ellipse intersects a ray.
    #[inline]
    pub fn intersects_ray(&self, ray: &Ray2<F>) -> bool {
        !self.intersect_ray(ray).is_empty()
    }

    /// Checks if this ellipse intersects a segment.
    #[inline]
    pub fn intersects_segment(&self, segment: &Segment2<F>) -> bool {
        !self.intersect_segment(segment).is_empty()
    }

    /// Checks if this ellipse intersects a circle.
    #[inline]
    pub fn intersects_circle(&self, circle: &Circle2<F>) -> bool {
        // Quick bounding box check first
        let (e_min, e_max) = self.bounding_box();
        let c_min = Point2::new(
            circle.center.x - circle.radius,
            circle.center.y - circle.radius,
        );
        let c_max = Point2::new(
            circle.center.x + circle.radius,
            circle.center.y + circle.radius,
        );

        if e_max.x < c_min.x || e_min.x > c_max.x || e_max.y < c_min.y || e_min.y > c_max.y {
            return false;
        }

        !self.intersect_circle(circle).is_empty()
    }

    /// Checks if this ellipse intersects another ellipse.
    #[inline]
    pub fn intersects_ellipse(&self, other: &Ellipse2<F>) -> bool {
        // Quick bounding box check
        let (a_min, a_max) = self.bounding_box();
        let (b_min, b_max) = other.bounding_box();

        if a_max.x < b_min.x || a_min.x > b_max.x || a_max.y < b_min.y || a_min.y > b_max.y {
            return false;
        }

        !self.intersect_ellipse(other).is_empty()
    }
}

impl<F: Float> Default for Ellipse2<F> {
    fn default() -> Self {
        Self::from_circle(Point2::origin(), F::one())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_axis_aligned() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::new(1.0, 2.0), 3.0, 2.0);
        assert_eq!(e.center.x, 1.0);
        assert_eq!(e.center.y, 2.0);
        assert_eq!(e.semi_major, 3.0);
        assert_eq!(e.semi_minor, 2.0);
        assert_eq!(e.rotation, 0.0);
    }

    #[test]
    fn test_from_circle() {
        let e: Ellipse2<f64> = Ellipse2::from_circle(Point2::origin(), 5.0);
        assert!(e.is_circle());
        assert_eq!(e.semi_major, 5.0);
        assert_eq!(e.semi_minor, 5.0);
    }

    #[test]
    fn test_is_circle() {
        let circle: Ellipse2<f64> = Ellipse2::from_circle(Point2::origin(), 1.0);
        let ellipse: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        assert!(circle.is_circle());
        assert!(!ellipse.is_circle());
    }

    #[test]
    fn test_eccentricity() {
        let circle: Ellipse2<f64> = Ellipse2::from_circle(Point2::origin(), 1.0);
        assert_relative_eq!(circle.eccentricity(), 0.0, epsilon = 1e-10);

        // For a=2, b=1: e = sqrt(1 - 1/4) = sqrt(3/4)
        let ellipse: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        assert_relative_eq!(ellipse.eccentricity(), (0.75_f64).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_area() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        assert_relative_eq!(e.area(), 2.0 * std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_circumference() {
        // For a circle, circumference should be 2*pi*r
        let circle: Ellipse2<f64> = Ellipse2::from_circle(Point2::origin(), 1.0);
        assert_relative_eq!(
            circle.circumference(),
            std::f64::consts::TAU,
            epsilon = 0.01
        );
    }

    #[test]
    fn test_contains_axis_aligned() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        assert!(e.contains(Point2::new(0.0, 0.0))); // Center
        assert!(e.contains(Point2::new(2.0, 0.0))); // On boundary (major)
        assert!(e.contains(Point2::new(0.0, 1.0))); // On boundary (minor)
        assert!(e.contains(Point2::new(1.0, 0.5))); // Inside
        assert!(!e.contains(Point2::new(2.0, 1.0))); // Outside
        assert!(!e.contains(Point2::new(3.0, 0.0))); // Outside
    }

    #[test]
    fn test_contains_rotated() {
        // Ellipse rotated 90 degrees: major axis now along y
        let e: Ellipse2<f64> =
            Ellipse2::new(Point2::origin(), 2.0, 1.0, std::f64::consts::FRAC_PI_2);

        assert!(e.contains(Point2::new(0.0, 2.0))); // On boundary (now along y)
        assert!(e.contains(Point2::new(1.0, 0.0))); // On boundary (minor, now along x)
        assert!(!e.contains(Point2::new(2.0, 0.0))); // Outside (was major, now minor direction)
    }

    #[test]
    fn test_point_at() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        let p0 = e.point_at(0.0);
        assert_relative_eq!(p0.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(p0.y, 0.0, epsilon = 1e-10);

        let p90 = e.point_at(std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(p90.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(p90.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bounding_box_axis_aligned() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::new(1.0, 2.0), 3.0, 2.0);
        let (min, max) = e.bounding_box();

        assert_relative_eq!(min.x, -2.0, epsilon = 1e-10);
        assert_relative_eq!(min.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max.x, 4.0, epsilon = 1e-10);
        assert_relative_eq!(max.y, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bounding_box_rotated() {
        // 45 degree rotation
        let e: Ellipse2<f64> =
            Ellipse2::new(Point2::origin(), 2.0, 1.0, std::f64::consts::FRAC_PI_4);
        let (min, max) = e.bounding_box();

        // At 45 degrees, both dimensions should be sqrt(a^2/2 + b^2/2) = sqrt(2.5)
        let expected = (2.5_f64).sqrt();
        assert_relative_eq!(min.x, -expected, epsilon = 1e-10);
        assert_relative_eq!(min.y, -expected, epsilon = 1e-10);
        assert_relative_eq!(max.x, expected, epsilon = 1e-10);
        assert_relative_eq!(max.y, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_point() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        // Point on major axis
        let nearest = e.nearest_point(Point2::new(5.0, 0.0));
        assert_relative_eq!(nearest.x, 2.0, epsilon = 1e-6);
        assert_relative_eq!(nearest.y, 0.0, epsilon = 1e-6);

        // Point on minor axis
        let nearest = e.nearest_point(Point2::new(0.0, 5.0));
        assert_relative_eq!(nearest.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(nearest.y, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_signed_distance() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        // Inside
        let dist = e.signed_distance(Point2::new(0.0, 0.0));
        assert!(dist < 0.0);

        // Outside
        let dist = e.signed_distance(Point2::new(3.0, 0.0));
        assert!(dist > 0.0);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_to_polygon() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let poly = e.to_polygon(4);

        assert_eq!(poly.len(), 4);
        assert_relative_eq!(poly[0].x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(poly[0].y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_foci() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let (f1, f2) = e.foci();

        // c = sqrt(a^2 - b^2) = sqrt(3)
        let c = (3.0_f64).sqrt();
        assert_relative_eq!(f1.x, -c, epsilon = 1e-10);
        assert_relative_eq!(f2.x, c, epsilon = 1e-10);
        assert_relative_eq!(f1.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(f2.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scaled() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let scaled = e.scaled(2.0);

        assert_eq!(scaled.semi_major, 4.0);
        assert_eq!(scaled.semi_minor, 2.0);
    }

    #[test]
    fn test_translated() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let translated = e.translated(Vec2::new(3.0, 4.0));

        assert_eq!(translated.center.x, 3.0);
        assert_eq!(translated.center.y, 4.0);
    }

    #[test]
    fn test_rotated() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let rotated = e.rotated(std::f64::consts::FRAC_PI_2);

        assert_relative_eq!(
            rotated.rotation,
            std::f64::consts::FRAC_PI_2,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_f32_support() {
        let e: Ellipse2<f32> = Ellipse2::axis_aligned(Point2::new(1.0, 2.0), 3.0, 2.0);
        assert!(e.contains(Point2::new(1.0, 2.0)));
    }

    #[test]
    fn test_major_minor_axis_directions() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        let major = e.major_axis_direction();
        let minor = e.minor_axis_direction();

        assert_relative_eq!(major.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(major.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(minor.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(minor.y, 1.0, epsilon = 1e-10);
    }

    // ==================== Intersection Tests ====================

    #[test]
    fn test_intersect_line_horizontal() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let line = Line2::horizontal(0.0);

        let hits = e.intersect_line(&line);
        assert_eq!(hits.len(), 2);

        // Should intersect at (-2, 0) and (2, 0)
        assert_relative_eq!(hits[0].0.x, -2.0, epsilon = 1e-10);
        assert_relative_eq!(hits[0].0.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.x, 2.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_line_vertical() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let line = Line2::vertical(0.0);

        let hits = e.intersect_line(&line);
        assert_eq!(hits.len(), 2);

        // Should intersect at (0, -1) and (0, 1)
        assert_relative_eq!(hits[0].0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(hits[0].0.y, -1.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_line_tangent() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let line = Line2::horizontal(1.0); // Tangent at top

        let hits = e.intersect_line(&line);
        assert_eq!(hits.len(), 1);
        assert_relative_eq!(hits[0].0.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(hits[0].0.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_line_miss() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let line = Line2::horizontal(5.0);

        let hits = e.intersect_line(&line);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_intersect_line_rotated_ellipse() {
        // Ellipse rotated 90 degrees
        let e: Ellipse2<f64> =
            Ellipse2::new(Point2::origin(), 2.0, 1.0, std::f64::consts::FRAC_PI_2);
        let line = Line2::horizontal(0.0);

        let hits = e.intersect_line(&line);
        assert_eq!(hits.len(), 2);

        // After rotation, major axis is along y, so horizontal line through center
        // should intersect at (-1, 0) and (1, 0)
        assert_relative_eq!(hits[0].0.x, -1.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.x, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_ray_through_center() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let ray = Ray2::new(Point2::new(-5.0, 0.0), Vec2::new(1.0, 0.0));

        let hits = e.intersect_ray(&ray);
        assert_eq!(hits.len(), 2);

        assert_relative_eq!(hits[0].0.x, -2.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_ray_from_inside() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let ray = Ray2::new(Point2::origin(), Vec2::new(1.0, 0.0));

        let hits = e.intersect_ray(&ray);
        assert_eq!(hits.len(), 1); // Only exit point

        assert_relative_eq!(hits[0].0.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_ray_miss() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let ray = Ray2::new(Point2::new(5.0, 5.0), Vec2::new(1.0, 0.0));

        let hits = e.intersect_ray(&ray);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_intersect_segment_through() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let seg = Segment2::new(Point2::new(-5.0, 0.0), Point2::new(5.0, 0.0));

        let hits = e.intersect_segment(&seg);
        assert_eq!(hits.len(), 2);

        assert_relative_eq!(hits[0].0.x, -2.0, epsilon = 1e-10);
        assert_relative_eq!(hits[1].0.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_segment_partial() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let seg = Segment2::new(Point2::new(0.0, 0.0), Point2::new(5.0, 0.0));

        let hits = e.intersect_segment(&seg);
        assert_eq!(hits.len(), 1);

        assert_relative_eq!(hits[0].0.x, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_intersect_segment_miss() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let seg = Segment2::new(Point2::new(3.0, 0.0), Point2::new(5.0, 0.0));

        let hits = e.intersect_segment(&seg);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_intersect_circle_two_points() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let circle = Circle2::new(Point2::new(1.5, 0.0), 1.0);

        let hits = e.intersect_circle(&circle);
        assert_eq!(hits.len(), 2);

        // Both points should be on both curves
        for p in &hits {
            // Check on ellipse
            let local = e.to_local(*p);
            let val = (local.x / 2.0).powi(2) + local.y.powi(2);
            assert_relative_eq!(val, 1.0, epsilon = 1e-6);

            // Check on circle
            let dist = p.distance(circle.center);
            assert_relative_eq!(dist, circle.radius, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_intersect_circle_no_intersection() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let circle = Circle2::new(Point2::new(10.0, 0.0), 1.0);

        let hits = e.intersect_circle(&circle);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_intersect_ellipse_two_ellipses() {
        let e1: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let e2: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::new(1.0, 0.0), 2.0, 1.0);

        let hits = e1.intersect_ellipse(&e2);
        assert!(hits.len() >= 2); // Should have at least 2 intersections

        // All points should be on both ellipses
        for p in &hits {
            assert!(e1.contains(*p) || e1.signed_distance(*p).abs() < 1e-5);
            assert!(e2.contains(*p) || e2.signed_distance(*p).abs() < 1e-5);
        }
    }

    #[test]
    fn test_intersect_ellipse_no_intersection() {
        let e1: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 1.0, 0.5);
        let e2: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::new(10.0, 0.0), 1.0, 0.5);

        let hits = e1.intersect_ellipse(&e2);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_intersects_line() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        assert!(e.intersects_line(&Line2::horizontal(0.0)));
        assert!(!e.intersects_line(&Line2::horizontal(5.0)));
    }

    #[test]
    fn test_intersects_ray() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        let ray_hit = Ray2::new(Point2::new(-5.0, 0.0), Vec2::new(1.0, 0.0));
        let ray_miss = Ray2::new(Point2::new(-5.0, 5.0), Vec2::new(1.0, 0.0));

        assert!(e.intersects_ray(&ray_hit));
        assert!(!e.intersects_ray(&ray_miss));
    }

    #[test]
    fn test_intersects_segment() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        let seg_hit = Segment2::new(Point2::new(-5.0, 0.0), Point2::new(5.0, 0.0));
        let seg_miss = Segment2::new(Point2::new(3.0, 0.0), Point2::new(5.0, 0.0));

        assert!(e.intersects_segment(&seg_hit));
        assert!(!e.intersects_segment(&seg_miss));
    }

    #[test]
    fn test_intersects_circle() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);

        let circle_hit = Circle2::new(Point2::new(1.5, 0.0), 1.0);
        let circle_miss = Circle2::new(Point2::new(10.0, 0.0), 1.0);

        assert!(e.intersects_circle(&circle_hit));
        assert!(!e.intersects_circle(&circle_miss));
    }

    #[test]
    fn test_diagonal_line_intersection() {
        let e: Ellipse2<f64> = Ellipse2::axis_aligned(Point2::origin(), 2.0, 1.0);
        let line = Line2::from_points(Point2::new(-3.0, -3.0), Point2::new(3.0, 3.0));

        let hits = e.intersect_line(&line);
        assert_eq!(hits.len(), 2);

        // Verify both points are on the ellipse
        for (p, _) in &hits {
            let val = (p.x / 2.0).powi(2) + p.y.powi(2);
            assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        }
    }
}
