//! 2D ellipse type.

use super::{Point2, Vec2};
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
    fn to_local(&self, point: Point2<F>) -> Point2<F> {
        let dx = point.x - self.center.x;
        let dy = point.y - self.center.y;
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        Point2::new(dx * cos_r + dy * sin_r, -dx * sin_r + dy * cos_r)
    }

    /// Transforms a point from local coordinates back to world coordinates.
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
        assert_relative_eq!(circle.circumference(), std::f64::consts::TAU, epsilon = 0.01);
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
        let e: Ellipse2<f64> = Ellipse2::new(
            Point2::origin(),
            2.0,
            1.0,
            std::f64::consts::FRAC_PI_2,
        );

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
        let e: Ellipse2<f64> = Ellipse2::new(
            Point2::origin(),
            2.0,
            1.0,
            std::f64::consts::FRAC_PI_4,
        );
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

        assert_relative_eq!(rotated.rotation, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
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
}
