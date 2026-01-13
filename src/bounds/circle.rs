//! Bounding circle and minimum enclosing circle computation.

use crate::primitives::Point2;
use num_traits::Float;

/// A 2D bounding circle defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingCircle<F> {
    /// Center of the circle.
    pub center: Point2<F>,
    /// Radius of the circle.
    pub radius: F,
}

impl<F: Float> BoundingCircle<F> {
    /// Creates a new bounding circle.
    #[inline]
    pub fn new(center: Point2<F>, radius: F) -> Self {
        Self { center, radius }
    }

    /// Creates a bounding circle containing a single point (radius = 0).
    #[inline]
    pub fn from_point(p: Point2<F>) -> Self {
        Self {
            center: p,
            radius: F::zero(),
        }
    }

    /// Creates the smallest circle containing two points.
    #[inline]
    pub fn from_two_points(a: Point2<F>, b: Point2<F>) -> Self {
        let center = a.midpoint(b);
        let radius = center.distance(a);
        Self { center, radius }
    }

    /// Creates the circumcircle of three points.
    ///
    /// Returns `None` if the points are collinear.
    pub fn from_three_points(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> Option<Self> {
        // Using circumcenter formula
        let d = (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
            * (F::one() + F::one());

        if d.abs() <= F::epsilon() {
            return None; // Collinear points
        }

        let a_sq = a.x * a.x + a.y * a.y;
        let b_sq = b.x * b.x + b.y * b.y;
        let c_sq = c.x * c.x + c.y * c.y;

        let cx = (a_sq * (b.y - c.y) + b_sq * (c.y - a.y) + c_sq * (a.y - b.y)) / d;
        let cy = (a_sq * (c.x - b.x) + b_sq * (a.x - c.x) + c_sq * (b.x - a.x)) / d;

        let center = Point2::new(cx, cy);
        let radius = center.distance(a);

        Some(Self { center, radius })
    }

    /// Returns the area of the circle.
    #[inline]
    pub fn area(self) -> F {
        let pi = F::from(std::f64::consts::PI).unwrap();
        pi * self.radius * self.radius
    }

    /// Returns the circumference of the circle.
    #[inline]
    pub fn circumference(self) -> F {
        let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();
        two_pi * self.radius
    }

    /// Returns `true` if the circle contains the given point.
    ///
    /// Points exactly on the boundary are considered inside.
    #[inline]
    pub fn contains_point(self, p: Point2<F>) -> bool {
        self.center.distance_squared(p) <= self.radius * self.radius
    }

    /// Returns `true` if the circle contains the given point within tolerance.
    #[inline]
    pub fn contains_point_eps(self, p: Point2<F>, eps: F) -> bool {
        let r_plus_eps = self.radius + eps;
        self.center.distance_squared(p) <= r_plus_eps * r_plus_eps
    }

    /// Returns `true` if this circle intersects another circle.
    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        let dist_sq = self.center.distance_squared(other.center);
        let r_sum = self.radius + other.radius;
        dist_sq <= r_sum * r_sum
    }

    /// Returns the distance from a point to the circle boundary.
    ///
    /// Negative if inside, positive if outside.
    #[inline]
    pub fn signed_distance_to_point(self, p: Point2<F>) -> F {
        self.center.distance(p) - self.radius
    }
}

/// Computes the minimum enclosing circle for a set of points.
///
/// Uses Welzl's algorithm with expected O(n) time complexity.
///
/// Returns `None` if the input is empty.
pub fn minimum_enclosing_circle<F: Float>(points: &[Point2<F>]) -> Option<BoundingCircle<F>> {
    if points.is_empty() {
        return None;
    }

    // Shuffle points for expected linear time
    let mut points: Vec<Point2<F>> = points.to_vec();

    // Simple deterministic shuffle based on coordinate values
    // This breaks adversarial worst-case inputs while remaining deterministic
    let n = points.len();
    if n > 1 {
        for i in (1..n).rev() {
            // Use coordinate values to compute a pseudo-random index
            let xi = points[i].x.to_f64().unwrap_or(0.0);
            let yi = points[i].y.to_f64().unwrap_or(0.0);
            let hash = ((xi * 1000.0) as usize).wrapping_add((yi * 1000.0) as usize);
            let j = hash % (i + 1);
            points.swap(i, j);
        }
    }

    Some(welzl_recursive(&points, &mut Vec::new()))
}

/// Recursive Welzl's algorithm.
///
/// `points` is the remaining points to process.
/// `boundary` contains 0-3 points that must be on the circle boundary.
fn welzl_recursive<F: Float>(points: &[Point2<F>], boundary: &mut Vec<Point2<F>>) -> BoundingCircle<F> {
    // Base cases: circle determined by boundary points
    if points.is_empty() || boundary.len() == 3 {
        return circle_from_boundary(boundary);
    }

    // Pick the last point (after shuffling, this is effectively random)
    let p = points[points.len() - 1];
    let rest = &points[..points.len() - 1];

    // Recursively find MEC without p
    let circle = welzl_recursive(rest, boundary);

    // If p is inside the circle, we're done
    if circle.contains_point_eps(p, F::epsilon() * F::from(100.0).unwrap()) {
        return circle;
    }

    // Otherwise, p must be on the boundary of the MEC
    boundary.push(p);
    let result = welzl_recursive(rest, boundary);
    boundary.pop();

    result
}

/// Constructs a circle from 0-3 boundary points.
fn circle_from_boundary<F: Float>(boundary: &[Point2<F>]) -> BoundingCircle<F> {
    match boundary.len() {
        0 => BoundingCircle::new(Point2::origin(), F::zero()),
        1 => BoundingCircle::from_point(boundary[0]),
        2 => BoundingCircle::from_two_points(boundary[0], boundary[1]),
        3 => {
            // Try circumcircle first
            if let Some(circle) = BoundingCircle::from_three_points(
                boundary[0],
                boundary[1],
                boundary[2],
            ) {
                circle
            } else {
                // Collinear: use the two furthest points
                let d01 = boundary[0].distance_squared(boundary[1]);
                let d02 = boundary[0].distance_squared(boundary[2]);
                let d12 = boundary[1].distance_squared(boundary[2]);

                if d01 >= d02 && d01 >= d12 {
                    BoundingCircle::from_two_points(boundary[0], boundary[1])
                } else if d02 >= d12 {
                    BoundingCircle::from_two_points(boundary[0], boundary[2])
                } else {
                    BoundingCircle::from_two_points(boundary[1], boundary[2])
                }
            }
        }
        _ => unreachable!("boundary should have at most 3 points"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bounding_circle_new() {
        let c: BoundingCircle<f64> = BoundingCircle::new(Point2::new(1.0, 2.0), 5.0);
        assert_eq!(c.center.x, 1.0);
        assert_eq!(c.center.y, 2.0);
        assert_eq!(c.radius, 5.0);
    }

    #[test]
    fn test_from_two_points() {
        let c: BoundingCircle<f64> =
            BoundingCircle::from_two_points(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0));
        assert_eq!(c.center.x, 5.0);
        assert_eq!(c.center.y, 0.0);
        assert_eq!(c.radius, 5.0);
    }

    #[test]
    fn test_from_three_points() {
        // Right triangle: (0,0), (1,0), (0,1) - circumcircle centered at (0.5, 0.5)
        let c: BoundingCircle<f64> = BoundingCircle::from_three_points(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        )
        .unwrap();
        assert_relative_eq!(c.center.x, 0.5, epsilon = 1e-10);
        assert_relative_eq!(c.center.y, 0.5, epsilon = 1e-10);
        assert_relative_eq!(c.radius, 0.5_f64.sqrt() + 0.5_f64.sqrt() - 0.5_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_from_three_points_collinear() {
        let c: Option<BoundingCircle<f64>> = BoundingCircle::from_three_points(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
        );
        assert!(c.is_none());
    }

    #[test]
    fn test_contains_point() {
        let c: BoundingCircle<f64> = BoundingCircle::new(Point2::new(0.0, 0.0), 5.0);
        assert!(c.contains_point(Point2::new(0.0, 0.0)));
        assert!(c.contains_point(Point2::new(3.0, 4.0))); // On boundary
        assert!(c.contains_point(Point2::new(3.0, 0.0)));
        assert!(!c.contains_point(Point2::new(6.0, 0.0)));
    }

    #[test]
    fn test_intersects() {
        let a: BoundingCircle<f64> = BoundingCircle::new(Point2::new(0.0, 0.0), 5.0);
        let b = BoundingCircle::new(Point2::new(8.0, 0.0), 5.0);
        let c = BoundingCircle::new(Point2::new(20.0, 0.0), 5.0);

        assert!(a.intersects(b)); // Overlapping
        assert!(!a.intersects(c)); // Too far apart
    }

    #[test]
    fn test_minimum_enclosing_circle_empty() {
        let points: Vec<Point2<f64>> = vec![];
        assert!(minimum_enclosing_circle(&points).is_none());
    }

    #[test]
    fn test_minimum_enclosing_circle_single() {
        let points = vec![Point2::new(5.0, 5.0)];
        let c: BoundingCircle<f64> = minimum_enclosing_circle(&points).unwrap();
        assert_eq!(c.center.x, 5.0);
        assert_eq!(c.center.y, 5.0);
        assert_eq!(c.radius, 0.0);
    }

    #[test]
    fn test_minimum_enclosing_circle_two() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)];
        let c: BoundingCircle<f64> = minimum_enclosing_circle(&points).unwrap();
        assert_relative_eq!(c.center.x, 5.0, epsilon = 1e-10);
        assert_relative_eq!(c.radius, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minimum_enclosing_circle_triangle() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(5.0, 5.0),
        ];
        let c: BoundingCircle<f64> = minimum_enclosing_circle(&points).unwrap();

        // All points should be inside or on the circle
        for p in &points {
            assert!(c.contains_point_eps(*p, 1e-9));
        }
    }

    #[test]
    fn test_minimum_enclosing_circle_square() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let c: BoundingCircle<f64> = minimum_enclosing_circle(&points).unwrap();

        // Center should be at (5, 5)
        assert_relative_eq!(c.center.x, 5.0, epsilon = 1e-9);
        assert_relative_eq!(c.center.y, 5.0, epsilon = 1e-9);

        // Radius should be diagonal/2 = sqrt(50)
        assert_relative_eq!(c.radius, 50.0_f64.sqrt(), epsilon = 1e-9);

        // All points should be on or inside the circle
        for p in &points {
            assert!(c.contains_point_eps(*p, 1e-9));
        }
    }

    #[test]
    fn test_minimum_enclosing_circle_with_interior_points() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(5.0, 5.0), // Interior point
            Point2::new(3.0, 2.0), // Interior point
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let c: BoundingCircle<f64> = minimum_enclosing_circle(&points).unwrap();

        // All points should be inside or on the circle
        for p in &points {
            assert!(c.contains_point_eps(*p, 1e-9));
        }

        // The circle should be the same as for just the square corners
        assert_relative_eq!(c.center.x, 5.0, epsilon = 1e-9);
        assert_relative_eq!(c.center.y, 5.0, epsilon = 1e-9);
    }
}
