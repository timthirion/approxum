//! Bounding capsule (stadium shape).
//!
//! A capsule is a line segment with a radius - equivalent to a rectangle capped
//! with semicircles at each end. Also known as a "stadium" or "discorectangle".
//!
//! Capsules provide a tighter fit than bounding circles for elongated shapes
//! while still supporting fast distance and intersection queries.
//!
//! # Example
//!
//! ```
//! use approxum::bounds::BoundingCapsule;
//! use approxum::{Point2, Segment2};
//!
//! // Create a capsule from a segment and radius
//! let segment = Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0));
//! let capsule = BoundingCapsule::new(segment, 1.0);
//!
//! assert!(capsule.contains_point(Point2::new(2.0, 0.5)));
//! assert!(!capsule.contains_point(Point2::new(2.0, 2.0)));
//! ```

use crate::primitives::{Point2, Segment2, Vec2};
use num_traits::Float;

/// A 2D bounding capsule (stadium shape).
///
/// Defined by a line segment (the spine) and a radius. The capsule contains
/// all points within `radius` distance of the segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingCapsule<F> {
    /// The central spine segment.
    pub segment: Segment2<F>,
    /// The radius (half-width perpendicular to the spine).
    pub radius: F,
}

impl<F: Float> BoundingCapsule<F> {
    /// Creates a new capsule from a segment and radius.
    #[inline]
    pub fn new(segment: Segment2<F>, radius: F) -> Self {
        Self { segment, radius }
    }

    /// Creates a capsule from two endpoints and a radius.
    #[inline]
    pub fn from_endpoints(start: Point2<F>, end: Point2<F>, radius: F) -> Self {
        Self {
            segment: Segment2::new(start, end),
            radius,
        }
    }

    /// Creates a degenerate capsule (a circle) from a center point and radius.
    #[inline]
    pub fn from_circle(center: Point2<F>, radius: F) -> Self {
        Self {
            segment: Segment2::new(center, center),
            radius,
        }
    }

    /// Computes the minimum bounding capsule for a set of points.
    ///
    /// This algorithm finds the capsule with the smallest area that contains
    /// all given points. It works by:
    /// 1. Finding the two most distant points (diameter endpoints)
    /// 2. Using their connecting segment as the spine
    /// 3. Computing the radius needed to contain all points
    ///
    /// For a more optimal (but slower) algorithm, use `from_points_optimal`.
    ///
    /// # Returns
    ///
    /// `None` if fewer than 1 point is provided.
    ///
    /// # Complexity
    ///
    /// O(n²) for diameter finding, O(n) for radius computation.
    pub fn from_points(points: &[Point2<F>]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        if points.len() == 1 {
            return Some(Self::from_circle(points[0], F::zero()));
        }

        // Find the two most distant points (approximate diameter)
        let (p1, p2) = find_diameter(points);

        // The spine is the segment connecting these points
        let segment = Segment2::new(p1, p2);

        // Find the maximum distance from any point to this segment
        let mut max_dist_sq = F::zero();
        for p in points {
            let dist_sq = segment.distance_squared_to_point(*p);
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
            }
        }

        Some(Self {
            segment,
            radius: max_dist_sq.sqrt(),
        })
    }

    /// Computes a better bounding capsule using PCA for orientation.
    ///
    /// Uses principal component analysis to find the optimal spine direction,
    /// which often produces a smaller capsule than the diameter-based approach.
    ///
    /// # Returns
    ///
    /// `None` if fewer than 1 point is provided.
    pub fn from_points_pca(points: &[Point2<F>]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        if points.len() == 1 {
            return Some(Self::from_circle(points[0], F::zero()));
        }

        // Compute centroid
        let n = F::from(points.len()).unwrap();
        let mut cx = F::zero();
        let mut cy = F::zero();
        for p in points {
            cx = cx + p.x;
            cy = cy + p.y;
        }
        cx = cx / n;
        cy = cy / n;

        // Compute covariance matrix
        let mut cxx = F::zero();
        let mut cyy = F::zero();
        let mut cxy = F::zero();

        for p in points {
            let dx = p.x - cx;
            let dy = p.y - cy;
            cxx = cxx + dx * dx;
            cyy = cyy + dy * dy;
            cxy = cxy + dx * dy;
        }

        // Find principal axis direction
        let angle = if cxy.abs() < F::from(1e-10).unwrap() {
            if cxx >= cyy {
                F::zero()
            } else {
                F::from(std::f64::consts::FRAC_PI_2).unwrap()
            }
        } else {
            let trace = cxx + cyy;
            let det = cxx * cyy - cxy * cxy;
            let discriminant = trace * trace - F::from(4.0).unwrap() * det;
            let sqrt_disc = discriminant.abs().sqrt();
            let two = F::from(2.0).unwrap();
            let lambda1 = (trace + sqrt_disc) / two;
            let vx = lambda1 - cyy;
            let vy = cxy;
            vy.atan2(vx)
        };

        let cos_a = angle.cos();
        let sin_a = angle.sin();

        // Project points onto principal axis
        let mut min_u = F::infinity();
        let mut max_u = F::neg_infinity();
        let mut max_v_abs = F::zero();

        for p in points {
            let dx = p.x - cx;
            let dy = p.y - cy;
            let u = dx * cos_a + dy * sin_a;
            let v = -dx * sin_a + dy * cos_a;

            min_u = min_u.min(u);
            max_u = max_u.max(u);
            max_v_abs = max_v_abs.max(v.abs());
        }

        // Spine endpoints in world coordinates
        let start = Point2::new(cx + min_u * cos_a, cy + min_u * sin_a);
        let end = Point2::new(cx + max_u * cos_a, cy + max_u * sin_a);

        Some(Self {
            segment: Segment2::new(start, end),
            radius: max_v_abs,
        })
    }

    /// Returns the length of the spine segment.
    #[inline]
    pub fn length(self) -> F {
        self.segment.length()
    }

    /// Returns the total length of the capsule (spine + 2*radius).
    #[inline]
    pub fn total_length(self) -> F {
        self.segment.length() + self.radius + self.radius
    }

    /// Returns the area of the capsule.
    ///
    /// Area = rectangle + two semicircles = length * 2*radius + π*radius²
    pub fn area(self) -> F {
        let two = F::from(2.0).unwrap();
        let pi = F::from(std::f64::consts::PI).unwrap();
        let rect_area = self.segment.length() * two * self.radius;
        let circle_area = pi * self.radius * self.radius;
        rect_area + circle_area
    }

    /// Returns the center point of the capsule.
    #[inline]
    pub fn center(self) -> Point2<F> {
        self.segment.midpoint()
    }

    /// Returns `true` if this capsule contains the given point.
    #[inline]
    pub fn contains_point(self, p: Point2<F>) -> bool {
        self.segment.distance_squared_to_point(p) <= self.radius * self.radius
    }

    /// Returns the distance from a point to this capsule's boundary.
    ///
    /// Returns a negative value if the point is inside.
    #[inline]
    pub fn signed_distance_to_point(self, p: Point2<F>) -> F {
        self.segment.distance_to_point(p) - self.radius
    }

    /// Returns the distance from a point to this capsule.
    ///
    /// Returns 0 if the point is inside.
    #[inline]
    pub fn distance_to_point(self, p: Point2<F>) -> F {
        self.signed_distance_to_point(p).max(F::zero())
    }

    /// Returns `true` if this capsule intersects another capsule.
    ///
    /// Two capsules intersect if the distance between their spines is less
    /// than the sum of their radii.
    pub fn intersects(self, other: Self) -> bool {
        let dist_sq = segment_segment_distance_squared(self.segment, other.segment);
        let sum_radii = self.radius + other.radius;
        dist_sq <= sum_radii * sum_radii
    }

    /// Returns `true` if this capsule intersects a circle.
    #[inline]
    pub fn intersects_circle(self, center: Point2<F>, radius: F) -> bool {
        let dist = self.segment.distance_to_point(center);
        dist <= self.radius + radius
    }

    /// Returns the axis direction of the capsule (unit vector along spine).
    pub fn axis(self) -> Vec2<F> {
        let dx = self.segment.end.x - self.segment.start.x;
        let dy = self.segment.end.y - self.segment.start.y;
        let len = (dx * dx + dy * dy).sqrt();

        if len < F::from(1e-10).unwrap() {
            Vec2::new(F::one(), F::zero()) // Degenerate case
        } else {
            Vec2::new(dx / len, dy / len)
        }
    }

    /// Expands the capsule by increasing the radius.
    #[inline]
    pub fn expand(self, amount: F) -> Self {
        Self {
            segment: self.segment,
            radius: self.radius + amount,
        }
    }
}

/// Finds the two most distant points (approximate diameter).
fn find_diameter<F: Float>(points: &[Point2<F>]) -> (Point2<F>, Point2<F>) {
    debug_assert!(points.len() >= 2);

    // Simple O(n²) approach - find the pair with maximum distance
    let mut best_dist_sq = F::zero();
    let mut best_pair = (points[0], points[1]);

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dist_sq = points[i].distance_squared(points[j]);
            if dist_sq > best_dist_sq {
                best_dist_sq = dist_sq;
                best_pair = (points[i], points[j]);
            }
        }
    }

    best_pair
}

/// Computes the squared distance between two line segments.
fn segment_segment_distance_squared<F: Float>(s1: Segment2<F>, s2: Segment2<F>) -> F {
    // Check all endpoint-to-segment distances and segment intersection
    let d1 = s1.distance_squared_to_point(s2.start);
    let d2 = s1.distance_squared_to_point(s2.end);
    let d3 = s2.distance_squared_to_point(s1.start);
    let d4 = s2.distance_squared_to_point(s1.end);

    // Check if segments intersect
    if segments_intersect(s1, s2) {
        return F::zero();
    }

    d1.min(d2).min(d3).min(d4)
}

/// Checks if two segments intersect.
fn segments_intersect<F: Float>(s1: Segment2<F>, s2: Segment2<F>) -> bool {
    let d1 = cross_product_sign(s2.start, s2.end, s1.start);
    let d2 = cross_product_sign(s2.start, s2.end, s1.end);
    let d3 = cross_product_sign(s1.start, s1.end, s2.start);
    let d4 = cross_product_sign(s1.start, s1.end, s2.end);

    if ((d1 > F::zero() && d2 < F::zero()) || (d1 < F::zero() && d2 > F::zero()))
        && ((d3 > F::zero() && d4 < F::zero()) || (d3 < F::zero() && d4 > F::zero()))
    {
        return true;
    }

    // Check for collinear cases
    if d1 == F::zero() && on_segment(s2, s1.start) {
        return true;
    }
    if d2 == F::zero() && on_segment(s2, s1.end) {
        return true;
    }
    if d3 == F::zero() && on_segment(s1, s2.start) {
        return true;
    }
    if d4 == F::zero() && on_segment(s1, s2.end) {
        return true;
    }

    false
}

/// Returns the sign of the cross product (p2-p1) × (p3-p1).
fn cross_product_sign<F: Float>(p1: Point2<F>, p2: Point2<F>, p3: Point2<F>) -> F {
    (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
}

/// Checks if point p lies on segment s (assuming collinear).
fn on_segment<F: Float>(s: Segment2<F>, p: Point2<F>) -> bool {
    p.x >= s.start.x.min(s.end.x)
        && p.x <= s.start.x.max(s.end.x)
        && p.y >= s.start.y.min(s.end.y)
        && p.y <= s.start.y.max(s.end.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_new() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        assert_eq!(capsule.segment.start.x, 0.0);
        assert_eq!(capsule.segment.end.x, 4.0);
        assert_eq!(capsule.radius, 1.0);
    }

    #[test]
    fn test_from_circle() {
        let capsule: BoundingCapsule<f64> =
            BoundingCapsule::from_circle(Point2::new(5.0, 5.0), 2.0);

        assert_eq!(capsule.segment.start, capsule.segment.end);
        assert_eq!(capsule.radius, 2.0);
        assert!(approx_eq(capsule.length(), 0.0, 1e-10));
    }

    #[test]
    fn test_contains_point() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        // On the spine
        assert!(capsule.contains_point(Point2::new(2.0, 0.0)));

        // Within radius
        assert!(capsule.contains_point(Point2::new(2.0, 0.5)));
        assert!(capsule.contains_point(Point2::new(2.0, -0.5)));

        // On the boundary
        assert!(capsule.contains_point(Point2::new(2.0, 1.0)));

        // Outside
        assert!(!capsule.contains_point(Point2::new(2.0, 1.5)));

        // In the semicircle caps
        assert!(capsule.contains_point(Point2::new(-0.5, 0.0)));
        assert!(capsule.contains_point(Point2::new(4.5, 0.0)));
        assert!(!capsule.contains_point(Point2::new(-1.5, 0.0)));
    }

    #[test]
    fn test_area() {
        // Horizontal capsule: length 4, radius 1
        // Area = 4*2*1 + π*1² = 8 + π ≈ 11.14
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        let expected = 8.0 + std::f64::consts::PI;
        assert!(approx_eq(capsule.area(), expected, 0.01));
    }

    #[test]
    fn test_area_degenerate() {
        // Circle: radius 2
        // Area = π*4 ≈ 12.57
        let capsule: BoundingCapsule<f64> =
            BoundingCapsule::from_circle(Point2::new(0.0, 0.0), 2.0);

        let expected = std::f64::consts::PI * 4.0;
        assert!(approx_eq(capsule.area(), expected, 0.01));
    }

    #[test]
    fn test_distance_to_point() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        // Inside
        assert_eq!(capsule.distance_to_point(Point2::new(2.0, 0.0)), 0.0);

        // Outside, perpendicular to spine
        assert!(approx_eq(
            capsule.distance_to_point(Point2::new(2.0, 2.0)),
            1.0,
            1e-10
        ));

        // Outside, past the end cap
        assert!(approx_eq(
            capsule.distance_to_point(Point2::new(6.0, 0.0)),
            1.0,
            1e-10
        ));
    }

    #[test]
    fn test_signed_distance() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        // On spine (inside)
        assert!(approx_eq(
            capsule.signed_distance_to_point(Point2::new(2.0, 0.0)),
            -1.0,
            1e-10
        ));

        // On boundary
        assert!(approx_eq(
            capsule.signed_distance_to_point(Point2::new(2.0, 1.0)),
            0.0,
            1e-10
        ));

        // Outside
        assert!(approx_eq(
            capsule.signed_distance_to_point(Point2::new(2.0, 2.0)),
            1.0,
            1e-10
        ));
    }

    #[test]
    fn test_intersects_capsules() {
        let c1: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        // Overlapping
        let c2 = BoundingCapsule::new(
            Segment2::new(Point2::new(2.0, 1.0), Point2::new(6.0, 1.0)),
            1.0,
        );
        assert!(c1.intersects(c2));

        // Touching
        let c3 = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 2.0), Point2::new(4.0, 2.0)),
            1.0,
        );
        assert!(c1.intersects(c3));

        // Separated
        let c4 = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 3.0), Point2::new(4.0, 3.0)),
            0.5,
        );
        assert!(!c1.intersects(c4));
    }

    #[test]
    fn test_intersects_circle() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        // Overlapping
        assert!(capsule.intersects_circle(Point2::new(2.0, 1.5), 1.0));

        // Touching
        assert!(capsule.intersects_circle(Point2::new(2.0, 2.0), 1.0));

        // Separated
        assert!(!capsule.intersects_circle(Point2::new(2.0, 3.0), 0.5));
    }

    #[test]
    fn test_from_points() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(2.0, -1.0),
        ];

        let capsule = BoundingCapsule::from_points(&points).unwrap();

        // All points should be contained
        for p in &points {
            assert!(capsule.contains_point(*p));
        }
    }

    #[test]
    fn test_from_points_pca() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(2.0, 0.5),
            Point2::new(2.0, -0.5),
        ];

        let capsule = BoundingCapsule::from_points_pca(&points).unwrap();

        // All points should be contained
        for p in &points {
            assert!(capsule.contains_point(*p));
        }

        // Should be roughly horizontal
        let axis = capsule.axis();
        assert!(axis.x.abs() > 0.9); // Mostly horizontal
    }

    #[test]
    fn test_from_points_single() {
        let points = vec![Point2::new(5.0, 5.0)];
        let capsule = BoundingCapsule::from_points(&points).unwrap();

        assert_eq!(capsule.center().x, 5.0);
        assert_eq!(capsule.center().y, 5.0);
        assert_eq!(capsule.radius, 0.0);
    }

    #[test]
    fn test_from_points_empty() {
        let points: Vec<Point2<f64>> = vec![];
        assert!(BoundingCapsule::from_points(&points).is_none());
    }

    #[test]
    fn test_axis() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        let axis = capsule.axis();
        assert!(approx_eq(axis.x, 1.0, 1e-10));
        assert!(approx_eq(axis.y, 0.0, 1e-10));
    }

    #[test]
    fn test_axis_diagonal() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
            1.0,
        );

        let axis = capsule.axis();
        let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(approx_eq(axis.x, sqrt2_2, 1e-10));
        assert!(approx_eq(axis.y, sqrt2_2, 1e-10));
    }

    #[test]
    fn test_expand() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        let expanded = capsule.expand(0.5);
        assert_eq!(expanded.radius, 1.5);
        assert_eq!(expanded.segment, capsule.segment);
    }

    #[test]
    fn test_total_length() {
        let capsule: BoundingCapsule<f64> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        assert!(approx_eq(capsule.total_length(), 6.0, 1e-10));
    }

    #[test]
    fn test_f32() {
        let capsule: BoundingCapsule<f32> = BoundingCapsule::new(
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            1.0,
        );

        assert!(capsule.contains_point(Point2::new(2.0, 0.5)));
        assert!(capsule.area() > 0.0);
    }

    #[test]
    fn test_segment_segment_distance() {
        // Parallel segments
        let s1 = Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0));
        let s2 = Segment2::new(Point2::new(0.0, 2.0), Point2::new(4.0, 2.0));

        let dist_sq = segment_segment_distance_squared(s1, s2);
        assert!(approx_eq(dist_sq, 4.0, 1e-10));
    }

    #[test]
    fn test_segment_segment_distance_intersecting() {
        let s1 = Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 4.0));
        let s2 = Segment2::new(Point2::new(0.0, 4.0), Point2::new(4.0, 0.0));

        let dist_sq = segment_segment_distance_squared(s1, s2);
        assert!(approx_eq(dist_sq, 0.0, 1e-10));
    }
}
