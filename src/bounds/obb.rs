//! Oriented bounding box.
//!
//! An OBB provides a tighter fit than an AABB for rotated or elongated shapes.
//! Two construction algorithms are provided:
//!
//! - **PCA-based** (`from_points_pca`): Fast O(n) approximation using principal component analysis
//! - **Rotating calipers** (`from_points_optimal`): Optimal O(n log n) using convex hull
//!
//! # Example
//!
//! ```
//! use approxum::bounds::Obb2;
//! use approxum::Point2;
//!
//! // A rotated rectangle's vertices
//! let points = vec![
//!     Point2::new(1.0, 0.0),
//!     Point2::new(3.0, 1.0),
//!     Point2::new(2.0, 3.0),
//!     Point2::new(0.0, 2.0),
//! ];
//!
//! let obb = Obb2::from_points_pca(&points).unwrap();
//! println!("Center: {:?}, Area: {}", obb.center, obb.area());
//! ```

use crate::hull::convex_hull;
use crate::primitives::{Point2, Vec2};
use num_traits::Float;
use std::f64::consts::PI;

/// A 2D oriented bounding box.
///
/// Represented by a center point, half-extents along local axes, and an
/// orientation angle (radians, counter-clockwise from positive x-axis).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obb2<F> {
    /// Center of the OBB.
    pub center: Point2<F>,
    /// Half-width along the local x-axis.
    pub half_width: F,
    /// Half-height along the local y-axis.
    pub half_height: F,
    /// Rotation angle in radians (counter-clockwise from positive x-axis).
    pub angle: F,
}

impl<F: Float> Obb2<F> {
    /// Creates a new OBB with the given parameters.
    #[inline]
    pub fn new(center: Point2<F>, half_width: F, half_height: F, angle: F) -> Self {
        Self {
            center,
            half_width,
            half_height,
            angle,
        }
    }

    /// Creates an axis-aligned OBB (equivalent to an AABB).
    #[inline]
    pub fn axis_aligned(center: Point2<F>, half_width: F, half_height: F) -> Self {
        Self {
            center,
            half_width,
            half_height,
            angle: F::zero(),
        }
    }

    /// Constructs an OBB using Principal Component Analysis.
    ///
    /// This is a fast O(n) algorithm that computes the covariance matrix of the
    /// points and uses the eigenvectors as the OBB axes. Works well for most
    /// shapes but may not produce the minimum-area bounding box.
    ///
    /// # Returns
    ///
    /// `None` if fewer than 2 points are provided.
    pub fn from_points_pca(points: &[Point2<F>]) -> Option<Self> {
        if points.len() < 2 {
            return None;
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

        // Compute covariance matrix elements
        // Cov = | cxx cxy |
        //       | cxy cyy |
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

        // Find eigenvalues and eigenvectors of the covariance matrix
        // For a 2x2 symmetric matrix, we can solve analytically
        let trace = cxx + cyy;
        let det = cxx * cyy - cxy * cxy;

        // Eigenvalues: λ = (trace ± sqrt(trace² - 4*det)) / 2
        let discriminant = trace * trace - F::from(4.0).unwrap() * det;
        let sqrt_disc = discriminant.abs().sqrt();

        // Principal axis direction (eigenvector for larger eigenvalue)
        let angle = if cxy.abs() < F::from(1e-10).unwrap() {
            // Matrix is already diagonal
            if cxx >= cyy {
                F::zero()
            } else {
                F::from(PI / 2.0).unwrap()
            }
        } else {
            // Eigenvector: (λ - cyy, cxy) or (cxy, λ - cxx)
            let two = F::from(2.0).unwrap();
            let lambda1 = (trace + sqrt_disc) / two;
            // The eigenvector for lambda1 is (lambda1 - cyy, cxy)
            let vx = lambda1 - cyy;
            let vy = cxy;
            vy.atan2(vx)
        };

        // Project points onto the rotated axes and find extents
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let mut min_u = F::infinity();
        let mut max_u = F::neg_infinity();
        let mut min_v = F::infinity();
        let mut max_v = F::neg_infinity();

        for p in points {
            let dx = p.x - cx;
            let dy = p.y - cy;
            // Rotate to local coordinates
            let u = dx * cos_a + dy * sin_a;
            let v = -dx * sin_a + dy * cos_a;

            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }

        let half_width = (max_u - min_u) / F::from(2.0).unwrap();
        let half_height = (max_v - min_v) / F::from(2.0).unwrap();

        // Adjust center to be at the center of the bounding box
        let center_u = (min_u + max_u) / F::from(2.0).unwrap();
        let center_v = (min_v + max_v) / F::from(2.0).unwrap();

        // Transform center back to world coordinates
        let adjusted_center = Point2::new(
            cx + center_u * cos_a - center_v * sin_a,
            cy + center_u * sin_a + center_v * cos_a,
        );

        Some(Self {
            center: adjusted_center,
            half_width,
            half_height,
            angle,
        })
    }

    /// Constructs the minimum-area OBB using rotating calipers.
    ///
    /// This is an O(n log n) algorithm that first computes the convex hull,
    /// then uses the rotating calipers technique to find the optimal orientation.
    /// Guaranteed to find the minimum-area bounding box.
    ///
    /// # Returns
    ///
    /// `None` if fewer than 2 points are provided.
    pub fn from_points_optimal(points: &[Point2<F>]) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }

        if points.len() == 2 {
            // Special case: line segment
            let mid = points[0].midpoint(points[1]);
            let dx = points[1].x - points[0].x;
            let dy = points[1].y - points[0].y;
            let len = (dx * dx + dy * dy).sqrt();
            let angle = dy.atan2(dx);

            return Some(Self {
                center: mid,
                half_width: len / F::from(2.0).unwrap(),
                half_height: F::zero(),
                angle,
            });
        }

        // Compute convex hull using Graham scan
        let hull = convex_hull(points);

        if hull.len() < 3 {
            // Degenerate case: all points collinear
            return Self::from_points_pca(points);
        }

        // Rotating calipers to find minimum area OBB
        let n = hull.len();
        let mut best_area = F::infinity();
        let mut best_obb = None;

        // For each edge of the hull, compute the OBB aligned to that edge
        for i in 0..n {
            let j = (i + 1) % n;

            // Edge direction
            let edge = Vec2::new(hull[j].x - hull[i].x, hull[j].y - hull[i].y);
            let edge_len = (edge.x * edge.x + edge.y * edge.y).sqrt();

            if edge_len < F::from(1e-10).unwrap() {
                continue;
            }

            // Unit vectors for this orientation
            let ux = edge.x / edge_len;
            let uy = edge.y / edge_len;
            // Perpendicular
            let vx = -uy;
            let vy = ux;

            // Project all hull points onto this orientation
            let mut min_u = F::infinity();
            let mut max_u = F::neg_infinity();
            let mut min_v = F::infinity();
            let mut max_v = F::neg_infinity();

            for p in &hull {
                let u = p.x * ux + p.y * uy;
                let v = p.x * vx + p.y * vy;

                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }

            let width = max_u - min_u;
            let height = max_v - min_v;
            let area = width * height;

            if area < best_area {
                best_area = area;

                let center_u = (min_u + max_u) / F::from(2.0).unwrap();
                let center_v = (min_v + max_v) / F::from(2.0).unwrap();

                let center =
                    Point2::new(center_u * ux + center_v * vx, center_u * uy + center_v * vy);

                let angle = uy.atan2(ux);

                best_obb = Some(Self {
                    center,
                    half_width: width / F::from(2.0).unwrap(),
                    half_height: height / F::from(2.0).unwrap(),
                    angle,
                });
            }
        }

        best_obb
    }

    /// Returns the area of the OBB.
    #[inline]
    pub fn area(self) -> F {
        F::from(4.0).unwrap() * self.half_width * self.half_height
    }

    /// Returns the four corners of the OBB in counter-clockwise order.
    pub fn corners(self) -> [Point2<F>; 4] {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();

        let hw = self.half_width;
        let hh = self.half_height;

        // Local corners: (+hw, +hh), (-hw, +hh), (-hw, -hh), (+hw, -hh)
        let local = [(hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh)];

        let mut corners = [self.center; 4];
        for (i, (lx, ly)) in local.iter().enumerate() {
            corners[i] = Point2::new(
                self.center.x + *lx * cos_a - *ly * sin_a,
                self.center.y + *lx * sin_a + *ly * cos_a,
            );
        }
        corners
    }

    /// Returns the local x-axis direction (unit vector).
    #[inline]
    pub fn axis_x(self) -> Vec2<F> {
        Vec2::new(self.angle.cos(), self.angle.sin())
    }

    /// Returns the local y-axis direction (unit vector).
    #[inline]
    pub fn axis_y(self) -> Vec2<F> {
        Vec2::new(-self.angle.sin(), self.angle.cos())
    }

    /// Returns `true` if this OBB contains the given point.
    pub fn contains_point(self, p: Point2<F>) -> bool {
        // Transform point to local coordinates
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;

        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();

        let local_x = dx * cos_a + dy * sin_a;
        let local_y = -dx * sin_a + dy * cos_a;

        local_x.abs() <= self.half_width && local_y.abs() <= self.half_height
    }

    /// Returns `true` if this OBB intersects another OBB.
    ///
    /// Uses the Separating Axis Theorem (SAT).
    pub fn intersects(self, other: Self) -> bool {
        // Get all four axes to test (two from each OBB)
        let axes = [self.axis_x(), self.axis_y(), other.axis_x(), other.axis_y()];

        let corners_a = self.corners();
        let corners_b = other.corners();

        for axis in &axes {
            // Project both OBBs onto the axis
            let (min_a, max_a) = project_corners(&corners_a, *axis);
            let (min_b, max_b) = project_corners(&corners_b, *axis);

            // Check for separation
            if max_a < min_b || max_b < min_a {
                return false;
            }
        }

        true
    }

    /// Returns the distance from a point to this OBB.
    ///
    /// Returns 0 if the point is inside.
    pub fn distance_to_point(self, p: Point2<F>) -> F {
        self.distance_squared_to_point(p).sqrt()
    }

    /// Returns the squared distance from a point to this OBB.
    ///
    /// Returns 0 if the point is inside.
    pub fn distance_squared_to_point(self, p: Point2<F>) -> F {
        // Transform point to local coordinates
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;

        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();

        let local_x = dx * cos_a + dy * sin_a;
        let local_y = -dx * sin_a + dy * cos_a;

        // Clamp to box and compute distance
        let clamped_x = local_x.max(-self.half_width).min(self.half_width);
        let clamped_y = local_y.max(-self.half_height).min(self.half_height);

        let dist_x = local_x - clamped_x;
        let dist_y = local_y - clamped_y;

        dist_x * dist_x + dist_y * dist_y
    }
}

/// Projects corners onto an axis and returns (min, max).
fn project_corners<F: Float>(corners: &[Point2<F>; 4], axis: Vec2<F>) -> (F, F) {
    let mut min = F::infinity();
    let mut max = F::neg_infinity();

    for c in corners {
        let proj = c.x * axis.x + c.y * axis.y;
        min = min.min(proj);
        max = max.max(proj);
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_axis_aligned() {
        let obb: Obb2<f64> = Obb2::axis_aligned(Point2::new(5.0, 5.0), 2.0, 3.0);

        assert_eq!(obb.center.x, 5.0);
        assert_eq!(obb.center.y, 5.0);
        assert_eq!(obb.half_width, 2.0);
        assert_eq!(obb.half_height, 3.0);
        assert_eq!(obb.angle, 0.0);
        assert_eq!(obb.area(), 24.0);
    }

    #[test]
    fn test_corners_axis_aligned() {
        let obb: Obb2<f64> = Obb2::axis_aligned(Point2::new(0.0, 0.0), 1.0, 2.0);
        let corners = obb.corners();

        // Should be at (±1, ±2)
        assert!(approx_eq(corners[0].x, 1.0, 1e-10));
        assert!(approx_eq(corners[0].y, 2.0, 1e-10));
        assert!(approx_eq(corners[1].x, -1.0, 1e-10));
        assert!(approx_eq(corners[1].y, 2.0, 1e-10));
        assert!(approx_eq(corners[2].x, -1.0, 1e-10));
        assert!(approx_eq(corners[2].y, -2.0, 1e-10));
        assert!(approx_eq(corners[3].x, 1.0, 1e-10));
        assert!(approx_eq(corners[3].y, -2.0, 1e-10));
    }

    #[test]
    fn test_corners_rotated() {
        let obb: Obb2<f64> = Obb2::new(
            Point2::new(0.0, 0.0),
            1.0,
            0.0,
            std::f64::consts::FRAC_PI_4, // 45 degrees
        );
        let corners = obb.corners();

        // Rotated 45 degrees, corners should be at (±√2/2, ±√2/2)
        let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(approx_eq(corners[0].x, sqrt2_2, 1e-10));
        assert!(approx_eq(corners[0].y, sqrt2_2, 1e-10));
    }

    #[test]
    fn test_contains_point_axis_aligned() {
        let obb: Obb2<f64> = Obb2::axis_aligned(Point2::new(0.0, 0.0), 1.0, 1.0);

        assert!(obb.contains_point(Point2::new(0.0, 0.0)));
        assert!(obb.contains_point(Point2::new(0.5, 0.5)));
        assert!(obb.contains_point(Point2::new(1.0, 1.0))); // On boundary
        assert!(!obb.contains_point(Point2::new(1.5, 0.0)));
        assert!(!obb.contains_point(Point2::new(0.0, 1.5)));
    }

    #[test]
    fn test_contains_point_rotated() {
        // 45 degree rotated box
        let obb: Obb2<f64> =
            Obb2::new(Point2::new(0.0, 0.0), 2.0, 1.0, std::f64::consts::FRAC_PI_4);

        // Center should be inside
        assert!(obb.contains_point(Point2::new(0.0, 0.0)));

        // Point along the rotated x-axis
        let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(obb.contains_point(Point2::new(sqrt2_2, sqrt2_2))); // Along major axis
    }

    #[test]
    fn test_from_points_pca_simple() {
        // Axis-aligned rectangle
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 2.0),
            Point2::new(0.0, 2.0),
        ];

        let obb = Obb2::from_points_pca(&points).unwrap();

        // Center should be at (2, 1)
        assert!(approx_eq(obb.center.x, 2.0, 0.1));
        assert!(approx_eq(obb.center.y, 1.0, 0.1));

        // Area should be close to 8
        assert!(obb.area() >= 7.5 && obb.area() <= 9.0);
    }

    #[test]
    fn test_from_points_optimal_simple() {
        // Axis-aligned rectangle
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(4.0, 2.0),
            Point2::new(0.0, 2.0),
        ];

        let obb = Obb2::from_points_optimal(&points).unwrap();

        // Should find the optimal bounding box with area 8
        assert!(approx_eq(obb.area(), 8.0, 0.01));
    }

    #[test]
    fn test_from_points_optimal_rotated() {
        // Diamond shape (rotated square)
        let points = vec![
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(-1.0, 0.0),
            Point2::new(0.0, -1.0),
        ];

        let obb = Obb2::from_points_optimal(&points).unwrap();

        // Optimal OBB should be rotated 45 degrees with area 2
        assert!(approx_eq(obb.area(), 2.0, 0.01));
    }

    #[test]
    fn test_intersects_same_obb() {
        let obb: Obb2<f64> = Obb2::axis_aligned(Point2::new(0.0, 0.0), 1.0, 1.0);
        assert!(obb.intersects(obb));
    }

    #[test]
    fn test_intersects_overlapping() {
        let obb1: Obb2<f64> = Obb2::axis_aligned(Point2::new(0.0, 0.0), 1.0, 1.0);
        let obb2: Obb2<f64> = Obb2::axis_aligned(Point2::new(1.0, 0.0), 1.0, 1.0);

        assert!(obb1.intersects(obb2));
        assert!(obb2.intersects(obb1));
    }

    #[test]
    fn test_intersects_separated() {
        let obb1: Obb2<f64> = Obb2::axis_aligned(Point2::new(0.0, 0.0), 1.0, 1.0);
        let obb2: Obb2<f64> = Obb2::axis_aligned(Point2::new(5.0, 0.0), 1.0, 1.0);

        assert!(!obb1.intersects(obb2));
        assert!(!obb2.intersects(obb1));
    }

    #[test]
    fn test_intersects_rotated() {
        // Two rotated boxes that intersect
        let obb1: Obb2<f64> =
            Obb2::new(Point2::new(0.0, 0.0), 2.0, 1.0, std::f64::consts::FRAC_PI_4);
        let obb2: Obb2<f64> = Obb2::axis_aligned(Point2::new(0.0, 0.0), 1.0, 1.0);

        assert!(obb1.intersects(obb2));
    }

    #[test]
    fn test_distance_to_point() {
        let obb: Obb2<f64> = Obb2::axis_aligned(Point2::new(0.0, 0.0), 1.0, 1.0);

        // Inside
        assert_eq!(obb.distance_to_point(Point2::new(0.0, 0.0)), 0.0);

        // On boundary
        assert!(approx_eq(
            obb.distance_to_point(Point2::new(1.0, 0.0)),
            0.0,
            1e-10
        ));

        // Outside, aligned with axis
        assert!(approx_eq(
            obb.distance_to_point(Point2::new(2.0, 0.0)),
            1.0,
            1e-10
        ));

        // Outside, diagonal
        assert!(approx_eq(
            obb.distance_to_point(Point2::new(2.0, 2.0)),
            std::f64::consts::SQRT_2,
            1e-10
        ));
    }

    #[test]
    fn test_convex_hull() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 0.5), // Interior point
        ];

        let hull = convex_hull(&points);

        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_from_points_two_points() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)];

        let obb = Obb2::from_points_optimal(&points).unwrap();

        assert!(approx_eq(obb.center.x, 1.0, 1e-10));
        assert!(approx_eq(obb.center.y, 0.0, 1e-10));
        assert!(approx_eq(obb.half_width, 1.0, 1e-10));
    }

    #[test]
    fn test_f32() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];

        let obb = Obb2::from_points_pca(&points).unwrap();
        assert!(obb.area() > 0.0);
    }
}
