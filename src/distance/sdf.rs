//! Signed distance field computations.
//!
//! Signed distance fields (SDFs) represent shapes as scalar fields where each point
//! stores the signed distance to the nearest surface. Positive values are outside,
//! negative values are inside.

use crate::primitives::{Point2, Segment2};
use num_traits::Float;

/// Trait for shapes that can compute signed distance.
pub trait Sdf2<F: Float> {
    /// Returns the signed distance from point `p` to this shape.
    ///
    /// - Positive: outside the shape
    /// - Zero: on the boundary
    /// - Negative: inside the shape
    fn signed_distance(&self, p: Point2<F>) -> F;

    /// Returns the unsigned distance from point `p` to this shape.
    fn distance(&self, p: Point2<F>) -> F {
        self.signed_distance(p).abs()
    }

    /// Returns true if the point is inside the shape (negative distance).
    fn contains(&self, p: Point2<F>) -> bool {
        self.signed_distance(p) < F::zero()
    }
}

/// Signed distance to a point (always positive except at the point itself).
#[inline]
pub fn sdf_point<F: Float>(p: Point2<F>, target: Point2<F>) -> F {
    p.distance(target)
}

/// Signed distance to a line segment (always positive, segment has no interior).
#[inline]
pub fn sdf_segment<F: Float>(p: Point2<F>, segment: Segment2<F>) -> F {
    segment.distance_to_point(p)
}

/// Signed distance to a circle.
///
/// Negative inside, positive outside.
#[inline]
pub fn sdf_circle<F: Float>(p: Point2<F>, center: Point2<F>, radius: F) -> F {
    p.distance(center) - radius
}

/// Signed distance to a closed polygon.
///
/// Uses the winding number to determine inside/outside.
/// Negative inside, positive outside.
///
/// # Arguments
///
/// * `p` - Query point
/// * `vertices` - Polygon vertices in order (first vertex should NOT be repeated at end)
///
/// # Returns
///
/// Signed distance: negative if inside, positive if outside.
pub fn sdf_polygon<F: Float>(p: Point2<F>, vertices: &[Point2<F>]) -> F {
    if vertices.len() < 3 {
        return F::infinity();
    }

    let mut min_dist_sq = F::infinity();

    // Find minimum distance to all edges
    let n = vertices.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let edge = Segment2::new(vertices[i], vertices[j]);
        let dist_sq = edge.distance_to_point(p).powi(2);
        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
        }
    }

    let dist = min_dist_sq.sqrt();

    // Determine sign using winding number
    let winding = winding_number(p, vertices);

    if winding != 0 {
        -dist // Inside
    } else {
        dist // Outside
    }
}

/// Computes the winding number of a point with respect to a polygon.
///
/// Returns 0 if the point is outside, non-zero if inside.
fn winding_number<F: Float>(p: Point2<F>, vertices: &[Point2<F>]) -> i32 {
    let mut winding = 0i32;
    let n = vertices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let v1 = vertices[i];
        let v2 = vertices[j];

        if v1.y <= p.y {
            if v2.y > p.y {
                // Upward crossing
                if is_left(v1, v2, p) > F::zero() {
                    winding += 1;
                }
            }
        } else if v2.y <= p.y {
            // Downward crossing
            if is_left(v1, v2, p) < F::zero() {
                winding -= 1;
            }
        }
    }

    winding
}

/// Returns positive if p is left of line v1->v2, negative if right, zero if on line.
#[inline]
fn is_left<F: Float>(v1: Point2<F>, v2: Point2<F>, p: Point2<F>) -> F {
    (v2.x - v1.x) * (p.y - v1.y) - (p.x - v1.x) * (v2.y - v1.y)
}

/// A 2D signed distance field stored on a regular grid.
pub struct SdfGrid<F> {
    /// Width of the grid in cells.
    pub width: usize,
    /// Height of the grid in cells.
    pub height: usize,
    /// Origin (bottom-left corner) of the grid in world space.
    pub origin: Point2<F>,
    /// Size of each cell.
    pub cell_size: F,
    /// Distance values (row-major order).
    pub values: Vec<F>,
}

impl<F: Float> SdfGrid<F> {
    /// Creates a new SDF grid with uninitialized values.
    pub fn new(width: usize, height: usize, origin: Point2<F>, cell_size: F) -> Self {
        Self {
            width,
            height,
            origin,
            cell_size,
            values: vec![F::zero(); width * height],
        }
    }

    /// Creates an SDF grid from a shape that implements Sdf2.
    pub fn from_shape<S: Sdf2<F>>(
        shape: &S,
        width: usize,
        height: usize,
        origin: Point2<F>,
        cell_size: F,
    ) -> Self {
        let mut grid = Self::new(width, height, origin, cell_size);
        grid.compute_from_shape(shape);
        grid
    }

    /// Computes the SDF from a shape.
    pub fn compute_from_shape<S: Sdf2<F>>(&mut self, shape: &S) {
        let half = F::from(0.5).unwrap();

        for y in 0..self.height {
            for x in 0..self.width {
                let world_x =
                    self.origin.x + F::from(x).unwrap() * self.cell_size + half * self.cell_size;
                let world_y =
                    self.origin.y + F::from(y).unwrap() * self.cell_size + half * self.cell_size;
                let p = Point2::new(world_x, world_y);

                self.values[y * self.width + x] = shape.signed_distance(p);
            }
        }
    }

    /// Returns the world-space position of a grid cell center.
    pub fn cell_center(&self, x: usize, y: usize) -> Point2<F> {
        let half = F::from(0.5).unwrap();
        Point2::new(
            self.origin.x + F::from(x).unwrap() * self.cell_size + half * self.cell_size,
            self.origin.y + F::from(y).unwrap() * self.cell_size + half * self.cell_size,
        )
    }

    /// Samples the SDF at a world-space point using bilinear interpolation.
    pub fn sample(&self, p: Point2<F>) -> F {
        let half = F::from(0.5).unwrap();

        // Convert to grid coordinates (cell centers)
        let gx = (p.x - self.origin.x) / self.cell_size - half;
        let gy = (p.y - self.origin.y) / self.cell_size - half;

        // Get integer cell coordinates
        let x0 = gx.floor().to_usize().unwrap_or(0);
        let y0 = gy.floor().to_usize().unwrap_or(0);
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        // Clamp x0, y0
        let x0 = x0.min(self.width - 1);
        let y0 = y0.min(self.height - 1);

        // Interpolation weights
        let fx = gx - F::from(x0).unwrap();
        let fy = gy - F::from(y0).unwrap();
        let fx = fx.max(F::zero()).min(F::one());
        let fy = fy.max(F::zero()).min(F::one());

        // Get corner values
        let v00 = self.values[y0 * self.width + x0];
        let v10 = self.values[y0 * self.width + x1];
        let v01 = self.values[y1 * self.width + x0];
        let v11 = self.values[y1 * self.width + x1];

        // Bilinear interpolation
        let one = F::one();
        let v0 = v00 * (one - fx) + v10 * fx;
        let v1 = v01 * (one - fx) + v11 * fx;
        v0 * (one - fy) + v1 * fy
    }

    /// Returns the value at grid coordinates.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> F {
        self.values[y * self.width + x]
    }

    /// Sets the value at grid coordinates.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: F) {
        self.values[y * self.width + x] = value;
    }
}

/// A circle for SDF computation.
pub struct Circle<F> {
    pub center: Point2<F>,
    pub radius: F,
}

impl<F: Float> Circle<F> {
    pub fn new(center: Point2<F>, radius: F) -> Self {
        Self { center, radius }
    }
}

impl<F: Float> Sdf2<F> for Circle<F> {
    fn signed_distance(&self, p: Point2<F>) -> F {
        sdf_circle(p, self.center, self.radius)
    }
}

/// A closed polygon for SDF computation.
pub struct Polygon<F> {
    pub vertices: Vec<Point2<F>>,
}

impl<F: Float> Polygon<F> {
    pub fn new(vertices: Vec<Point2<F>>) -> Self {
        Self { vertices }
    }
}

impl<F: Float> Sdf2<F> for Polygon<F> {
    fn signed_distance(&self, p: Point2<F>) -> F {
        sdf_polygon(p, &self.vertices)
    }
}

/// A 2D distance field that can be queried at any point.
///
/// This is a trait for objects that can compute distance to a shape.
pub trait DistanceField2<F: Float> {
    /// Returns the distance from point `p` to the nearest surface.
    fn distance_at(&self, p: Point2<F>) -> F;
}

impl<F: Float, S: Sdf2<F>> DistanceField2<F> for S {
    fn distance_at(&self, p: Point2<F>) -> F {
        self.signed_distance(p).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_point() {
        let target = Point2::new(5.0, 5.0);
        let p = Point2::new(8.0, 9.0); // Distance = 5.0

        let dist = sdf_point(p, target);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sdf_segment() {
        let seg = Segment2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0));

        // Point above middle of segment
        let p1 = Point2::new(5.0, 3.0);
        assert!((sdf_segment(p1, seg) - 3.0).abs() < 1e-10);

        // Point at endpoint
        let p2 = Point2::new(0.0, 0.0);
        assert!(sdf_segment(p2, seg) < 1e-10);

        // Point beyond endpoint
        let p3 = Point2::new(-3.0, 4.0);
        assert!((sdf_segment(p3, seg) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sdf_circle() {
        let center = Point2::new(0.0, 0.0);
        let radius = 5.0;

        // Outside
        let p1 = Point2::new(8.0, 0.0);
        assert!((sdf_circle(p1, center, radius) - 3.0).abs() < 1e-10);

        // Inside
        let p2 = Point2::new(2.0, 0.0);
        assert!((sdf_circle(p2, center, radius) - (-3.0)).abs() < 1e-10);

        // On boundary
        let p3 = Point2::new(5.0, 0.0);
        assert!(sdf_circle(p3, center, radius).abs() < 1e-10);
    }

    #[test]
    fn test_sdf_polygon_square() {
        // Unit square centered at origin
        let vertices = vec![
            Point2::new(-1.0, -1.0),
            Point2::new(1.0, -1.0),
            Point2::new(1.0, 1.0),
            Point2::new(-1.0, 1.0),
        ];

        // Center (inside)
        let p1 = Point2::new(0.0, 0.0);
        let d1 = sdf_polygon(p1, &vertices);
        assert!(d1 < 0.0, "Center should be inside: {}", d1);
        assert!((d1 - (-1.0)).abs() < 1e-10);

        // Outside
        let p2 = Point2::new(2.0, 0.0);
        let d2 = sdf_polygon(p2, &vertices);
        assert!(d2 > 0.0, "Outside point should be positive: {}", d2);
        assert!((d2 - 1.0).abs() < 1e-10);

        // On edge
        let p3 = Point2::new(1.0, 0.0);
        let d3 = sdf_polygon(p3, &vertices);
        assert!(d3.abs() < 1e-10, "Edge point should be ~0: {}", d3);
    }

    #[test]
    fn test_sdf_polygon_triangle() {
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(2.0, 3.0),
        ];

        // Centroid (inside)
        let centroid = Point2::new(2.0, 1.0);
        let d = sdf_polygon(centroid, &vertices);
        assert!(d < 0.0, "Centroid should be inside: {}", d);

        // Far outside
        let outside = Point2::new(10.0, 10.0);
        let d2 = sdf_polygon(outside, &vertices);
        assert!(d2 > 0.0, "Far point should be outside: {}", d2);
    }

    #[test]
    fn test_circle_sdf_trait() {
        let circle = Circle::new(Point2::new(0.0, 0.0), 5.0);

        assert!(circle.contains(Point2::new(0.0, 0.0)));
        assert!(circle.contains(Point2::new(3.0, 0.0)));
        assert!(!circle.contains(Point2::new(6.0, 0.0)));
    }

    #[test]
    fn test_polygon_sdf_trait() {
        let poly = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        assert!(poly.contains(Point2::new(5.0, 5.0)));
        assert!(!poly.contains(Point2::new(15.0, 5.0)));
    }

    #[test]
    fn test_sdf_grid_circle() {
        let circle = Circle::new(Point2::new(5.0, 5.0), 3.0);
        let grid = SdfGrid::from_shape(&circle, 20, 20, Point2::new(0.0, 0.0), 0.5);

        // Center should be inside (negative)
        let center_val = grid.sample(Point2::new(5.0, 5.0));
        assert!(center_val < 0.0);
        assert!((center_val - (-3.0)).abs() < 0.5); // Approximate due to grid

        // Far corner should be outside (positive)
        let corner_val = grid.sample(Point2::new(0.0, 0.0));
        assert!(corner_val > 0.0);
    }

    #[test]
    fn test_sdf_grid_sample_interpolation() {
        let circle = Circle::new(Point2::new(5.0, 5.0), 3.0);
        let grid = SdfGrid::from_shape(&circle, 100, 100, Point2::new(0.0, 0.0), 0.1);

        // Sample at a point and compare to actual SDF
        let p = Point2::new(7.5, 5.0);
        let grid_val = grid.sample(p);
        let actual_val = circle.signed_distance(p);

        // Should be close (within cell size)
        assert!(
            (grid_val - actual_val).abs() < 0.2,
            "Grid: {}, Actual: {}",
            grid_val,
            actual_val
        );
    }

    #[test]
    fn test_winding_number() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];

        // Inside
        assert_ne!(winding_number(Point2::new(5.0, 5.0), &square), 0);

        // Outside
        assert_eq!(winding_number(Point2::new(15.0, 5.0), &square), 0);
        assert_eq!(winding_number(Point2::new(-5.0, 5.0), &square), 0);
        assert_eq!(winding_number(Point2::new(5.0, -5.0), &square), 0);
    }
}
