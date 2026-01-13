//! Bridge between approxum (floating-point) and exactum (integer) geometry.
//!
//! This module provides conversions between approxum's floating-point types
//! and exactum's integer types. Enable with the `exactum` feature flag:
//!
//! ```toml
//! approxum = { version = "0.1", features = ["exactum"] }
//! ```
//!
//! # Conversion Strategy
//!
//! - **Exactum → Approxum**: Lossless conversion from integers to floats.
//! - **Approxum → Exactum**: Lossy conversion requiring a scale factor.
//!   Floating-point coordinates are multiplied by the scale, rounded, and
//!   converted to integers.
//!
//! # Example
//!
//! ```ignore
//! use approxum::Point2 as FloatPoint;
//! use exactum::Point2 as ExactPoint;
//! use approxum::bridge::snap_to_grid;
//!
//! // Float to integer (lossy)
//! let float_points = vec![
//!     FloatPoint::new(1.5, 2.7),
//!     FloatPoint::new(3.14159, 2.71828),
//! ];
//! let exact_points: Vec<ExactPoint<i64>> = snap_to_grid(&float_points, 1000.0);
//! // Results: [(1500, 2700), (3142, 2718)]
//!
//! // Integer to float (lossless)
//! let exact = ExactPoint::<i64>::new(1500, 2700);
//! let float: FloatPoint<f64> = exact.into();
//! // Result: (1500.0, 2700.0)
//! ```

use crate::primitives::{Point2, Point3, Vec2, Vec3};

/// Converts floating-point points to integer points by snapping to a grid.
///
/// Each coordinate is multiplied by `scale`, rounded to the nearest integer,
/// and converted to the target integer type.
///
/// # Arguments
///
/// * `points` - Slice of floating-point points
/// * `scale` - Scale factor (e.g., 1000.0 means 3 decimal places of precision)
///
/// # Example
///
/// ```ignore
/// use approxum::Point2;
/// use approxum::bridge::snap_to_grid;
///
/// let float_points = vec![
///     Point2::new(1.234, 5.678),
///     Point2::new(0.001, 0.999),
/// ];
///
/// // Snap to grid with scale 1000 (millimeter precision if units are meters)
/// let int_points: Vec<exactum::Point2<i64>> = snap_to_grid(&float_points, 1000.0);
/// ```
pub fn snap_to_grid(points: &[Point2<f64>], scale: f64) -> Vec<exactum::Point2<i64>> {
    points
        .iter()
        .map(|p| {
            exactum::Point2::new(
                (p.x * scale).round() as i64,
                (p.y * scale).round() as i64,
            )
        })
        .collect()
}

/// Converts floating-point 3D points to integer points by snapping to a grid.
pub fn snap_to_grid_3d(points: &[Point3<f64>], scale: f64) -> Vec<exactum::Point3<i64>> {
    points
        .iter()
        .map(|p| {
            exactum::Point3::new(
                (p.x * scale).round() as i64,
                (p.y * scale).round() as i64,
                (p.z * scale).round() as i64,
            )
        })
        .collect()
}

/// Converts floating-point points to i32 integer points.
///
/// Use this when coordinates are known to fit in i32 range after scaling.
pub fn snap_to_grid_i32(points: &[Point2<f64>], scale: f64) -> Vec<exactum::Point2<i32>> {
    points
        .iter()
        .map(|p| {
            exactum::Point2::new(
                (p.x * scale).round() as i32,
                (p.y * scale).round() as i32,
            )
        })
        .collect()
}

/// Converts a single floating-point point to an integer point.
pub fn snap_point(p: Point2<f64>, scale: f64) -> exactum::Point2<i64> {
    exactum::Point2::new(
        (p.x * scale).round() as i64,
        (p.y * scale).round() as i64,
    )
}

/// Converts a single floating-point 3D point to an integer point.
pub fn snap_point_3d(p: Point3<f64>, scale: f64) -> exactum::Point3<i64> {
    exactum::Point3::new(
        (p.x * scale).round() as i64,
        (p.y * scale).round() as i64,
        (p.z * scale).round() as i64,
    )
}

/// Converts integer points back to floating-point points.
///
/// # Arguments
///
/// * `points` - Slice of integer points
/// * `scale` - The scale factor that was used when snapping (coordinates will be divided by this)
pub fn unsnap_from_grid(points: &[exactum::Point2<i64>], scale: f64) -> Vec<Point2<f64>> {
    let inv_scale = 1.0 / scale;
    points
        .iter()
        .map(|p| Point2::new(p.x as f64 * inv_scale, p.y as f64 * inv_scale))
        .collect()
}

/// Converts integer 3D points back to floating-point points.
pub fn unsnap_from_grid_3d(points: &[exactum::Point3<i64>], scale: f64) -> Vec<Point3<f64>> {
    let inv_scale = 1.0 / scale;
    points
        .iter()
        .map(|p| {
            Point3::new(
                p.x as f64 * inv_scale,
                p.y as f64 * inv_scale,
                p.z as f64 * inv_scale,
            )
        })
        .collect()
}

// ============================================================================
// From implementations: exactum → approxum (lossless)
// ============================================================================

impl From<exactum::Point2<i32>> for Point2<f64> {
    fn from(p: exactum::Point2<i32>) -> Self {
        Point2::new(p.x as f64, p.y as f64)
    }
}

impl From<exactum::Point2<i64>> for Point2<f64> {
    fn from(p: exactum::Point2<i64>) -> Self {
        Point2::new(p.x as f64, p.y as f64)
    }
}

impl From<exactum::Point2<i32>> for Point2<f32> {
    fn from(p: exactum::Point2<i32>) -> Self {
        Point2::new(p.x as f32, p.y as f32)
    }
}

impl From<exactum::Point2<i64>> for Point2<f32> {
    fn from(p: exactum::Point2<i64>) -> Self {
        Point2::new(p.x as f32, p.y as f32)
    }
}

impl From<exactum::Point3<i32>> for Point3<f64> {
    fn from(p: exactum::Point3<i32>) -> Self {
        Point3::new(p.x as f64, p.y as f64, p.z as f64)
    }
}

impl From<exactum::Point3<i64>> for Point3<f64> {
    fn from(p: exactum::Point3<i64>) -> Self {
        Point3::new(p.x as f64, p.y as f64, p.z as f64)
    }
}

impl From<exactum::Vector2<i32>> for Vec2<f64> {
    fn from(v: exactum::Vector2<i32>) -> Self {
        Vec2::new(v.x as f64, v.y as f64)
    }
}

impl From<exactum::Vector2<i64>> for Vec2<f64> {
    fn from(v: exactum::Vector2<i64>) -> Self {
        Vec2::new(v.x as f64, v.y as f64)
    }
}

impl From<exactum::Vector3<i32>> for Vec3<f64> {
    fn from(v: exactum::Vector3<i32>) -> Self {
        Vec3::new(v.x as f64, v.y as f64, v.z as f64)
    }
}

impl From<exactum::Vector3<i64>> for Vec3<f64> {
    fn from(v: exactum::Vector3<i64>) -> Self {
        Vec3::new(v.x as f64, v.y as f64, v.z as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_to_grid() {
        let float_points = vec![
            Point2::new(1.234, 5.678),
            Point2::new(-0.5, 0.5),
            Point2::new(0.0, 0.0),
        ];

        let int_points = snap_to_grid(&float_points, 1000.0);

        assert_eq!(int_points[0].x, 1234);
        assert_eq!(int_points[0].y, 5678);
        assert_eq!(int_points[1].x, -500);
        assert_eq!(int_points[1].y, 500);
        assert_eq!(int_points[2].x, 0);
        assert_eq!(int_points[2].y, 0);
    }

    #[test]
    fn test_snap_point() {
        let p = Point2::new(3.14159, 2.71828);
        let snapped = snap_point(p, 100.0);

        assert_eq!(snapped.x, 314);
        assert_eq!(snapped.y, 272);
    }

    #[test]
    fn test_unsnap_from_grid() {
        let int_points = vec![
            exactum::Point2::new(1234_i64, 5678),
            exactum::Point2::new(-500, 500),
        ];

        let float_points = unsnap_from_grid(&int_points, 1000.0);

        assert!((float_points[0].x - 1.234).abs() < 1e-10);
        assert!((float_points[0].y - 5.678).abs() < 1e-10);
        assert!((float_points[1].x - (-0.5)).abs() < 1e-10);
        assert!((float_points[1].y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip() {
        let original = vec![
            Point2::new(1.0, 2.0),
            Point2::new(3.5, 4.5),
            Point2::new(-1.25, -2.75),
        ];

        let scale = 100.0;
        let snapped = snap_to_grid(&original, scale);
        let recovered = unsnap_from_grid(&snapped, scale);

        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig.x - rec.x).abs() < 0.01);
            assert!((orig.y - rec.y).abs() < 0.01);
        }
    }

    #[test]
    fn test_from_exactum_point2_i64() {
        let exact = exactum::Point2::new(100_i64, 200);
        let float: Point2<f64> = exact.into();

        assert_eq!(float.x, 100.0);
        assert_eq!(float.y, 200.0);
    }

    #[test]
    fn test_from_exactum_point2_i32() {
        let exact = exactum::Point2::new(100_i32, 200);
        let float: Point2<f64> = exact.into();

        assert_eq!(float.x, 100.0);
        assert_eq!(float.y, 200.0);
    }

    #[test]
    fn test_from_exactum_vector2() {
        let exact = exactum::Vector2::new(10_i64, 20);
        let float: Vec2<f64> = exact.into();

        assert_eq!(float.x, 10.0);
        assert_eq!(float.y, 20.0);
    }

    #[test]
    fn test_snap_to_grid_3d() {
        let float_points = vec![Point3::new(1.5, 2.5, 3.5)];
        let int_points = snap_to_grid_3d(&float_points, 10.0);

        assert_eq!(int_points[0].x, 15);
        assert_eq!(int_points[0].y, 25);
        assert_eq!(int_points[0].z, 35);
    }

    #[test]
    fn test_from_exactum_point3() {
        let exact = exactum::Point3::new(1_i64, 2, 3);
        let float: Point3<f64> = exact.into();

        assert_eq!(float.x, 1.0);
        assert_eq!(float.y, 2.0);
        assert_eq!(float.z, 3.0);
    }
}
