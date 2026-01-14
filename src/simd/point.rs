//! SIMD point and vector types.
//!
//! These types represent batches of 4 or 8 2D points/vectors for parallel processing.

use wide::{f32x4, f32x8};

use crate::primitives::{Point2, Vec2};

/// A batch of 4 2D vectors using SIMD.
#[derive(Clone, Copy, Debug)]
pub struct Vec2x4 {
    pub x: f32x4,
    pub y: f32x4,
}

impl Vec2x4 {
    /// Creates a new batch of vectors.
    #[inline]
    pub fn new(x: f32x4, y: f32x4) -> Self {
        Self { x, y }
    }

    /// Creates a batch from 4 individual vectors.
    #[inline]
    pub fn from_vecs(v0: Vec2<f32>, v1: Vec2<f32>, v2: Vec2<f32>, v3: Vec2<f32>) -> Self {
        Self {
            x: f32x4::new([v0.x, v1.x, v2.x, v3.x]),
            y: f32x4::new([v0.y, v1.y, v2.y, v3.y]),
        }
    }

    /// Creates a batch where all 4 vectors are the same.
    #[inline]
    pub fn splat(v: Vec2<f32>) -> Self {
        Self {
            x: f32x4::splat(v.x),
            y: f32x4::splat(v.y),
        }
    }

    /// Computes the dot product of each vector pair.
    #[inline]
    pub fn dot(self, other: Self) -> f32x4 {
        self.x * other.x + self.y * other.y
    }

    /// Computes the squared length of each vector.
    #[inline]
    pub fn length_squared(self) -> f32x4 {
        self.dot(self)
    }

    /// Computes the length of each vector.
    #[inline]
    pub fn length(self) -> f32x4 {
        self.length_squared().sqrt()
    }

    /// Extracts the 4 vectors as an array.
    #[inline]
    pub fn to_array(self) -> [Vec2<f32>; 4] {
        let x = self.x.to_array();
        let y = self.y.to_array();
        [
            Vec2::new(x[0], y[0]),
            Vec2::new(x[1], y[1]),
            Vec2::new(x[2], y[2]),
            Vec2::new(x[3], y[3]),
        ]
    }
}

impl std::ops::Add for Vec2x4 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Vec2x4 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Mul<f32x4> for Vec2x4 {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f32x4) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

/// A batch of 4 2D points using SIMD.
#[derive(Clone, Copy, Debug)]
pub struct Point2x4 {
    pub x: f32x4,
    pub y: f32x4,
}

impl Point2x4 {
    /// Creates a new batch of points.
    #[inline]
    pub fn new(x: f32x4, y: f32x4) -> Self {
        Self { x, y }
    }

    /// Creates a batch from 4 individual points.
    #[inline]
    pub fn from_points(p0: Point2<f32>, p1: Point2<f32>, p2: Point2<f32>, p3: Point2<f32>) -> Self {
        Self {
            x: f32x4::new([p0.x, p1.x, p2.x, p3.x]),
            y: f32x4::new([p0.y, p1.y, p2.y, p3.y]),
        }
    }

    /// Creates a batch from a slice of points (must have at least 4 points).
    ///
    /// # Panics
    ///
    /// Panics if the slice has fewer than 4 points.
    #[inline]
    pub fn from_slice(points: &[Point2<f32>]) -> Self {
        Self {
            x: f32x4::new([points[0].x, points[1].x, points[2].x, points[3].x]),
            y: f32x4::new([points[0].y, points[1].y, points[2].y, points[3].y]),
        }
    }

    /// Creates a batch where all 4 points are the same.
    #[inline]
    pub fn splat(p: Point2<f32>) -> Self {
        Self {
            x: f32x4::splat(p.x),
            y: f32x4::splat(p.y),
        }
    }

    /// Computes the squared distance from each point to a single target point.
    #[inline]
    pub fn distance_squared_to(self, target: Point2<f32>) -> f32x4 {
        let dx = self.x - f32x4::splat(target.x);
        let dy = self.y - f32x4::splat(target.y);
        dx * dx + dy * dy
    }

    /// Computes the distance from each point to a single target point.
    #[inline]
    pub fn distance_to(self, target: Point2<f32>) -> f32x4 {
        self.distance_squared_to(target).sqrt()
    }

    /// Computes the squared distance between corresponding points in two batches.
    #[inline]
    pub fn distance_squared_to_batch(self, other: Self) -> f32x4 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Computes the distance between corresponding points in two batches.
    #[inline]
    pub fn distance_to_batch(self, other: Self) -> f32x4 {
        self.distance_squared_to_batch(other).sqrt()
    }

    /// Linear interpolation between two batches of points.
    #[inline]
    pub fn lerp(self, other: Self, t: f32x4) -> Self {
        let one_minus_t = f32x4::splat(1.0) - t;
        Self {
            x: self.x * one_minus_t + other.x * t,
            y: self.y * one_minus_t + other.y * t,
        }
    }

    /// Extracts the 4 points as an array.
    #[inline]
    pub fn to_array(self) -> [Point2<f32>; 4] {
        let x = self.x.to_array();
        let y = self.y.to_array();
        [
            Point2::new(x[0], y[0]),
            Point2::new(x[1], y[1]),
            Point2::new(x[2], y[2]),
            Point2::new(x[3], y[3]),
        ]
    }
}

impl std::ops::Sub for Point2x4 {
    type Output = Vec2x4;

    #[inline]
    fn sub(self, other: Self) -> Vec2x4 {
        Vec2x4 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Add<Vec2x4> for Point2x4 {
    type Output = Self;

    #[inline]
    fn add(self, v: Vec2x4) -> Self {
        Self {
            x: self.x + v.x,
            y: self.y + v.y,
        }
    }
}

// ============================================================================
// 8-wide versions for AVX
// ============================================================================

/// A batch of 8 2D vectors using SIMD (AVX).
#[derive(Clone, Copy, Debug)]
pub struct Vec2x8 {
    pub x: f32x8,
    pub y: f32x8,
}

impl Vec2x8 {
    /// Creates a new batch of vectors.
    #[inline]
    pub fn new(x: f32x8, y: f32x8) -> Self {
        Self { x, y }
    }

    /// Creates a batch where all 8 vectors are the same.
    #[inline]
    pub fn splat(v: Vec2<f32>) -> Self {
        Self {
            x: f32x8::splat(v.x),
            y: f32x8::splat(v.y),
        }
    }

    /// Computes the dot product of each vector pair.
    #[inline]
    pub fn dot(self, other: Self) -> f32x8 {
        self.x * other.x + self.y * other.y
    }

    /// Computes the squared length of each vector.
    #[inline]
    pub fn length_squared(self) -> f32x8 {
        self.dot(self)
    }

    /// Computes the length of each vector.
    #[inline]
    pub fn length(self) -> f32x8 {
        self.length_squared().sqrt()
    }
}

impl std::ops::Add for Vec2x8 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Vec2x8 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Mul<f32x8> for Vec2x8 {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f32x8) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

/// A batch of 8 2D points using SIMD (AVX).
#[derive(Clone, Copy, Debug)]
pub struct Point2x8 {
    pub x: f32x8,
    pub y: f32x8,
}

impl Point2x8 {
    /// Creates a new batch of points.
    #[inline]
    pub fn new(x: f32x8, y: f32x8) -> Self {
        Self { x, y }
    }

    /// Creates a batch from a slice of points (must have at least 8 points).
    ///
    /// # Panics
    ///
    /// Panics if the slice has fewer than 8 points.
    #[inline]
    pub fn from_slice(points: &[Point2<f32>]) -> Self {
        Self {
            x: f32x8::new([
                points[0].x,
                points[1].x,
                points[2].x,
                points[3].x,
                points[4].x,
                points[5].x,
                points[6].x,
                points[7].x,
            ]),
            y: f32x8::new([
                points[0].y,
                points[1].y,
                points[2].y,
                points[3].y,
                points[4].y,
                points[5].y,
                points[6].y,
                points[7].y,
            ]),
        }
    }

    /// Creates a batch where all 8 points are the same.
    #[inline]
    pub fn splat(p: Point2<f32>) -> Self {
        Self {
            x: f32x8::splat(p.x),
            y: f32x8::splat(p.y),
        }
    }

    /// Computes the squared distance from each point to a single target point.
    #[inline]
    pub fn distance_squared_to(self, target: Point2<f32>) -> f32x8 {
        let dx = self.x - f32x8::splat(target.x);
        let dy = self.y - f32x8::splat(target.y);
        dx * dx + dy * dy
    }

    /// Computes the distance from each point to a single target point.
    #[inline]
    pub fn distance_to(self, target: Point2<f32>) -> f32x8 {
        self.distance_squared_to(target).sqrt()
    }

    /// Linear interpolation between two batches of points.
    #[inline]
    pub fn lerp(self, other: Self, t: f32x8) -> Self {
        let one_minus_t = f32x8::splat(1.0) - t;
        Self {
            x: self.x * one_minus_t + other.x * t,
            y: self.y * one_minus_t + other.y * t,
        }
    }

    /// Extracts the 8 points as an array.
    #[inline]
    pub fn to_array(self) -> [Point2<f32>; 8] {
        let x = self.x.to_array();
        let y = self.y.to_array();
        [
            Point2::new(x[0], y[0]),
            Point2::new(x[1], y[1]),
            Point2::new(x[2], y[2]),
            Point2::new(x[3], y[3]),
            Point2::new(x[4], y[4]),
            Point2::new(x[5], y[5]),
            Point2::new(x[6], y[6]),
            Point2::new(x[7], y[7]),
        ]
    }
}

impl std::ops::Sub for Point2x8 {
    type Output = Vec2x8;

    #[inline]
    fn sub(self, other: Self) -> Vec2x8 {
        Vec2x8 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Add<Vec2x8> for Point2x8 {
    type Output = Self;

    #[inline]
    fn add(self, v: Vec2x8) -> Self {
        Self {
            x: self.x + v.x,
            y: self.y + v.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2x4_distance() {
        let points = Point2x4::from_points(
            Point2::new(0.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(0.0, 4.0),
            Point2::new(3.0, 4.0),
        );

        let target = Point2::new(0.0, 0.0);
        let distances = points.distance_to(target).to_array();

        assert!((distances[0] - 0.0).abs() < 1e-6);
        assert!((distances[1] - 3.0).abs() < 1e-6);
        assert!((distances[2] - 4.0).abs() < 1e-6);
        assert!((distances[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_point2x4_lerp() {
        let a = Point2x4::splat(Point2::new(0.0, 0.0));
        let b = Point2x4::splat(Point2::new(10.0, 10.0));
        let t = f32x4::splat(0.5);

        let result = a.lerp(b, t).to_array();

        for p in &result {
            assert!((p.x - 5.0).abs() < 1e-6);
            assert!((p.y - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vec2x4_dot() {
        let a = Vec2x4::from_vecs(
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 3.0),
        );
        let b = Vec2x4::from_vecs(
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(4.0, 5.0),
        );

        let dots = a.dot(b).to_array();

        assert!((dots[0] - 1.0).abs() < 1e-6);
        assert!((dots[1] - 1.0).abs() < 1e-6);
        assert!((dots[2] - 2.0).abs() < 1e-6);
        assert!((dots[3] - 23.0).abs() < 1e-6); // 2*4 + 3*5
    }

    #[test]
    fn test_vec2x4_length() {
        let v = Vec2x4::from_vecs(
            Vec2::new(3.0, 4.0),
            Vec2::new(5.0, 12.0),
            Vec2::new(8.0, 15.0),
            Vec2::new(1.0, 0.0),
        );

        let lengths = v.length().to_array();

        assert!((lengths[0] - 5.0).abs() < 1e-6);
        assert!((lengths[1] - 13.0).abs() < 1e-6);
        assert!((lengths[2] - 17.0).abs() < 1e-6);
        assert!((lengths[3] - 1.0).abs() < 1e-6);
    }
}
