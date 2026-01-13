//! 3D point type.

use super::Vec3;
use num_traits::Float;
use std::ops::{Add, Sub};

/// A 3D point with x, y, and z coordinates.
///
/// Generic over floating-point types (`f32` or `f64`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3<F> {
    pub x: F,
    pub y: F,
    pub z: F,
}

impl<F: Float> Point3<F> {
    /// Creates a new point.
    #[inline]
    pub fn new(x: F, y: F, z: F) -> Self {
        Self { x, y, z }
    }

    /// Creates a point at the origin (0, 0, 0).
    #[inline]
    pub fn origin() -> Self {
        Self {
            x: F::zero(),
            y: F::zero(),
            z: F::zero(),
        }
    }

    /// Computes the squared distance to another point.
    #[inline]
    pub fn distance_squared(self, other: Self) -> F {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dz = other.z - self.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Computes the Euclidean distance to another point.
    #[inline]
    pub fn distance(self, other: Self) -> F {
        self.distance_squared(other).sqrt()
    }

    /// Linearly interpolates between `self` and `other`.
    ///
    /// When `t = 0`, returns `self`. When `t = 1`, returns `other`.
    #[inline]
    pub fn lerp(self, other: Self, t: F) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    /// Returns the midpoint between `self` and `other`.
    #[inline]
    pub fn midpoint(self, other: Self) -> Self {
        let two = F::one() + F::one();
        Self {
            x: (self.x + other.x) / two,
            y: (self.y + other.y) / two,
            z: (self.z + other.z) / two,
        }
    }

    /// Converts this point to a vector from the origin.
    #[inline]
    pub fn to_vec(self) -> Vec3<F> {
        Vec3::new(self.x, self.y, self.z)
    }
}

// Point - Point = Vec3
impl<F: Float> Sub for Point3<F> {
    type Output = Vec3<F>;

    #[inline]
    fn sub(self, other: Self) -> Vec3<F> {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

// Point + Vec3 = Point
impl<F: Float> Add<Vec3<F>> for Point3<F> {
    type Output = Self;

    #[inline]
    fn add(self, v: Vec3<F>) -> Self {
        Self {
            x: self.x + v.x,
            y: self.y + v.y,
            z: self.z + v.z,
        }
    }
}

// Point - Vec3 = Point
impl<F: Float> Sub<Vec3<F>> for Point3<F> {
    type Output = Self;

    #[inline]
    fn sub(self, v: Vec3<F>) -> Self {
        Self {
            x: self.x - v.x,
            y: self.y - v.y,
            z: self.z - v.z,
        }
    }
}

impl<F: Float> Default for Point3<F> {
    fn default() -> Self {
        Self::origin()
    }
}

impl<F: Float> From<Vec3<F>> for Point3<F> {
    fn from(v: Vec3<F>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let p: Point3<f64> = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_origin() {
        let p: Point3<f64> = Point3::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
        assert_eq!(p.z, 0.0);
    }

    #[test]
    fn test_distance() {
        let a: Point3<f64> = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(2.0, 3.0, 6.0);
        assert_eq!(a.distance_squared(b), 49.0);
        assert_eq!(a.distance(b), 7.0);
    }

    #[test]
    fn test_lerp() {
        let a: Point3<f64> = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 20.0, 30.0);

        let start = a.lerp(b, 0.0);
        assert_eq!(start.x, 0.0);
        assert_eq!(start.y, 0.0);
        assert_eq!(start.z, 0.0);

        let end = a.lerp(b, 1.0);
        assert_eq!(end.x, 10.0);
        assert_eq!(end.y, 20.0);
        assert_eq!(end.z, 30.0);

        let mid = a.lerp(b, 0.5);
        assert_eq!(mid.x, 5.0);
        assert_eq!(mid.y, 10.0);
        assert_eq!(mid.z, 15.0);
    }

    #[test]
    fn test_midpoint() {
        let a: Point3<f64> = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 20.0, 30.0);
        let m = a.midpoint(b);
        assert_eq!(m.x, 5.0);
        assert_eq!(m.y, 10.0);
        assert_eq!(m.z, 15.0);
    }

    #[test]
    fn test_point_sub_point() {
        let a: Point3<f64> = Point3::new(1.0, 2.0, 3.0);
        let b = Point3::new(4.0, 6.0, 9.0);
        let v: Vec3<f64> = b - a;
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
        assert_eq!(v.z, 6.0);
    }

    #[test]
    fn test_point_add_vec() {
        let p: Point3<f64> = Point3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(3.0, 4.0, 5.0);
        let result = p + v;
        assert_eq!(result.x, 4.0);
        assert_eq!(result.y, 6.0);
        assert_eq!(result.z, 8.0);
    }

    #[test]
    fn test_point_sub_vec() {
        let p: Point3<f64> = Point3::new(4.0, 6.0, 8.0);
        let v = Vec3::new(3.0, 4.0, 5.0);
        let result = p - v;
        assert_eq!(result.x, 1.0);
        assert_eq!(result.y, 2.0);
        assert_eq!(result.z, 3.0);
    }

    #[test]
    fn test_to_vec() {
        let p: Point3<f64> = Point3::new(3.0, 4.0, 5.0);
        let v = p.to_vec();
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
        assert_eq!(v.z, 5.0);
    }
}
