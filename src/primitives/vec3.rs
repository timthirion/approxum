//! 3D vector type for directions and offsets.

use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 3D vector representing a direction or offset.
///
/// Generic over floating-point types (`f32` or `f64`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3<F> {
    pub x: F,
    pub y: F,
    pub z: F,
}

impl<F: Float> Vec3<F> {
    /// Creates a new vector.
    #[inline]
    pub fn new(x: F, y: F, z: F) -> Self {
        Self { x, y, z }
    }

    /// Creates a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self {
            x: F::zero(),
            y: F::zero(),
            z: F::zero(),
        }
    }

    /// Creates a unit vector along the X axis.
    #[inline]
    pub fn unit_x() -> Self {
        Self {
            x: F::one(),
            y: F::zero(),
            z: F::zero(),
        }
    }

    /// Creates a unit vector along the Y axis.
    #[inline]
    pub fn unit_y() -> Self {
        Self {
            x: F::zero(),
            y: F::one(),
            z: F::zero(),
        }
    }

    /// Creates a unit vector along the Z axis.
    #[inline]
    pub fn unit_z() -> Self {
        Self {
            x: F::zero(),
            y: F::zero(),
            z: F::one(),
        }
    }

    /// Computes the dot product with another vector.
    #[inline]
    pub fn dot(self, other: Self) -> F {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes the cross product with another vector.
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns the squared magnitude (length squared).
    #[inline]
    pub fn magnitude_squared(self) -> F {
        self.dot(self)
    }

    /// Returns the magnitude (length) of the vector.
    #[inline]
    pub fn magnitude(self) -> F {
        self.magnitude_squared().sqrt()
    }

    /// Returns a normalized (unit length) vector.
    ///
    /// Returns `None` if the vector is zero or too small to normalize reliably.
    #[inline]
    pub fn normalize(self) -> Option<Self> {
        let mag = self.magnitude();
        if mag > F::epsilon() {
            Some(self / mag)
        } else {
            None
        }
    }

    /// Linearly interpolates between `self` and `other`.
    ///
    /// When `t = 0`, returns `self`. When `t = 1`, returns `other`.
    #[inline]
    pub fn lerp(self, other: Self, t: F) -> Self {
        self + (other - self) * t
    }
}

impl<F: Float> Add for Vec3<F> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<F: Float> Sub for Vec3<F> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<F: Float> Mul<F> for Vec3<F> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: F) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl<F: Float> Div<F> for Vec3<F> {
    type Output = Self;

    #[inline]
    fn div(self, scalar: F) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl<F: Float> Neg for Vec3<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<F: Float> Default for Vec3<F> {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new() {
        let v: Vec3<f64> = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_dot_product() {
        let a: Vec3<f64> = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_cross_product() {
        let x: Vec3<f64> = Vec3::unit_x();
        let y = Vec3::unit_y();
        let z = x.cross(y);
        assert_relative_eq!(z.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(z.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(z.z, 1.0, epsilon = 1e-10);

        // Cross product is anti-commutative
        let z_rev = y.cross(x);
        assert_relative_eq!(z_rev.z, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_magnitude() {
        let v: Vec3<f64> = Vec3::new(2.0, 3.0, 6.0);
        assert_eq!(v.magnitude_squared(), 49.0);
        assert_eq!(v.magnitude(), 7.0);
    }

    #[test]
    fn test_normalize() {
        let v: Vec3<f64> = Vec3::new(2.0, 3.0, 6.0);
        let n = v.normalize().unwrap();
        assert_relative_eq!(n.magnitude(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(n.x, 2.0 / 7.0, epsilon = 1e-10);
        assert_relative_eq!(n.y, 3.0 / 7.0, epsilon = 1e-10);
        assert_relative_eq!(n.z, 6.0 / 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let v: Vec3<f64> = Vec3::zero();
        assert!(v.normalize().is_none());
    }

    #[test]
    fn test_lerp() {
        let a: Vec3<f64> = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 20.0, 30.0);
        let mid = a.lerp(b, 0.5);
        assert_eq!(mid.x, 5.0);
        assert_eq!(mid.y, 10.0);
        assert_eq!(mid.z, 15.0);
    }

    #[test]
    fn test_arithmetic() {
        let a: Vec3<f64> = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);

        let sum = a + b;
        assert_eq!(sum.x, 5.0);
        assert_eq!(sum.y, 7.0);
        assert_eq!(sum.z, 9.0);

        let diff = b - a;
        assert_eq!(diff.x, 3.0);
        assert_eq!(diff.y, 3.0);
        assert_eq!(diff.z, 3.0);

        let scaled = a * 2.0;
        assert_eq!(scaled.x, 2.0);
        assert_eq!(scaled.y, 4.0);
        assert_eq!(scaled.z, 6.0);

        let divided = b / 2.0;
        assert_eq!(divided.x, 2.0);
        assert_eq!(divided.y, 2.5);
        assert_eq!(divided.z, 3.0);

        let neg = -a;
        assert_eq!(neg.x, -1.0);
        assert_eq!(neg.y, -2.0);
        assert_eq!(neg.z, -3.0);
    }
}
