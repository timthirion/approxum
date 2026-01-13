//! 2D vector type for directions and offsets.

use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 2D vector representing a direction or offset.
///
/// Generic over floating-point types (`f32` or `f64`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2<F> {
    pub x: F,
    pub y: F,
}

impl<F: Float> Vec2<F> {
    /// Creates a new vector.
    #[inline]
    pub fn new(x: F, y: F) -> Self {
        Self { x, y }
    }

    /// Creates a zero vector.
    #[inline]
    pub fn zero() -> Self {
        Self {
            x: F::zero(),
            y: F::zero(),
        }
    }

    /// Creates a unit vector along the X axis.
    #[inline]
    pub fn unit_x() -> Self {
        Self {
            x: F::one(),
            y: F::zero(),
        }
    }

    /// Creates a unit vector along the Y axis.
    #[inline]
    pub fn unit_y() -> Self {
        Self {
            x: F::zero(),
            y: F::one(),
        }
    }

    /// Computes the dot product with another vector.
    #[inline]
    pub fn dot(self, other: Self) -> F {
        self.x * other.x + self.y * other.y
    }

    /// Computes the 2D cross product (perpendicular dot product).
    ///
    /// Returns the z-component of the 3D cross product if the vectors
    /// were extended to 3D with z=0. Positive means `other` is counter-clockwise
    /// from `self`.
    #[inline]
    pub fn cross(self, other: Self) -> F {
        self.x * other.y - self.y * other.x
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

    /// Returns a vector perpendicular to this one (rotated 90 degrees counter-clockwise).
    #[inline]
    pub fn perpendicular(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
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

impl<F: Float> Add for Vec2<F> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<F: Float> Sub for Vec2<F> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<F: Float> Mul<F> for Vec2<F> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: F) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl<F: Float> Div<F> for Vec2<F> {
    type Output = Self;

    #[inline]
    fn div(self, scalar: F) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
        }
    }
}

impl<F: Float> Neg for Vec2<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl<F: Float> Default for Vec2<F> {
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
        let v: Vec2<f64> = Vec2::new(3.0, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
    }

    #[test]
    fn test_dot_product() {
        let a: Vec2<f64> = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a.dot(b), 11.0);
    }

    #[test]
    fn test_cross_product() {
        let a: Vec2<f64> = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert_eq!(a.cross(b), 1.0);
        assert_eq!(b.cross(a), -1.0);
    }

    #[test]
    fn test_magnitude() {
        let v: Vec2<f64> = Vec2::new(3.0, 4.0);
        assert_eq!(v.magnitude_squared(), 25.0);
        assert_eq!(v.magnitude(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let v: Vec2<f64> = Vec2::new(3.0, 4.0);
        let n = v.normalize().unwrap();
        assert_relative_eq!(n.magnitude(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(n.x, 0.6, epsilon = 1e-10);
        assert_relative_eq!(n.y, 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let v: Vec2<f64> = Vec2::zero();
        assert!(v.normalize().is_none());
    }

    #[test]
    fn test_perpendicular() {
        let v: Vec2<f64> = Vec2::new(1.0, 0.0);
        let p = v.perpendicular();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 1.0);
        assert_eq!(v.dot(p), 0.0);
    }

    #[test]
    fn test_lerp() {
        let a: Vec2<f64> = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 20.0);
        let mid = a.lerp(b, 0.5);
        assert_eq!(mid.x, 5.0);
        assert_eq!(mid.y, 10.0);
    }

    #[test]
    fn test_arithmetic() {
        let a: Vec2<f64> = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);

        let sum = a + b;
        assert_eq!(sum.x, 4.0);
        assert_eq!(sum.y, 6.0);

        let diff = b - a;
        assert_eq!(diff.x, 2.0);
        assert_eq!(diff.y, 2.0);

        let scaled = a * 2.0;
        assert_eq!(scaled.x, 2.0);
        assert_eq!(scaled.y, 4.0);

        let divided = b / 2.0;
        assert_eq!(divided.x, 1.5);
        assert_eq!(divided.y, 2.0);

        let neg = -a;
        assert_eq!(neg.x, -1.0);
        assert_eq!(neg.y, -2.0);
    }
}
