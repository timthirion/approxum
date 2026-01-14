//! 2D affine transformation type.

use super::{Point2, Vec2};
use num_traits::Float;
use std::ops::Mul;

/// A 2D affine transformation matrix.
///
/// Represented as a 2x3 matrix in row-major order:
/// ```text
/// | a  b  tx |
/// | c  d  ty |
/// ```
///
/// Transforms are applied as:
/// - Point: `(a*x + b*y + tx, c*x + d*y + ty)`
/// - Vector: `(a*x + b*y, c*x + d*y)` (no translation)
///
/// # Example
///
/// ```
/// use approxum::primitives::{Affine2, Point2, Vec2};
/// use std::f64::consts::FRAC_PI_2;
///
/// // Rotate 90 degrees then translate
/// let transform: Affine2<f64> = Affine2::rotation(FRAC_PI_2)
///     .then_translate(Vec2::new(10.0, 0.0));
///
/// let point = transform.apply_point(Point2::new(1.0, 0.0));
/// // (1, 0) rotated 90° = (0, 1), then translated = (10, 1)
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2<F> {
    /// Matrix element [0,0] - x scale / rotation component
    pub a: F,
    /// Matrix element [0,1] - x shear / rotation component
    pub b: F,
    /// Matrix element [1,0] - y shear / rotation component
    pub c: F,
    /// Matrix element [1,1] - y scale / rotation component
    pub d: F,
    /// Translation x component
    pub tx: F,
    /// Translation y component
    pub ty: F,
}

impl<F: Float> Affine2<F> {
    /// Creates a new affine transform from matrix components.
    ///
    /// The matrix is:
    /// ```text
    /// | a  b  tx |
    /// | c  d  ty |
    /// ```
    #[inline]
    pub fn new(a: F, b: F, c: F, d: F, tx: F, ty: F) -> Self {
        Self { a, b, c, d, tx, ty }
    }

    /// Creates the identity transform (no change).
    #[inline]
    pub fn identity() -> Self {
        Self {
            a: F::one(),
            b: F::zero(),
            c: F::zero(),
            d: F::one(),
            tx: F::zero(),
            ty: F::zero(),
        }
    }

    /// Creates a translation transform.
    #[inline]
    pub fn translation(offset: Vec2<F>) -> Self {
        Self {
            a: F::one(),
            b: F::zero(),
            c: F::zero(),
            d: F::one(),
            tx: offset.x,
            ty: offset.y,
        }
    }

    /// Creates a translation transform from x and y components.
    #[inline]
    pub fn translate(tx: F, ty: F) -> Self {
        Self::translation(Vec2::new(tx, ty))
    }

    /// Creates a rotation transform around the origin.
    ///
    /// Angle is in radians, positive is counter-clockwise.
    #[inline]
    pub fn rotation(angle: F) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Self {
            a: cos_a,
            b: -sin_a,
            c: sin_a,
            d: cos_a,
            tx: F::zero(),
            ty: F::zero(),
        }
    }

    /// Creates a rotation transform around a specific point.
    pub fn rotation_around(angle: F, center: Point2<F>) -> Self {
        // Translate to origin, rotate, translate back
        Self::translation(Vec2::new(-center.x, -center.y))
            .then(Self::rotation(angle))
            .then_translate(Vec2::new(center.x, center.y))
    }

    /// Creates a uniform scaling transform around the origin.
    #[inline]
    pub fn scale(factor: F) -> Self {
        Self {
            a: factor,
            b: F::zero(),
            c: F::zero(),
            d: factor,
            tx: F::zero(),
            ty: F::zero(),
        }
    }

    /// Creates a non-uniform scaling transform around the origin.
    #[inline]
    pub fn scale_xy(sx: F, sy: F) -> Self {
        Self {
            a: sx,
            b: F::zero(),
            c: F::zero(),
            d: sy,
            tx: F::zero(),
            ty: F::zero(),
        }
    }

    /// Creates a scaling transform around a specific point.
    pub fn scale_around(factor: F, center: Point2<F>) -> Self {
        Self::translation(Vec2::new(-center.x, -center.y))
            .then(Self::scale(factor))
            .then_translate(Vec2::new(center.x, center.y))
    }

    /// Creates a non-uniform scaling transform around a specific point.
    pub fn scale_xy_around(sx: F, sy: F, center: Point2<F>) -> Self {
        Self::translation(Vec2::new(-center.x, -center.y))
            .then(Self::scale_xy(sx, sy))
            .then_translate(Vec2::new(center.x, center.y))
    }

    /// Creates a horizontal shear transform.
    ///
    /// Points are shifted horizontally by `factor * y`.
    #[inline]
    pub fn shear_x(factor: F) -> Self {
        Self {
            a: F::one(),
            b: factor,
            c: F::zero(),
            d: F::one(),
            tx: F::zero(),
            ty: F::zero(),
        }
    }

    /// Creates a vertical shear transform.
    ///
    /// Points are shifted vertically by `factor * x`.
    #[inline]
    pub fn shear_y(factor: F) -> Self {
        Self {
            a: F::one(),
            b: F::zero(),
            c: factor,
            d: F::one(),
            tx: F::zero(),
            ty: F::zero(),
        }
    }

    /// Creates a reflection across the x-axis (y = 0).
    #[inline]
    pub fn reflect_x() -> Self {
        Self::scale_xy(F::one(), -F::one())
    }

    /// Creates a reflection across the y-axis (x = 0).
    #[inline]
    pub fn reflect_y() -> Self {
        Self::scale_xy(-F::one(), F::one())
    }

    /// Creates a reflection across the origin (point reflection).
    #[inline]
    pub fn reflect_origin() -> Self {
        Self::scale(-F::one())
    }

    /// Creates a reflection across an arbitrary line through the origin.
    ///
    /// The line direction doesn't need to be normalized.
    pub fn reflect_line(direction: Vec2<F>) -> Self {
        let len_sq = direction.x * direction.x + direction.y * direction.y;
        if len_sq < F::epsilon() {
            return Self::identity();
        }

        // Reflection matrix: R = 2 * (n ⊗ n) - I
        // where n is the unit normal to the line
        let nx = direction.x / len_sq.sqrt();
        let ny = direction.y / len_sq.sqrt();

        let two = F::from(2.0).unwrap();
        Self {
            a: two * nx * nx - F::one(),
            b: two * nx * ny,
            c: two * nx * ny,
            d: two * ny * ny - F::one(),
            tx: F::zero(),
            ty: F::zero(),
        }
    }

    /// Applies this transform to a point.
    #[inline]
    pub fn apply_point(&self, p: Point2<F>) -> Point2<F> {
        Point2::new(
            self.a * p.x + self.b * p.y + self.tx,
            self.c * p.x + self.d * p.y + self.ty,
        )
    }

    /// Applies this transform to a vector (no translation).
    #[inline]
    pub fn apply_vec(&self, v: Vec2<F>) -> Vec2<F> {
        Vec2::new(self.a * v.x + self.b * v.y, self.c * v.x + self.d * v.y)
    }

    /// Applies this transform to multiple points.
    pub fn apply_points(&self, points: &[Point2<F>]) -> Vec<Point2<F>> {
        points.iter().map(|p| self.apply_point(*p)).collect()
    }

    /// Composes this transform with another (self * other).
    ///
    /// The resulting transform applies `other` first, then `self`.
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
            tx: self.a * other.tx + self.b * other.ty + self.tx,
            ty: self.c * other.tx + self.d * other.ty + self.ty,
        }
    }

    /// Returns a transform that applies `self` first, then `other`.
    ///
    /// Equivalent to `other.compose(self)`.
    #[inline]
    pub fn then(&self, other: Self) -> Self {
        other.compose(self)
    }

    /// Returns a transform that applies `self` first, then translates.
    #[inline]
    pub fn then_translate(&self, offset: Vec2<F>) -> Self {
        self.then(Self::translation(offset))
    }

    /// Returns a transform that applies `self` first, then rotates.
    #[inline]
    pub fn then_rotate(&self, angle: F) -> Self {
        self.then(Self::rotation(angle))
    }

    /// Returns a transform that applies `self` first, then scales.
    #[inline]
    pub fn then_scale(&self, factor: F) -> Self {
        self.then(Self::scale(factor))
    }

    /// Returns the determinant of the linear part.
    ///
    /// - Positive: preserves orientation
    /// - Negative: flips orientation (reflection)
    /// - Zero: singular (collapses to line or point)
    #[inline]
    pub fn determinant(&self) -> F {
        self.a * self.d - self.b * self.c
    }

    /// Returns true if this transform preserves orientation.
    #[inline]
    pub fn preserves_orientation(&self) -> bool {
        self.determinant() > F::zero()
    }

    /// Returns true if this transform is invertible.
    #[inline]
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > F::epsilon()
    }

    /// Returns the inverse transform, if it exists.
    ///
    /// Returns `None` if the transform is singular (determinant is zero).
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < F::epsilon() {
            return None;
        }

        let inv_det = F::one() / det;
        Some(Self {
            a: self.d * inv_det,
            b: -self.b * inv_det,
            c: -self.c * inv_det,
            d: self.a * inv_det,
            tx: (self.b * self.ty - self.d * self.tx) * inv_det,
            ty: (self.c * self.tx - self.a * self.ty) * inv_det,
        })
    }

    /// Returns only the translation component.
    #[inline]
    pub fn translation_component(&self) -> Vec2<F> {
        Vec2::new(self.tx, self.ty)
    }

    /// Returns true if this is approximately the identity transform.
    pub fn is_identity(&self, epsilon: F) -> bool {
        (self.a - F::one()).abs() < epsilon
            && self.b.abs() < epsilon
            && self.c.abs() < epsilon
            && (self.d - F::one()).abs() < epsilon
            && self.tx.abs() < epsilon
            && self.ty.abs() < epsilon
    }

    /// Returns true if this transform has only translation (no rotation/scale/shear).
    pub fn is_translation_only(&self, epsilon: F) -> bool {
        (self.a - F::one()).abs() < epsilon
            && self.b.abs() < epsilon
            && self.c.abs() < epsilon
            && (self.d - F::one()).abs() < epsilon
    }

    /// Returns true if this is a rigid transform (rotation + translation only).
    ///
    /// Rigid transforms preserve distances and angles.
    pub fn is_rigid(&self, epsilon: F) -> bool {
        // Check that the linear part is orthogonal with determinant ±1
        let det = self.determinant();
        if (det.abs() - F::one()).abs() > epsilon {
            return false;
        }

        // Check orthogonality: columns should be unit length and perpendicular
        let col1_len_sq = self.a * self.a + self.c * self.c;
        let col2_len_sq = self.b * self.b + self.d * self.d;
        let dot = self.a * self.b + self.c * self.d;

        (col1_len_sq - F::one()).abs() < epsilon
            && (col2_len_sq - F::one()).abs() < epsilon
            && dot.abs() < epsilon
    }

    /// Returns true if this is a similarity transform (uniform scale + rotation + translation).
    ///
    /// Similarity transforms preserve angles but may change distances uniformly.
    pub fn is_similarity(&self, epsilon: F) -> bool {
        // For a similarity, the matrix should be: s * [cos -sin; sin cos]
        // This means: a = d and b = -c (or the reflection case: a = -d and b = c)
        let case1 = (self.a - self.d).abs() < epsilon && (self.b + self.c).abs() < epsilon;
        let case2 = (self.a + self.d).abs() < epsilon && (self.b - self.c).abs() < epsilon;
        case1 || case2
    }

    /// Extracts the rotation angle from the transform (assuming no shear).
    ///
    /// Returns the angle in radians.
    pub fn rotation_angle(&self) -> F {
        self.c.atan2(self.a)
    }

    /// Extracts the scale factors from the transform.
    ///
    /// Returns (scale_x, scale_y). For transforms with rotation,
    /// these are the singular values of the linear part.
    pub fn scale_factors(&self) -> (F, F) {
        let sx = (self.a * self.a + self.c * self.c).sqrt();
        let sy = (self.b * self.b + self.d * self.d).sqrt();
        (sx, sy)
    }

    /// Converts to a 3x3 matrix in row-major order.
    ///
    /// Returns [a, b, tx, c, d, ty, 0, 0, 1].
    pub fn to_matrix(&self) -> [F; 9] {
        [
            self.a,
            self.b,
            self.tx,
            self.c,
            self.d,
            self.ty,
            F::zero(),
            F::zero(),
            F::one(),
        ]
    }

    /// Creates from a 3x3 matrix in row-major order.
    ///
    /// The bottom row is ignored.
    pub fn from_matrix(m: &[F; 9]) -> Self {
        Self {
            a: m[0],
            b: m[1],
            tx: m[2],
            c: m[3],
            d: m[4],
            ty: m[5],
        }
    }

    /// Linearly interpolates between two transforms.
    ///
    /// Note: This does a component-wise lerp, which may not produce
    /// smooth rotation interpolation. For rotation, consider using
    /// separate angle interpolation.
    pub fn lerp(&self, other: &Self, t: F) -> Self {
        let one_minus_t = F::one() - t;
        Self {
            a: self.a * one_minus_t + other.a * t,
            b: self.b * one_minus_t + other.b * t,
            c: self.c * one_minus_t + other.c * t,
            d: self.d * one_minus_t + other.d * t,
            tx: self.tx * one_minus_t + other.tx * t,
            ty: self.ty * one_minus_t + other.ty * t,
        }
    }
}

impl<F: Float> Default for Affine2<F> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<F: Float> Mul for Affine2<F> {
    type Output = Self;

    /// Multiplies two transforms (self * rhs).
    ///
    /// The result applies `rhs` first, then `self`.
    fn mul(self, rhs: Self) -> Self {
        self.compose(&rhs)
    }
}

impl<F: Float> Mul<Point2<F>> for Affine2<F> {
    type Output = Point2<F>;

    fn mul(self, rhs: Point2<F>) -> Point2<F> {
        self.apply_point(rhs)
    }
}

impl<F: Float> Mul<Vec2<F>> for Affine2<F> {
    type Output = Vec2<F>;

    fn mul(self, rhs: Vec2<F>) -> Vec2<F> {
        self.apply_vec(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[test]
    fn test_identity() {
        let t: Affine2<f64> = Affine2::identity();
        let p = Point2::new(3.0, 4.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 3.0);
        assert_eq!(result.y, 4.0);
    }

    #[test]
    fn test_translation() {
        let t: Affine2<f64> = Affine2::translation(Vec2::new(10.0, 20.0));
        let p = Point2::new(3.0, 4.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 13.0);
        assert_eq!(result.y, 24.0);
    }

    #[test]
    fn test_translation_vec() {
        let t: Affine2<f64> = Affine2::translation(Vec2::new(10.0, 20.0));
        let v = Vec2::new(3.0, 4.0);
        let result = t.apply_vec(v);

        // Vectors are not affected by translation
        assert_eq!(result.x, 3.0);
        assert_eq!(result.y, 4.0);
    }

    #[test]
    fn test_rotation_90() {
        let t: Affine2<f64> = Affine2::rotation(FRAC_PI_2);
        let p = Point2::new(1.0, 0.0);
        let result = t.apply_point(p);

        assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_180() {
        let t: Affine2<f64> = Affine2::rotation(PI);
        let p = Point2::new(1.0, 0.0);
        let result = t.apply_point(p);

        assert_relative_eq!(result.x, -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_around() {
        let t: Affine2<f64> = Affine2::rotation_around(FRAC_PI_2, Point2::new(1.0, 1.0));
        let p = Point2::new(2.0, 1.0);
        let result = t.apply_point(p);

        // Rotating (2,1) 90° around (1,1) should give (1,2)
        assert_relative_eq!(result.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scale() {
        let t: Affine2<f64> = Affine2::scale(2.0);
        let p = Point2::new(3.0, 4.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 6.0);
        assert_eq!(result.y, 8.0);
    }

    #[test]
    fn test_scale_xy() {
        let t: Affine2<f64> = Affine2::scale_xy(2.0, 3.0);
        let p = Point2::new(4.0, 5.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 8.0);
        assert_eq!(result.y, 15.0);
    }

    #[test]
    fn test_scale_around() {
        let t: Affine2<f64> = Affine2::scale_around(2.0, Point2::new(1.0, 1.0));
        let p = Point2::new(2.0, 2.0);
        let result = t.apply_point(p);

        // (2,2) is (1,1) from center, scaled 2x = (2,2) from center = (3,3)
        assert_relative_eq!(result.x, 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shear_x() {
        let t: Affine2<f64> = Affine2::shear_x(1.0);
        let p = Point2::new(0.0, 2.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 2.0); // x += 1.0 * y
        assert_eq!(result.y, 2.0);
    }

    #[test]
    fn test_shear_y() {
        let t: Affine2<f64> = Affine2::shear_y(1.0);
        let p = Point2::new(2.0, 0.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 2.0);
        assert_eq!(result.y, 2.0); // y += 1.0 * x
    }

    #[test]
    fn test_reflect_x() {
        let t: Affine2<f64> = Affine2::reflect_x();
        let p = Point2::new(3.0, 4.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 3.0);
        assert_eq!(result.y, -4.0);
    }

    #[test]
    fn test_reflect_y() {
        let t: Affine2<f64> = Affine2::reflect_y();
        let p = Point2::new(3.0, 4.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, -3.0);
        assert_eq!(result.y, 4.0);
    }

    #[test]
    fn test_reflect_line() {
        // Reflect across y = x line
        let t: Affine2<f64> = Affine2::reflect_line(Vec2::new(1.0, 1.0));
        let p = Point2::new(1.0, 0.0);
        let result = t.apply_point(p);

        assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compose() {
        // Scale then translate
        let scale: Affine2<f64> = Affine2::scale(2.0);
        let translate = Affine2::translation(Vec2::new(10.0, 0.0));
        let composed = translate.compose(&scale); // translate * scale

        let p = Point2::new(1.0, 0.0);
        let result = composed.apply_point(p);

        // (1,0) scaled by 2 = (2,0), then translated = (12,0)
        assert_eq!(result.x, 12.0);
        assert_eq!(result.y, 0.0);
    }

    #[test]
    fn test_then() {
        let t: Affine2<f64> = Affine2::scale(2.0).then(Affine2::translation(Vec2::new(10.0, 0.0)));

        let p = Point2::new(1.0, 0.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 12.0);
        assert_eq!(result.y, 0.0);
    }

    #[test]
    fn test_then_translate() {
        let t: Affine2<f64> = Affine2::scale(2.0).then_translate(Vec2::new(10.0, 0.0));

        let p = Point2::new(1.0, 0.0);
        let result = t.apply_point(p);

        assert_eq!(result.x, 12.0);
    }

    #[test]
    fn test_then_rotate() {
        let t: Affine2<f64> = Affine2::translation(Vec2::new(1.0, 0.0)).then_rotate(FRAC_PI_2);

        let p = Point2::new(0.0, 0.0);
        let result = t.apply_point(p);

        // (0,0) translated to (1,0), then rotated 90° = (0,1)
        assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant() {
        let identity: Affine2<f64> = Affine2::identity();
        assert_relative_eq!(identity.determinant(), 1.0, epsilon = 1e-10);

        let scale = Affine2::scale(2.0);
        assert_relative_eq!(scale.determinant(), 4.0, epsilon = 1e-10);

        let reflect: Affine2<f64> = Affine2::reflect_x();
        assert_relative_eq!(reflect.determinant(), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse() {
        let t: Affine2<f64> = Affine2::rotation(FRAC_PI_4)
            .then_translate(Vec2::new(10.0, 20.0))
            .then_scale(2.0);

        let inv = t.inverse().unwrap();

        let p = Point2::new(3.0, 4.0);
        let transformed = t.apply_point(p);
        let back = inv.apply_point(transformed);

        assert_relative_eq!(back.x, p.x, epsilon = 1e-10);
        assert_relative_eq!(back.y, p.y, epsilon = 1e-10);
    }

    #[test]
    fn test_singular_no_inverse() {
        // Singular transform (scales y to 0)
        let t: Affine2<f64> = Affine2::scale_xy(1.0, 0.0);
        assert!(t.inverse().is_none());
    }

    #[test]
    fn test_is_identity() {
        let identity: Affine2<f64> = Affine2::identity();
        assert!(identity.is_identity(1e-10));

        let not_identity = Affine2::translation(Vec2::new(1.0, 0.0));
        assert!(!not_identity.is_identity(1e-10));
    }

    #[test]
    fn test_is_translation_only() {
        let t: Affine2<f64> = Affine2::translation(Vec2::new(5.0, 10.0));
        assert!(t.is_translation_only(1e-10));

        let not_trans = Affine2::rotation(0.1);
        assert!(!not_trans.is_translation_only(1e-10));
    }

    #[test]
    fn test_is_rigid() {
        let rigid: Affine2<f64> = Affine2::rotation(0.5).then_translate(Vec2::new(10.0, 20.0));
        assert!(rigid.is_rigid(1e-10));

        let not_rigid = Affine2::scale(2.0);
        assert!(!not_rigid.is_rigid(1e-10));
    }

    #[test]
    fn test_is_similarity() {
        let sim: Affine2<f64> = Affine2::rotation(0.5)
            .then_scale(2.0)
            .then_translate(Vec2::new(10.0, 20.0));
        assert!(sim.is_similarity(1e-10));

        let not_sim = Affine2::scale_xy(2.0, 3.0);
        assert!(!not_sim.is_similarity(1e-10));
    }

    #[test]
    fn test_rotation_angle() {
        let t: Affine2<f64> = Affine2::rotation(0.5);
        assert_relative_eq!(t.rotation_angle(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_scale_factors() {
        let t: Affine2<f64> = Affine2::scale_xy(2.0, 3.0);
        let (sx, sy) = t.scale_factors();

        assert_relative_eq!(sx, 2.0, epsilon = 1e-10);
        assert_relative_eq!(sy, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mul_operator() {
        let a: Affine2<f64> = Affine2::scale(2.0);
        let b = Affine2::translation(Vec2::new(10.0, 0.0));
        let c = a * b; // scale then translate

        let p = Point2::new(1.0, 0.0);
        let result = c.apply_point(p);

        // (1,0) with scale first: (1,0) -> (2,0) -> (12,0)
        // Note: a * b means apply b first, then a
        // So: translate(10,0) then scale(2) = (1+10)*2 = 22? No wait...
        // Actually compose means: (a * b)(p) = a(b(p))
        // So b is applied first: translate(1,0) = (11,0), then scale: (22,0)

        // Let me reconsider: compose(&self, other) applies other first, then self
        // a * b = a.compose(&b) = apply b first, then a
        // So: b(p) = (11, 0), then a(b(p)) = (22, 0)
        assert_eq!(result.x, 22.0);
    }

    #[test]
    fn test_mul_point() {
        let t: Affine2<f64> = Affine2::translation(Vec2::new(10.0, 20.0));
        let p = Point2::new(1.0, 2.0);
        let result = t * p;

        assert_eq!(result.x, 11.0);
        assert_eq!(result.y, 22.0);
    }

    #[test]
    fn test_mul_vec() {
        let t: Affine2<f64> = Affine2::scale(2.0);
        let v = Vec2::new(3.0, 4.0);
        let result = t * v;

        assert_eq!(result.x, 6.0);
        assert_eq!(result.y, 8.0);
    }

    #[test]
    fn test_apply_points() {
        let t: Affine2<f64> = Affine2::scale(2.0);
        let points = vec![
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        ];

        let result = t.apply_points(&points);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].x, 2.0);
        assert_eq!(result[1].y, 2.0);
        assert_eq!(result[2].x, 2.0);
        assert_eq!(result[2].y, 2.0);
    }

    #[test]
    fn test_lerp() {
        let a: Affine2<f64> = Affine2::identity();
        let b = Affine2::scale(2.0);

        let mid = a.lerp(&b, 0.5);
        let (sx, sy) = mid.scale_factors();

        assert_relative_eq!(sx, 1.5, epsilon = 1e-10);
        assert_relative_eq!(sy, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_to_from_matrix() {
        let t: Affine2<f64> = Affine2::rotation(0.5).then_translate(Vec2::new(10.0, 20.0));
        let matrix = t.to_matrix();
        let restored = Affine2::from_matrix(&matrix);

        assert_relative_eq!(t.a, restored.a, epsilon = 1e-10);
        assert_relative_eq!(t.b, restored.b, epsilon = 1e-10);
        assert_relative_eq!(t.c, restored.c, epsilon = 1e-10);
        assert_relative_eq!(t.d, restored.d, epsilon = 1e-10);
        assert_relative_eq!(t.tx, restored.tx, epsilon = 1e-10);
        assert_relative_eq!(t.ty, restored.ty, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_support() {
        let t: Affine2<f32> = Affine2::rotation(0.5);
        let p = Point2::new(1.0, 0.0);
        let result = t.apply_point(p);

        assert!((result.x - 0.5_f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn test_complex_composition() {
        // Build a complex transform step by step
        let t: Affine2<f64> = Affine2::identity()
            .then_translate(Vec2::new(1.0, 0.0))
            .then_rotate(FRAC_PI_2)
            .then_scale(2.0)
            .then_translate(Vec2::new(0.0, 5.0));

        // Starting with origin:
        // (0,0) -> translate -> (1,0)
        // (1,0) -> rotate 90° -> (0,1)
        // (0,1) -> scale 2 -> (0,2)
        // (0,2) -> translate -> (0,7)
        let result = t.apply_point(Point2::origin());

        assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 7.0, epsilon = 1e-10);
    }
}
