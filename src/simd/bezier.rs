//! SIMD-accelerated Bézier curve evaluation.
//!
//! Evaluates curves at 4 parameter values simultaneously.

#![allow(clippy::needless_range_loop)]

use wide::f32x4;

use crate::curves::{CubicBezier2, QuadraticBezier2};
use crate::primitives::Point2;

use super::point::Point2x4;

/// A quadratic Bézier curve that can evaluate 4 parameter values at once.
#[derive(Clone, Copy, Debug)]
pub struct QuadraticBezier2x4 {
    /// Start point (same for all 4 evaluations)
    p0: Point2x4,
    /// Control point
    p1: Point2x4,
    /// End point
    p2: Point2x4,
}

impl QuadraticBezier2x4 {
    /// Creates a new SIMD quadratic Bézier from control points.
    #[inline]
    pub fn new(p0: Point2<f32>, p1: Point2<f32>, p2: Point2<f32>) -> Self {
        Self {
            p0: Point2x4::splat(p0),
            p1: Point2x4::splat(p1),
            p2: Point2x4::splat(p2),
        }
    }

    /// Creates from an existing QuadraticBezier2.
    #[inline]
    pub fn from_curve(curve: &QuadraticBezier2<f32>) -> Self {
        Self::new(curve.p0, curve.p1, curve.p2)
    }

    /// Evaluates the curve at 4 parameter values simultaneously.
    ///
    /// Uses de Casteljau's algorithm.
    #[inline]
    pub fn eval(&self, t: f32x4) -> Point2x4 {
        // First level interpolation
        let q0 = self.p0.lerp(self.p1, t);
        let q1 = self.p1.lerp(self.p2, t);

        // Second level interpolation
        q0.lerp(q1, t)
    }

    /// Evaluates the curve at uniformly spaced parameter values.
    ///
    /// Returns points at t = [0, 1/3, 2/3, 1] (or similar spacing).
    #[inline]
    pub fn eval_uniform(&self, start_t: f32, end_t: f32) -> Point2x4 {
        let delta = (end_t - start_t) / 3.0;
        let t = f32x4::new([start_t, start_t + delta, start_t + 2.0 * delta, end_t]);
        self.eval(t)
    }

    /// Evaluates the curve at an array of parameter values.
    #[inline]
    pub fn eval_array(&self, t: [f32; 4]) -> [Point2<f32>; 4] {
        self.eval(f32x4::new(t)).to_array()
    }
}

/// A cubic Bézier curve that can evaluate 4 parameter values at once.
#[derive(Clone, Copy, Debug)]
pub struct CubicBezier2x4 {
    /// Start point
    p0: Point2x4,
    /// First control point
    p1: Point2x4,
    /// Second control point
    p2: Point2x4,
    /// End point
    p3: Point2x4,
}

impl CubicBezier2x4 {
    /// Creates a new SIMD cubic Bézier from control points.
    #[inline]
    pub fn new(p0: Point2<f32>, p1: Point2<f32>, p2: Point2<f32>, p3: Point2<f32>) -> Self {
        Self {
            p0: Point2x4::splat(p0),
            p1: Point2x4::splat(p1),
            p2: Point2x4::splat(p2),
            p3: Point2x4::splat(p3),
        }
    }

    /// Creates from an existing CubicBezier2.
    #[inline]
    pub fn from_curve(curve: &CubicBezier2<f32>) -> Self {
        Self::new(curve.p0, curve.p1, curve.p2, curve.p3)
    }

    /// Evaluates the curve at 4 parameter values simultaneously.
    ///
    /// Uses de Casteljau's algorithm.
    #[inline]
    pub fn eval(&self, t: f32x4) -> Point2x4 {
        // First level interpolation
        let q0 = self.p0.lerp(self.p1, t);
        let q1 = self.p1.lerp(self.p2, t);
        let q2 = self.p2.lerp(self.p3, t);

        // Second level interpolation
        let r0 = q0.lerp(q1, t);
        let r1 = q1.lerp(q2, t);

        // Third level interpolation
        r0.lerp(r1, t)
    }

    /// Evaluates the curve at uniformly spaced parameter values.
    #[inline]
    pub fn eval_uniform(&self, start_t: f32, end_t: f32) -> Point2x4 {
        let delta = (end_t - start_t) / 3.0;
        let t = f32x4::new([start_t, start_t + delta, start_t + 2.0 * delta, end_t]);
        self.eval(t)
    }

    /// Evaluates the curve at an array of parameter values.
    #[inline]
    pub fn eval_array(&self, t: [f32; 4]) -> [Point2<f32>; 4] {
        self.eval(f32x4::new(t)).to_array()
    }

    /// Evaluates many parameter values efficiently.
    ///
    /// Processes in batches of 4 for SIMD efficiency.
    pub fn eval_many(&self, params: &[f32]) -> Vec<Point2<f32>> {
        let n = params.len();
        let mut result = Vec::with_capacity(n);

        // Process in batches of 4
        let chunks = n / 4;
        for i in 0..chunks {
            let t = f32x4::new([
                params[i * 4],
                params[i * 4 + 1],
                params[i * 4 + 2],
                params[i * 4 + 3],
            ]);
            result.extend_from_slice(&self.eval(t).to_array());
        }

        // Handle remainder with scalar curve
        let scalar_curve = CubicBezier2::new(
            self.p0.to_array()[0],
            self.p1.to_array()[0],
            self.p2.to_array()[0],
            self.p3.to_array()[0],
        );
        for i in (chunks * 4)..n {
            result.push(scalar_curve.eval(params[i]));
        }

        result
    }
}

/// Batch evaluates a cubic Bézier curve at many parameter values.
///
/// This is a convenience function that creates a SIMD curve internally.
pub fn eval_cubic_batch(curve: &CubicBezier2<f32>, params: &[f32]) -> Vec<Point2<f32>> {
    CubicBezier2x4::from_curve(curve).eval_many(params)
}

/// Batch evaluates a quadratic Bézier curve at many parameter values.
pub fn eval_quadratic_batch(curve: &QuadraticBezier2<f32>, params: &[f32]) -> Vec<Point2<f32>> {
    let simd_curve = QuadraticBezier2x4::from_curve(curve);
    let n = params.len();
    let mut result = Vec::with_capacity(n);

    // Process in batches of 4
    let chunks = n / 4;
    for i in 0..chunks {
        let t = f32x4::new([
            params[i * 4],
            params[i * 4 + 1],
            params[i * 4 + 2],
            params[i * 4 + 3],
        ]);
        result.extend_from_slice(&simd_curve.eval(t).to_array());
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        result.push(curve.eval(params[i]));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_bezier_eval() {
        let curve = QuadraticBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(5.0, 10.0),
            Point2::new(10.0, 0.0),
        );
        let simd_curve = QuadraticBezier2x4::from_curve(&curve);

        let t = f32x4::new([0.0, 0.25, 0.5, 1.0]);
        let points = simd_curve.eval(t).to_array();

        // Check against scalar implementation
        for (i, &t_val) in [0.0, 0.25, 0.5, 1.0].iter().enumerate() {
            let expected = curve.eval(t_val);
            assert!(
                (points[i].x - expected.x).abs() < 1e-5,
                "x mismatch at t={}: {} vs {}",
                t_val,
                points[i].x,
                expected.x
            );
            assert!(
                (points[i].y - expected.y).abs() < 1e-5,
                "y mismatch at t={}: {} vs {}",
                t_val,
                points[i].y,
                expected.y
            );
        }
    }

    #[test]
    fn test_cubic_bezier_eval() {
        let curve = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(3.0, 10.0),
            Point2::new(7.0, 10.0),
            Point2::new(10.0, 0.0),
        );
        let simd_curve = CubicBezier2x4::from_curve(&curve);

        let t = f32x4::new([0.0, 0.25, 0.5, 1.0]);
        let points = simd_curve.eval(t).to_array();

        // Check against scalar implementation
        for (i, &t_val) in [0.0, 0.25, 0.5, 1.0].iter().enumerate() {
            let expected = curve.eval(t_val);
            assert!(
                (points[i].x - expected.x).abs() < 1e-5,
                "x mismatch at t={}: {} vs {}",
                t_val,
                points[i].x,
                expected.x
            );
            assert!(
                (points[i].y - expected.y).abs() < 1e-5,
                "y mismatch at t={}: {} vs {}",
                t_val,
                points[i].y,
                expected.y
            );
        }
    }

    #[test]
    fn test_cubic_eval_many() {
        let curve = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(3.0, 10.0),
            Point2::new(7.0, 10.0),
            Point2::new(10.0, 0.0),
        );

        let params: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let simd_results = eval_cubic_batch(&curve, &params);

        // Verify against scalar
        for (i, &t) in params.iter().enumerate() {
            let expected = curve.eval(t);
            assert!(
                (simd_results[i].x - expected.x).abs() < 1e-5,
                "x mismatch at i={}: {} vs {}",
                i,
                simd_results[i].x,
                expected.x
            );
            assert!(
                (simd_results[i].y - expected.y).abs() < 1e-5,
                "y mismatch at i={}: {} vs {}",
                i,
                simd_results[i].y,
                expected.y
            );
        }
    }

    #[test]
    fn test_endpoints() {
        let curve = CubicBezier2::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 2.0),
            Point2::new(4.0, 0.0),
        );
        let simd_curve = CubicBezier2x4::from_curve(&curve);

        let t = f32x4::new([0.0, 0.0, 1.0, 1.0]);
        let points = simd_curve.eval(t).to_array();

        // Start point
        assert!((points[0].x - 0.0).abs() < 1e-6);
        assert!((points[0].y - 0.0).abs() < 1e-6);

        // End point
        assert!((points[2].x - 4.0).abs() < 1e-6);
        assert!((points[2].y - 0.0).abs() < 1e-6);
    }
}
