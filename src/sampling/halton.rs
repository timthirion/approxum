//! Halton low-discrepancy sequences.
//!
//! Halton sequences are deterministic quasi-random sequences with low discrepancy,
//! meaning points are more evenly distributed than pseudo-random sequences.
//! They're useful for:
//!
//! - Numerical integration (quasi-Monte Carlo)
//! - Sampling and coverage testing
//! - Computer graphics (anti-aliasing, ray tracing)
//! - Optimization (space-filling designs)
//!
//! # How It Works
//!
//! The Halton sequence uses the radical inverse function in a given base.
//! For a number n in base b, the radical inverse reflects the digits about
//! the decimal point. For example, in base 2:
//! - n=1 (binary: 1) → 0.1 binary = 0.5
//! - n=2 (binary: 10) → 0.01 binary = 0.25
//! - n=3 (binary: 11) → 0.11 binary = 0.75
//!
//! For 2D points, different prime bases (typically 2 and 3) are used for
//! each dimension to avoid correlation.
//!
//! # Example
//!
//! ```
//! use approxum::sampling::{halton_sequence, halton_2d};
//! use approxum::Point2;
//!
//! // Generate 100 quasi-random 2D points in [0,1]²
//! let points: Vec<Point2<f64>> = halton_sequence(100);
//! assert_eq!(points.len(), 100);
//!
//! // Get a single 2D point at index 42
//! let (x, y) = halton_2d(42);
//! assert!(x >= 0.0 && x <= 1.0);
//! assert!(y >= 0.0 && y <= 1.0);
//! ```

use crate::primitives::Point2;
use num_traits::Float;

/// First 20 prime numbers for higher-dimensional Halton sequences.
const PRIMES: [u32; 20] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
];

/// Computes the radical inverse of `n` in the given `base`.
///
/// The radical inverse takes a number and reflects its digits about the
/// decimal point in the specified base.
///
/// # Arguments
///
/// * `n` - The index (positive integer)
/// * `base` - The base for digit representation (typically a prime number)
///
/// # Returns
///
/// A value in [0, 1).
///
/// # Example
///
/// ```
/// use approxum::sampling::radical_inverse;
///
/// // In base 2: 1 -> 0.1 (binary) = 0.5
/// assert!((radical_inverse(1, 2) - 0.5).abs() < 1e-10);
///
/// // In base 2: 2 -> 0.01 (binary) = 0.25
/// assert!((radical_inverse(2, 2) - 0.25).abs() < 1e-10);
///
/// // In base 2: 3 -> 0.11 (binary) = 0.75
/// assert!((radical_inverse(3, 2) - 0.75).abs() < 1e-10);
/// ```
pub fn radical_inverse(n: u32, base: u32) -> f64 {
    let mut n = n;
    let mut result = 0.0;
    let mut fraction = 1.0 / base as f64;

    while n > 0 {
        let digit = n % base;
        result += digit as f64 * fraction;
        n /= base;
        fraction /= base as f64;
    }

    result
}

/// Computes the radical inverse with a generic float type.
pub fn radical_inverse_f<F: Float>(n: u32, base: u32) -> F {
    let mut n = n;
    let mut result = F::zero();
    let base_f = F::from(base).unwrap();
    let mut fraction = F::one() / base_f;

    while n > 0 {
        let digit = n % base;
        result = result + F::from(digit).unwrap() * fraction;
        n /= base;
        fraction = fraction / base_f;
    }

    result
}

/// Returns the nth 2D Halton point using bases 2 and 3.
///
/// # Arguments
///
/// * `index` - The sequence index (0-based)
///
/// # Returns
///
/// A tuple (x, y) where both values are in [0, 1).
///
/// # Example
///
/// ```
/// use approxum::sampling::halton_2d;
///
/// let (x, y) = halton_2d(0);
/// assert_eq!(x, 0.0);
/// assert_eq!(y, 0.0);
///
/// let (x, y) = halton_2d(1);
/// assert!((x - 0.5).abs() < 1e-10);
/// assert!((y - 1.0/3.0).abs() < 1e-10);
/// ```
#[inline]
pub fn halton_2d(index: u32) -> (f64, f64) {
    (radical_inverse(index, 2), radical_inverse(index, 3))
}

/// Returns the nth 2D Halton point as a Point2.
#[inline]
pub fn halton_2d_point<F: Float>(index: u32) -> Point2<F> {
    Point2::new(radical_inverse_f(index, 2), radical_inverse_f(index, 3))
}

/// Generates a sequence of 2D Halton points.
///
/// Points are generated in [0, 1)² using bases 2 and 3.
///
/// # Arguments
///
/// * `count` - Number of points to generate
///
/// # Returns
///
/// A vector of 2D points with low discrepancy.
///
/// # Example
///
/// ```
/// use approxum::sampling::halton_sequence;
/// use approxum::Point2;
///
/// let points: Vec<Point2<f64>> = halton_sequence(10);
/// assert_eq!(points.len(), 10);
///
/// // All points should be in [0, 1)²
/// for p in &points {
///     assert!(p.x >= 0.0 && p.x < 1.0);
///     assert!(p.y >= 0.0 && p.y < 1.0);
/// }
/// ```
pub fn halton_sequence<F: Float>(count: usize) -> Vec<Point2<F>> {
    (0..count as u32).map(halton_2d_point).collect()
}

/// Generates a sequence of 2D Halton points in a rectangular region.
///
/// # Arguments
///
/// * `count` - Number of points to generate
/// * `min` - Minimum corner of the region
/// * `max` - Maximum corner of the region
///
/// # Returns
///
/// Points distributed in the specified rectangle.
///
/// # Example
///
/// ```
/// use approxum::sampling::halton_sequence_in_rect;
/// use approxum::Point2;
///
/// let min = Point2::new(0.0_f64, 0.0);
/// let max = Point2::new(10.0, 5.0);
/// let points = halton_sequence_in_rect(20, min, max);
///
/// for p in &points {
///     assert!(p.x >= 0.0 && p.x <= 10.0);
///     assert!(p.y >= 0.0 && p.y <= 5.0);
/// }
/// ```
pub fn halton_sequence_in_rect<F: Float>(
    count: usize,
    min: Point2<F>,
    max: Point2<F>,
) -> Vec<Point2<F>> {
    let width = max.x - min.x;
    let height = max.y - min.y;

    (0..count as u32)
        .map(|i| {
            let (hx, hy) = (radical_inverse_f::<F>(i, 2), radical_inverse_f::<F>(i, 3));
            Point2::new(min.x + hx * width, min.y + hy * height)
        })
        .collect()
}

/// Generates a sequence of 2D Halton points, skipping the first `skip` indices.
///
/// Skipping the first few indices can help avoid correlation in some applications.
/// A common choice is to skip 20-100 initial samples.
///
/// # Arguments
///
/// * `count` - Number of points to generate
/// * `skip` - Number of initial indices to skip
///
/// # Returns
///
/// Points starting from index `skip`.
pub fn halton_sequence_skip<F: Float>(count: usize, skip: u32) -> Vec<Point2<F>> {
    (skip..skip + count as u32).map(halton_2d_point).collect()
}

/// Returns the nth point in a higher-dimensional Halton sequence.
///
/// Uses the first `dimensions` prime numbers as bases.
///
/// # Arguments
///
/// * `index` - The sequence index
/// * `dimensions` - Number of dimensions (max 20)
///
/// # Returns
///
/// A vector of coordinates, each in [0, 1).
///
/// # Panics
///
/// Panics if `dimensions` > 20.
///
/// # Example
///
/// ```
/// use approxum::sampling::halton_nd;
///
/// let point = halton_nd(5, 4);
/// assert_eq!(point.len(), 4);
/// ```
pub fn halton_nd(index: u32, dimensions: usize) -> Vec<f64> {
    assert!(
        dimensions <= PRIMES.len(),
        "Maximum 20 dimensions supported"
    );

    PRIMES[..dimensions]
        .iter()
        .map(|&base| radical_inverse(index, base))
        .collect()
}

/// Generates a scrambled Halton sequence using digit permutations.
///
/// Scrambled Halton sequences can have better properties for some applications
/// by applying a fixed permutation to the digits in each base. This uses
/// the Faure permutation.
///
/// # Arguments
///
/// * `count` - Number of points to generate
///
/// # Returns
///
/// Scrambled 2D Halton points.
pub fn halton_sequence_scrambled<F: Float>(count: usize) -> Vec<Point2<F>> {
    (0..count as u32)
        .map(|i| {
            Point2::new(
                radical_inverse_scrambled(i, 2),
                radical_inverse_scrambled(i, 3),
            )
        })
        .collect()
}

/// Computes scrambled radical inverse using Faure permutation.
fn radical_inverse_scrambled<F: Float>(n: u32, base: u32) -> F {
    let mut n = n;
    let mut result = F::zero();
    let base_f = F::from(base).unwrap();
    let mut fraction = F::one() / base_f;

    while n > 0 {
        let digit = n % base;
        // Apply Faure scrambling: permute(digit, base)
        let scrambled = faure_permutation(digit, base);
        result = result + F::from(scrambled).unwrap() * fraction;
        n /= base;
        fraction = fraction / base_f;
    }

    result
}

/// Faure permutation for digit scrambling.
fn faure_permutation(digit: u32, base: u32) -> u32 {
    if base == 2 {
        digit // No permutation needed for base 2
    } else {
        // Simple permutation: reverse order for odd bases
        (base - 1 - digit) % base
    }
}

/// Iterator that generates Halton sequence points on demand.
///
/// This is memory-efficient for large sequences since points are
/// computed lazily.
///
/// # Example
///
/// ```
/// use approxum::sampling::HaltonIterator;
/// use approxum::Point2;
///
/// let mut iter: HaltonIterator<f64> = HaltonIterator::new();
///
/// let first: Point2<f64> = iter.next().unwrap();
/// let second: Point2<f64> = iter.next().unwrap();
///
/// assert_eq!(first.x, 0.0);
/// assert!((second.x - 0.5).abs() < 1e-10);
/// ```
pub struct HaltonIterator<F> {
    index: u32,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> HaltonIterator<F> {
    /// Creates a new Halton iterator starting at index 0.
    pub fn new() -> Self {
        Self {
            index: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Creates a new Halton iterator starting at the given index.
    pub fn from_index(start: u32) -> Self {
        Self {
            index: start,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the current index.
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl<F: Float> Default for HaltonIterator<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> Iterator for HaltonIterator<F> {
    type Item = Point2<F>;

    fn next(&mut self) -> Option<Self::Item> {
        let point = halton_2d_point(self.index);
        self.index = self.index.saturating_add(1);
        Some(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_radical_inverse_base_2() {
        // n=1: 1 binary -> 0.1 binary = 0.5
        assert!(approx_eq(radical_inverse(1, 2), 0.5, 1e-10));
        // n=2: 10 binary -> 0.01 binary = 0.25
        assert!(approx_eq(radical_inverse(2, 2), 0.25, 1e-10));
        // n=3: 11 binary -> 0.11 binary = 0.75
        assert!(approx_eq(radical_inverse(3, 2), 0.75, 1e-10));
        // n=4: 100 binary -> 0.001 binary = 0.125
        assert!(approx_eq(radical_inverse(4, 2), 0.125, 1e-10));
        // n=5: 101 binary -> 0.101 binary = 0.625
        assert!(approx_eq(radical_inverse(5, 2), 0.625, 1e-10));
    }

    #[test]
    fn test_radical_inverse_base_3() {
        // n=1: 1 in base 3 -> 0.1 = 1/3
        assert!(approx_eq(radical_inverse(1, 3), 1.0 / 3.0, 1e-10));
        // n=2: 2 in base 3 -> 0.2 = 2/3
        assert!(approx_eq(radical_inverse(2, 3), 2.0 / 3.0, 1e-10));
        // n=3: 10 in base 3 -> 0.01 = 1/9
        assert!(approx_eq(radical_inverse(3, 3), 1.0 / 9.0, 1e-10));
        // n=4: 11 in base 3 -> 0.11 = 1/3 + 1/9 = 4/9
        assert!(approx_eq(radical_inverse(4, 3), 4.0 / 9.0, 1e-10));
    }

    #[test]
    fn test_radical_inverse_zero() {
        assert_eq!(radical_inverse(0, 2), 0.0);
        assert_eq!(radical_inverse(0, 3), 0.0);
    }

    #[test]
    fn test_halton_2d() {
        let (x, y) = halton_2d(0);
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);

        let (x, y) = halton_2d(1);
        assert!(approx_eq(x, 0.5, 1e-10));
        assert!(approx_eq(y, 1.0 / 3.0, 1e-10));

        let (x, y) = halton_2d(2);
        assert!(approx_eq(x, 0.25, 1e-10));
        assert!(approx_eq(y, 2.0 / 3.0, 1e-10));
    }

    #[test]
    fn test_halton_sequence() {
        let points: Vec<Point2<f64>> = halton_sequence(10);
        assert_eq!(points.len(), 10);

        // All points should be in [0, 1)
        for p in &points {
            assert!(p.x >= 0.0 && p.x < 1.0);
            assert!(p.y >= 0.0 && p.y < 1.0);
        }

        // Check first few specific values
        assert!(approx_eq(points[0].x, 0.0, 1e-10));
        assert!(approx_eq(points[1].x, 0.5, 1e-10));
        assert!(approx_eq(points[2].x, 0.25, 1e-10));
    }

    #[test]
    fn test_halton_sequence_in_rect() {
        let min = Point2::new(10.0_f64, 20.0);
        let max = Point2::new(20.0, 30.0);
        let points = halton_sequence_in_rect(50, min, max);

        assert_eq!(points.len(), 50);

        for p in &points {
            assert!(p.x >= 10.0 && p.x <= 20.0);
            assert!(p.y >= 20.0 && p.y <= 30.0);
        }
    }

    #[test]
    fn test_halton_sequence_skip() {
        let skip = 10u32;
        let points: Vec<Point2<f64>> = halton_sequence_skip(5, skip);

        assert_eq!(points.len(), 5);

        // First point should match halton_2d(10)
        let expected = halton_2d_point::<f64>(10);
        assert!(approx_eq(points[0].x, expected.x, 1e-10));
        assert!(approx_eq(points[0].y, expected.y, 1e-10));
    }

    #[test]
    fn test_halton_nd() {
        let point = halton_nd(5, 4);
        assert_eq!(point.len(), 4);

        // Check that values are in [0, 1)
        for &val in &point {
            assert!(val >= 0.0 && val < 1.0);
        }

        // First dimension should match base-2 radical inverse
        assert!(approx_eq(point[0], radical_inverse(5, 2), 1e-10));
        // Second dimension should match base-3
        assert!(approx_eq(point[1], radical_inverse(5, 3), 1e-10));
    }

    #[test]
    fn test_halton_iterator() {
        let mut iter: HaltonIterator<f64> = HaltonIterator::new();

        let p0 = iter.next().unwrap();
        assert_eq!(iter.index(), 1);
        assert!(approx_eq(p0.x, 0.0, 1e-10));

        let p1 = iter.next().unwrap();
        assert_eq!(iter.index(), 2);
        assert!(approx_eq(p1.x, 0.5, 1e-10));
    }

    #[test]
    fn test_halton_iterator_from_index() {
        let mut iter: HaltonIterator<f64> = HaltonIterator::from_index(100);
        assert_eq!(iter.index(), 100);

        let p = iter.next().unwrap();
        let expected = halton_2d_point::<f64>(100);
        assert!(approx_eq(p.x, expected.x, 1e-10));
    }

    #[test]
    fn test_halton_sequence_scrambled() {
        let points: Vec<Point2<f64>> = halton_sequence_scrambled(20);
        assert_eq!(points.len(), 20);

        for p in &points {
            assert!(p.x >= 0.0 && p.x < 1.0);
            assert!(p.y >= 0.0 && p.y < 1.0);
        }
    }

    #[test]
    fn test_low_discrepancy() {
        // Halton sequences should have lower discrepancy than random
        // We'll just verify that points cover the space reasonably well
        let points: Vec<Point2<f64>> = halton_sequence(100);

        // Count points in each quadrant
        let mut quadrants = [0; 4];
        for p in &points {
            let qx = if p.x < 0.5 { 0 } else { 1 };
            let qy = if p.y < 0.5 { 0 } else { 2 };
            quadrants[qx + qy] += 1;
        }

        // Each quadrant should have roughly 25 points (± some tolerance)
        for &count in &quadrants {
            assert!(count >= 15 && count <= 35);
        }
    }

    #[test]
    fn test_f32() {
        let points: Vec<Point2<f32>> = halton_sequence(10);
        assert_eq!(points.len(), 10);

        assert!((points[1].x - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_radical_inverse_f() {
        let val: f64 = radical_inverse_f(5, 2);
        assert!(approx_eq(val, 0.625, 1e-10));

        let val32: f32 = radical_inverse_f(5, 2);
        assert!((val32 - 0.625).abs() < 0.001);
    }
}
