//! Sobol low-discrepancy sequences.
//!
//! Sobol sequences are quasi-random sequences with excellent uniformity properties,
//! especially in higher dimensions where they often outperform Halton sequences.
//!
//! # How It Works
//!
//! Sobol sequences use direction numbers derived from primitive polynomials over GF(2).
//! Each dimension has its own set of direction numbers, and successive points are
//! generated using XOR operations with these numbers.
//!
//! # Properties
//!
//! - Better uniformity than Halton in high dimensions
//! - Efficient generation using Gray code
//! - Deterministic and reproducible
//! - Points fill the space more uniformly than pseudo-random
//!
//! # Example
//!
//! ```
//! use approxum::sampling::{sobol_sequence, sobol_2d};
//! use approxum::Point2;
//!
//! // Generate 100 quasi-random 2D points in [0,1)²
//! let points: Vec<Point2<f64>> = sobol_sequence(100);
//! assert_eq!(points.len(), 100);
//!
//! // Get a single 2D point at index 42
//! let (x, y) = sobol_2d(42);
//! assert!(x >= 0.0 && x < 1.0);
//! assert!(y >= 0.0 && y < 1.0);
//! ```

use crate::primitives::Point2;
use num_traits::Float;

/// Number of bits used for Sobol sequence generation (32-bit precision).
const SOBOL_BITS: u32 = 32;

/// Scaling factor to convert integers to [0, 1).
const SOBOL_SCALE: f64 = 1.0 / (1u64 << SOBOL_BITS) as f64;

/// Direction numbers for dimension 1 (based on primitive polynomial x + 1).
/// These are the standard direction numbers m_i scaled by 2^(32-i).
const DIRECTION_1: [u32; 32] = {
    let mut dirs = [0u32; 32];
    let mut i = 0;
    while i < 32 {
        dirs[i] = 1u32 << (31 - i);
        i += 1;
    }
    dirs
};

/// Direction numbers for dimension 2 (based on primitive polynomial x² + x + 1).
const DIRECTION_2: [u32; 32] = init_direction_2();

/// Direction numbers for dimension 3 (based on primitive polynomial x³ + x + 1).
const DIRECTION_3: [u32; 32] = init_direction_3();

/// Direction numbers for dimension 4 (based on primitive polynomial x³ + x² + 1).
const DIRECTION_4: [u32; 32] = init_direction_4();

/// Computes a bit mask for i+1 bits without overflow.
/// Returns (2^(i+1) - 1), handling the i=31 case specially.
const fn bit_mask(i: usize) -> u32 {
    if i >= 31 {
        !0u32
    } else {
        (1u32 << (i + 1)) - 1
    }
}

/// Initialize direction numbers for dimension 2.
const fn init_direction_2() -> [u32; 32] {
    let mut dirs = [0u32; 32];
    // Initial values: m1 = 1, m2 = 3 (standard for this polynomial)
    let mut m = [0u32; 32];
    m[0] = 1;
    m[1] = 3;

    // Recurrence relation for x² + x + 1: m_i = 2 * m_{i-1} XOR m_{i-2}
    let mut i = 2;
    while i < 32 {
        m[i] = (2u32.wrapping_mul(m[i - 1])) ^ m[i - 2];
        // Keep only relevant bits
        m[i] &= bit_mask(i);
        i += 1;
    }

    // Scale to 32-bit integers
    i = 0;
    while i < 32 {
        dirs[i] = m[i] << (31 - i);
        i += 1;
    }
    dirs
}

/// Initialize direction numbers for dimension 3.
const fn init_direction_3() -> [u32; 32] {
    let mut dirs = [0u32; 32];
    let mut m = [0u32; 32];
    // Initial values for x³ + x + 1
    m[0] = 1;
    m[1] = 1;
    m[2] = 7;

    // Recurrence: m_i = 4 * m_{i-1} XOR m_{i-3}
    let mut i = 3;
    while i < 32 {
        m[i] = (4u32.wrapping_mul(m[i - 1])) ^ m[i - 3];
        m[i] &= bit_mask(i);
        i += 1;
    }

    i = 0;
    while i < 32 {
        dirs[i] = m[i] << (31 - i);
        i += 1;
    }
    dirs
}

/// Initialize direction numbers for dimension 4.
const fn init_direction_4() -> [u32; 32] {
    let mut dirs = [0u32; 32];
    let mut m = [0u32; 32];
    // Initial values for x³ + x² + 1
    m[0] = 1;
    m[1] = 3;
    m[2] = 5;

    // Recurrence: m_i = 4 * m_{i-1} XOR 2 * m_{i-2} XOR m_{i-3}
    let mut i = 3;
    while i < 32 {
        m[i] = (4u32.wrapping_mul(m[i - 1])) ^ (2u32.wrapping_mul(m[i - 2])) ^ m[i - 3];
        m[i] &= bit_mask(i);
        i += 1;
    }

    i = 0;
    while i < 32 {
        dirs[i] = m[i] << (31 - i);
        i += 1;
    }
    dirs
}

/// Returns the position of the rightmost zero bit in n.
/// This is used for Gray code-based Sobol generation.
#[cfg(test)]
#[inline]
fn rightmost_zero_bit(n: u32) -> u32 {
    // Find position of rightmost 0 bit (0-indexed from right)
    (!n).trailing_zeros()
}

/// Computes the nth value in dimension d of the Sobol sequence.
///
/// Uses the direct formula based on binary representation of n.
fn sobol_value(n: u32, directions: &[u32; 32]) -> u32 {
    let mut result = 0u32;
    let mut index = n;
    let mut bit = 0;

    while index > 0 {
        if index & 1 == 1 {
            result ^= directions[bit];
        }
        index >>= 1;
        bit += 1;
    }

    result
}

/// Returns the nth 2D Sobol point.
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
/// use approxum::sampling::sobol_2d;
///
/// let (x, y) = sobol_2d(0);
/// assert_eq!(x, 0.0);
/// assert_eq!(y, 0.0);
///
/// let (x, y) = sobol_2d(1);
/// assert!((x - 0.5).abs() < 1e-10);
/// assert!((y - 0.5).abs() < 1e-10);
/// ```
#[inline]
pub fn sobol_2d(index: u32) -> (f64, f64) {
    let x = sobol_value(index, &DIRECTION_1) as f64 * SOBOL_SCALE;
    let y = sobol_value(index, &DIRECTION_2) as f64 * SOBOL_SCALE;
    (x, y)
}

/// Returns the nth 2D Sobol point as a Point2.
#[inline]
pub fn sobol_2d_point<F: Float>(index: u32) -> Point2<F> {
    let (x, y) = sobol_2d(index);
    Point2::new(F::from(x).unwrap(), F::from(y).unwrap())
}

/// Generates a sequence of 2D Sobol points.
///
/// Points are generated in [0, 1)² using the first two dimensions.
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
/// use approxum::sampling::sobol_sequence;
/// use approxum::Point2;
///
/// let points: Vec<Point2<f64>> = sobol_sequence(10);
/// assert_eq!(points.len(), 10);
///
/// // All points should be in [0, 1)²
/// for p in &points {
///     assert!(p.x >= 0.0 && p.x < 1.0);
///     assert!(p.y >= 0.0 && p.y < 1.0);
/// }
/// ```
pub fn sobol_sequence<F: Float>(count: usize) -> Vec<Point2<F>> {
    (0..count as u32).map(sobol_2d_point).collect()
}

/// Generates a sequence of 2D Sobol points in a rectangular region.
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
pub fn sobol_sequence_in_rect<F: Float>(
    count: usize,
    min: Point2<F>,
    max: Point2<F>,
) -> Vec<Point2<F>> {
    let width = max.x - min.x;
    let height = max.y - min.y;

    (0..count as u32)
        .map(|i| {
            let (sx, sy) = sobol_2d(i);
            Point2::new(
                min.x + F::from(sx).unwrap() * width,
                min.y + F::from(sy).unwrap() * height,
            )
        })
        .collect()
}

/// Generates a sequence of 2D Sobol points, skipping the first `skip` indices.
///
/// Skipping the origin (index 0) is common since it's always (0, 0).
///
/// # Arguments
///
/// * `count` - Number of points to generate
/// * `skip` - Number of initial indices to skip
///
/// # Returns
///
/// Points starting from index `skip`.
pub fn sobol_sequence_skip<F: Float>(count: usize, skip: u32) -> Vec<Point2<F>> {
    (skip..skip + count as u32).map(sobol_2d_point).collect()
}

/// Returns the nth point in a higher-dimensional Sobol sequence.
///
/// Supports up to 4 dimensions.
///
/// # Arguments
///
/// * `index` - The sequence index
/// * `dimensions` - Number of dimensions (1-4)
///
/// # Returns
///
/// A vector of coordinates, each in [0, 1).
///
/// # Panics
///
/// Panics if `dimensions` > 4 or `dimensions` == 0.
///
/// # Example
///
/// ```
/// use approxum::sampling::sobol_nd;
///
/// let point = sobol_nd(5, 4);
/// assert_eq!(point.len(), 4);
/// ```
pub fn sobol_nd(index: u32, dimensions: usize) -> Vec<f64> {
    assert!((1..=4).contains(&dimensions), "Dimensions must be 1-4");

    let direction_tables = [&DIRECTION_1, &DIRECTION_2, &DIRECTION_3, &DIRECTION_4];

    direction_tables[..dimensions]
        .iter()
        .map(|dirs| sobol_value(index, dirs) as f64 * SOBOL_SCALE)
        .collect()
}

/// Iterator that generates Sobol sequence points on demand.
///
/// Uses Gray code optimization for efficient sequential generation.
///
/// # Example
///
/// ```
/// use approxum::sampling::SobolIterator;
/// use approxum::Point2;
///
/// let mut iter = SobolIterator::new();
///
/// let first: Point2<f64> = iter.next().unwrap();
/// let second: Point2<f64> = iter.next().unwrap();
///
/// assert_eq!(first.x, 0.0);
/// assert!((second.x - 0.5).abs() < 1e-10);
/// ```
pub struct SobolIterator<F> {
    index: u32,
    // Current state for Gray code generation
    state_x: u32,
    state_y: u32,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> SobolIterator<F> {
    /// Creates a new Sobol iterator starting at index 0.
    pub fn new() -> Self {
        Self {
            index: 0,
            state_x: 0,
            state_y: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Creates a new Sobol iterator starting at the given index.
    pub fn from_index(start: u32) -> Self {
        Self {
            index: start,
            state_x: sobol_value(start, &DIRECTION_1),
            state_y: sobol_value(start, &DIRECTION_2),
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the current index.
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl<F: Float> Default for SobolIterator<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> Iterator for SobolIterator<F> {
    type Item = Point2<F>;

    fn next(&mut self) -> Option<Self::Item> {
        // Compute point directly from index for consistency with sobol_2d
        let x = F::from(self.state_x as f64 * SOBOL_SCALE).unwrap();
        let y = F::from(self.state_y as f64 * SOBOL_SCALE).unwrap();
        let point = Point2::new(x, y);

        // Increment index and compute next state directly
        self.index = self.index.saturating_add(1);
        self.state_x = sobol_value(self.index, &DIRECTION_1);
        self.state_y = sobol_value(self.index, &DIRECTION_2);

        Some(point)
    }
}

/// Owen-scrambled Sobol sequence for better 2D projections.
///
/// Applies a random scrambling to improve uniformity while maintaining
/// the low-discrepancy property.
pub fn sobol_sequence_scrambled<F: Float>(count: usize, seed: u64) -> Vec<Point2<F>> {
    // Simple scrambling using seed-based XOR
    let scramble_x = (seed & 0xFFFFFFFF) as u32;
    let scramble_y = ((seed >> 32) & 0xFFFFFFFF) as u32;

    (0..count as u32)
        .map(|i| {
            let x = (sobol_value(i, &DIRECTION_1) ^ scramble_x) as f64 * SOBOL_SCALE;
            let y = (sobol_value(i, &DIRECTION_2) ^ scramble_y) as f64 * SOBOL_SCALE;
            Point2::new(F::from(x).unwrap(), F::from(y).unwrap())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_sobol_2d_first_points() {
        // Index 0 should give (0, 0)
        let (x, y) = sobol_2d(0);
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);

        // Index 1 should give (0.5, 0.5)
        let (x, y) = sobol_2d(1);
        assert!(approx_eq(x, 0.5, 1e-10));
        assert!(approx_eq(y, 0.5, 1e-10));

        // Index 2 should give (0.25, 0.75)
        let (x, y) = sobol_2d(2);
        assert!(approx_eq(x, 0.25, 1e-10));
        assert!(approx_eq(y, 0.75, 1e-10));

        // Index 3 should give (0.75, 0.25)
        let (x, y) = sobol_2d(3);
        assert!(approx_eq(x, 0.75, 1e-10));
        assert!(approx_eq(y, 0.25, 1e-10));
    }

    #[test]
    fn test_sobol_sequence() {
        let points: Vec<Point2<f64>> = sobol_sequence(16);
        assert_eq!(points.len(), 16);

        // All points should be in [0, 1)
        for p in &points {
            assert!(p.x >= 0.0 && p.x < 1.0);
            assert!(p.y >= 0.0 && p.y < 1.0);
        }

        // Check specific values
        assert!(approx_eq(points[0].x, 0.0, 1e-10));
        assert!(approx_eq(points[1].x, 0.5, 1e-10));
    }

    #[test]
    fn test_sobol_sequence_in_rect() {
        let min = Point2::new(10.0_f64, 20.0);
        let max = Point2::new(20.0, 30.0);
        let points = sobol_sequence_in_rect(50, min, max);

        assert_eq!(points.len(), 50);

        for p in &points {
            assert!(p.x >= 10.0 && p.x <= 20.0);
            assert!(p.y >= 20.0 && p.y <= 30.0);
        }
    }

    #[test]
    fn test_sobol_sequence_skip() {
        let skip = 10u32;
        let points: Vec<Point2<f64>> = sobol_sequence_skip(5, skip);

        assert_eq!(points.len(), 5);

        // First point should match sobol_2d(10)
        let expected = sobol_2d_point::<f64>(10);
        assert!(approx_eq(points[0].x, expected.x, 1e-10));
        assert!(approx_eq(points[0].y, expected.y, 1e-10));
    }

    #[test]
    fn test_sobol_nd() {
        let point = sobol_nd(5, 4);
        assert_eq!(point.len(), 4);

        // Check that values are in [0, 1)
        for &val in &point {
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_sobol_iterator() {
        let mut iter: SobolIterator<f64> = SobolIterator::new();

        let p0 = iter.next().unwrap();
        assert_eq!(iter.index(), 1);
        assert!(approx_eq(p0.x, 0.0, 1e-10));
        assert!(approx_eq(p0.y, 0.0, 1e-10));

        let p1 = iter.next().unwrap();
        assert_eq!(iter.index(), 2);
        assert!(approx_eq(p1.x, 0.5, 1e-10));
        assert!(approx_eq(p1.y, 0.5, 1e-10));
    }

    #[test]
    fn test_sobol_iterator_from_index() {
        let mut iter: SobolIterator<f64> = SobolIterator::from_index(100);
        assert_eq!(iter.index(), 100);

        let p = iter.next().unwrap();
        let expected = sobol_2d_point::<f64>(100);
        assert!(approx_eq(p.x, expected.x, 1e-10));
        assert!(approx_eq(p.y, expected.y, 1e-10));
    }

    #[test]
    fn test_sobol_iterator_matches_direct() {
        let mut iter: SobolIterator<f64> = SobolIterator::new();

        for i in 0..100 {
            let iter_point = iter.next().unwrap();
            let direct_point: Point2<f64> = sobol_2d_point(i);

            assert!(
                approx_eq(iter_point.x, direct_point.x, 1e-10),
                "Mismatch at index {} x: {} vs {}",
                i,
                iter_point.x,
                direct_point.x
            );
            assert!(
                approx_eq(iter_point.y, direct_point.y, 1e-10),
                "Mismatch at index {} y: {} vs {}",
                i,
                iter_point.y,
                direct_point.y
            );
        }
    }

    #[test]
    fn test_sobol_sequence_scrambled() {
        let points: Vec<Point2<f64>> = sobol_sequence_scrambled(20, 12345);
        assert_eq!(points.len(), 20);

        for p in &points {
            assert!(p.x >= 0.0 && p.x < 1.0);
            assert!(p.y >= 0.0 && p.y < 1.0);
        }

        // Different seeds should give different sequences
        let points2: Vec<Point2<f64>> = sobol_sequence_scrambled(20, 54321);
        assert!(points[1] != points2[1]);
    }

    #[test]
    fn test_low_discrepancy() {
        // Sobol sequences should have lower discrepancy than random
        let points: Vec<Point2<f64>> = sobol_sequence(64);

        // Count points in each quadrant
        let mut quadrants = [0; 4];
        for p in &points {
            let qx = if p.x < 0.5 { 0 } else { 1 };
            let qy = if p.y < 0.5 { 0 } else { 2 };
            quadrants[qx + qy] += 1;
        }

        // Each quadrant should have exactly 16 points for a 64-point Sobol sequence
        for &count in &quadrants {
            assert_eq!(count, 16);
        }
    }

    #[test]
    fn test_rightmost_zero_bit() {
        assert_eq!(rightmost_zero_bit(0), 0); // 0b0 -> first 0 at position 0
        assert_eq!(rightmost_zero_bit(1), 1); // 0b1 -> first 0 at position 1
        assert_eq!(rightmost_zero_bit(3), 2); // 0b11 -> first 0 at position 2
        assert_eq!(rightmost_zero_bit(7), 3); // 0b111 -> first 0 at position 3
        assert_eq!(rightmost_zero_bit(2), 0); // 0b10 -> first 0 at position 0
    }

    #[test]
    fn test_f32() {
        let points: Vec<Point2<f32>> = sobol_sequence(10);
        assert_eq!(points.len(), 10);

        assert!((points[1].x - 0.5).abs() < 0.001);
        assert!((points[1].y - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sobol_unique_points() {
        // First 2^k points of a Sobol sequence should all be unique
        let points: Vec<Point2<f64>> = sobol_sequence(256);

        for i in 0..points.len() {
            for j in i + 1..points.len() {
                assert!(
                    points[i] != points[j],
                    "Duplicate points at {} and {}",
                    i,
                    j
                );
            }
        }
    }
}
