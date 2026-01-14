//! Poisson disk sampling.
//!
//! Generates blue noise point distributions where no two points are closer
//! than a specified minimum distance. Uses Bridson's fast algorithm with
//! O(n) time complexity.

use crate::primitives::Point2;
use num_traits::Float;

/// Generates a Poisson disk sample in a rectangular domain.
///
/// Uses Bridson's fast algorithm for O(n) time complexity.
///
/// # Arguments
///
/// * `width` - Width of the sampling domain
/// * `height` - Height of the sampling domain
/// * `min_distance` - Minimum distance between any two points
/// * `max_attempts` - Maximum attempts to place a point around an active sample (typically 30)
///
/// # Returns
///
/// A vector of points with blue noise distribution.
///
/// # Example
///
/// ```
/// use approxum::sampling::poisson_disk;
///
/// let points = poisson_disk(10.0, 10.0, 1.0, 30);
///
/// // Check minimum distance property
/// for i in 0..points.len() {
///     for j in (i + 1)..points.len() {
///         let dist = points[i].distance(points[j]);
///         assert!(dist >= 0.99); // Allow small floating point tolerance
///     }
/// }
/// ```
pub fn poisson_disk<F: Float>(
    width: F,
    height: F,
    min_distance: F,
    max_attempts: usize,
) -> Vec<Point2<F>> {
    // Use a deterministic seed based on parameters
    let seed = (width.to_f64().unwrap_or(1.0) * 1000.0
        + height.to_f64().unwrap_or(1.0) * 100.0
        + min_distance.to_f64().unwrap_or(1.0) * 10.0) as u64;

    poisson_disk_with_seed(width, height, min_distance, max_attempts, seed)
}

/// Generates a Poisson disk sample with a specific random seed.
///
/// This allows for reproducible results.
pub fn poisson_disk_with_seed<F: Float>(
    width: F,
    height: F,
    min_distance: F,
    max_attempts: usize,
    seed: u64,
) -> Vec<Point2<F>> {
    let mut sampler = PoissonDiskSampler::new(width, height, min_distance, seed);
    sampler.generate(max_attempts)
}

/// A Poisson disk sampler that can be configured and reused.
pub struct PoissonDiskSampler<F> {
    width: F,
    height: F,
    min_distance: F,
    cell_size: F,
    grid_width: usize,
    grid_height: usize,
    grid: Vec<Option<usize>>,
    points: Vec<Point2<F>>,
    active: Vec<usize>,
    rng_state: u64,
}

impl<F: Float> PoissonDiskSampler<F> {
    /// Creates a new Poisson disk sampler.
    pub fn new(width: F, height: F, min_distance: F, seed: u64) -> Self {
        // Cell size = r / sqrt(2) ensures at most one point per cell
        let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap();
        let cell_size = min_distance / sqrt2;

        let grid_width = (width / cell_size).ceil().to_usize().unwrap_or(1).max(1);
        let grid_height = (height / cell_size).ceil().to_usize().unwrap_or(1).max(1);

        let grid = vec![None; grid_width * grid_height];

        Self {
            width,
            height,
            min_distance,
            cell_size,
            grid_width,
            grid_height,
            grid,
            points: Vec::new(),
            active: Vec::new(),
            rng_state: seed,
        }
    }

    /// Generates the Poisson disk sample.
    pub fn generate(&mut self, max_attempts: usize) -> Vec<Point2<F>> {
        self.points.clear();
        self.active.clear();
        for cell in &mut self.grid {
            *cell = None;
        }

        // Start with a random point
        let first = self.random_point_in_domain();
        self.add_point(first);

        // Process active list
        while !self.active.is_empty() {
            // Pick a random active point
            let active_idx = self.random_usize(self.active.len());
            let point_idx = self.active[active_idx];
            let center = self.points[point_idx];

            let mut found = false;

            // Try to place new points around it
            for _ in 0..max_attempts {
                let candidate = self.random_point_in_annulus(center);

                if self.is_valid(&candidate) {
                    self.add_point(candidate);
                    found = true;
                    break;
                }
            }

            // If no valid point found, remove from active list
            if !found {
                self.active.swap_remove(active_idx);
            }
        }

        std::mem::take(&mut self.points)
    }

    /// Adds a point to the sample.
    fn add_point(&mut self, p: Point2<F>) {
        let idx = self.points.len();
        self.points.push(p);
        self.active.push(idx);

        // Add to grid
        let (gx, gy) = self.grid_coords(p);
        if gx < self.grid_width && gy < self.grid_height {
            self.grid[gy * self.grid_width + gx] = Some(idx);
        }
    }

    /// Checks if a point is valid (inside domain and far enough from existing points).
    fn is_valid(&self, p: &Point2<F>) -> bool {
        // Check bounds
        if p.x < F::zero() || p.x >= self.width || p.y < F::zero() || p.y >= self.height {
            return false;
        }

        // Check distance to nearby points using grid
        let (gx, gy) = self.grid_coords(*p);
        let search_radius = 2; // Check 5x5 neighborhood

        let min_gx = gx.saturating_sub(search_radius);
        let max_gx = (gx + search_radius + 1).min(self.grid_width);
        let min_gy = gy.saturating_sub(search_radius);
        let max_gy = (gy + search_radius + 1).min(self.grid_height);

        let min_dist_sq = self.min_distance * self.min_distance;

        for cy in min_gy..max_gy {
            for cx in min_gx..max_gx {
                if let Some(idx) = self.grid[cy * self.grid_width + cx] {
                    let other = self.points[idx];
                    if p.distance_squared(other) < min_dist_sq {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Converts a point to grid coordinates.
    fn grid_coords(&self, p: Point2<F>) -> (usize, usize) {
        let gx = (p.x / self.cell_size).floor().to_usize().unwrap_or(0);
        let gy = (p.y / self.cell_size).floor().to_usize().unwrap_or(0);
        (gx, gy)
    }

    /// Generates a random point in the domain.
    fn random_point_in_domain(&mut self) -> Point2<F> {
        let x = self.random_f() * self.width;
        let y = self.random_f() * self.height;
        Point2::new(x, y)
    }

    /// Generates a random point in the annulus [r, 2r] around center.
    fn random_point_in_annulus(&mut self, center: Point2<F>) -> Point2<F> {
        let one = F::one();
        let two = one + one;
        let pi = F::from(std::f64::consts::PI).unwrap();

        // Random angle
        let angle = self.random_f() * two * pi;

        // Random radius in [r, 2r]
        let r = self.min_distance * (one + self.random_f());

        Point2::new(center.x + r * angle.cos(), center.y + r * angle.sin())
    }

    /// Simple xorshift64 PRNG - returns value in [0, 1).
    fn random_f(&mut self) -> F {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;

        // Convert to [0, 1)
        let max = u64::MAX as f64;
        F::from(self.rng_state as f64 / max).unwrap()
    }

    /// Returns a random usize in [0, max).
    fn random_usize(&mut self, max: usize) -> usize {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;

        (self.rng_state as usize) % max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_disk_basic() {
        let points: Vec<Point2<f64>> = poisson_disk(10.0, 10.0, 1.0, 30);

        // Should generate some points
        assert!(!points.is_empty());

        // All points should be in bounds
        for p in &points {
            assert!(p.x >= 0.0 && p.x < 10.0);
            assert!(p.y >= 0.0 && p.y < 10.0);
        }
    }

    #[test]
    fn test_poisson_disk_min_distance() {
        let min_dist = 1.5;
        let points: Vec<Point2<f64>> = poisson_disk(20.0, 20.0, min_dist, 30);

        // Check minimum distance property
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dist = points[i].distance(points[j]);
                // Allow small tolerance for floating point
                assert!(
                    dist >= min_dist * 0.99,
                    "Points {} and {} too close: {} < {}",
                    i,
                    j,
                    dist,
                    min_dist
                );
            }
        }
    }

    #[test]
    fn test_poisson_disk_deterministic() {
        let seed = 12345u64;

        let points1: Vec<Point2<f64>> = poisson_disk_with_seed(10.0, 10.0, 1.0, 30, seed);
        let points2: Vec<Point2<f64>> = poisson_disk_with_seed(10.0, 10.0, 1.0, 30, seed);

        // Same seed should produce same results
        assert_eq!(points1.len(), points2.len());
        for (p1, p2) in points1.iter().zip(points2.iter()) {
            assert_eq!(p1.x, p2.x);
            assert_eq!(p1.y, p2.y);
        }
    }

    #[test]
    fn test_poisson_disk_different_seeds() {
        let points1: Vec<Point2<f64>> = poisson_disk_with_seed(10.0, 10.0, 1.0, 30, 111);
        let points2: Vec<Point2<f64>> = poisson_disk_with_seed(10.0, 10.0, 1.0, 30, 222);

        // Different seeds should produce different results
        // (at least different point counts or positions)
        let same = points1.len() == points2.len()
            && points1
                .iter()
                .zip(points2.iter())
                .all(|(a, b)| a.x == b.x && a.y == b.y);
        assert!(!same);
    }

    #[test]
    fn test_poisson_disk_coverage() {
        // With small min_distance, should generate many points
        let points: Vec<Point2<f64>> = poisson_disk(10.0, 10.0, 0.5, 30);

        // Rough estimate: area / (pi * r^2) gives approximate max points
        // With r=0.5, area=100, we expect roughly 100 / (pi * 0.25) ≈ 127 points
        // But actual count depends on packing, typically 60-80% efficient
        assert!(points.len() > 50);
    }

    #[test]
    fn test_poisson_disk_large_distance() {
        // With large min_distance, should generate few points
        let points: Vec<Point2<f64>> = poisson_disk(10.0, 10.0, 5.0, 30);

        // With r=5 in 10x10 area, expect only a handful of points
        assert!(points.len() < 20);
        assert!(!points.is_empty());
    }

    #[test]
    fn test_poisson_disk_small_domain() {
        let points: Vec<Point2<f64>> = poisson_disk(1.0, 1.0, 0.3, 30);

        assert!(!points.is_empty());

        // All should be in bounds
        for p in &points {
            assert!(p.x >= 0.0 && p.x < 1.0);
            assert!(p.y >= 0.0 && p.y < 1.0);
        }
    }

    #[test]
    fn test_poisson_disk_f32() {
        let points: Vec<Point2<f32>> = poisson_disk(10.0f32, 10.0f32, 1.0f32, 30);

        assert!(!points.is_empty());

        for p in &points {
            assert!(p.x >= 0.0 && p.x < 10.0);
            assert!(p.y >= 0.0 && p.y < 10.0);
        }
    }

    #[test]
    fn test_poisson_sampler_reuse() {
        let mut sampler: PoissonDiskSampler<f64> = PoissonDiskSampler::new(10.0, 10.0, 1.0, 12345);

        let points1 = sampler.generate(30);
        assert!(!points1.is_empty());

        // Reset and generate again (will continue from RNG state)
        let points2 = sampler.generate(30);
        assert!(!points2.is_empty());

        // Results should be different (RNG state advanced)
        assert_ne!(points1.len(), points2.len());
    }

    #[test]
    fn test_poisson_disk_aspect_ratio() {
        // Test with non-square domain
        let points: Vec<Point2<f64>> = poisson_disk(20.0, 5.0, 1.0, 30);

        assert!(!points.is_empty());

        for p in &points {
            assert!(p.x >= 0.0 && p.x < 20.0);
            assert!(p.y >= 0.0 && p.y < 5.0);
        }

        // Check min distance
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                assert!(points[i].distance(points[j]) >= 0.99);
            }
        }
    }
}
