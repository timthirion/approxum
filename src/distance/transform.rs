//! Grid-based distance transform algorithms.
//!
//! Distance transforms compute the distance from each pixel/cell to the nearest
//! boundary or feature point in a binary image.

use num_traits::Float;

/// Computes the Euclidean distance transform of a binary grid.
///
/// Uses a two-pass algorithm based on Meijster et al. for O(n) complexity.
///
/// # Arguments
///
/// * `grid` - Binary grid where `true` represents feature/boundary pixels
/// * `width` - Grid width
/// * `height` - Grid height
///
/// # Returns
///
/// A vector of distances where each cell contains the Euclidean distance
/// to the nearest `true` cell in the input.
///
/// # Example
///
/// ```
/// use approxum::distance::distance_transform;
///
/// // Create a 5x5 grid with a single point in the center
/// let mut grid = vec![false; 25];
/// grid[12] = true; // Center point
///
/// let distances = distance_transform::<f64>(&grid, 5, 5);
///
/// // Center should be 0
/// assert!(distances[12] < 0.01);
///
/// // Corners should be sqrt(8) ≈ 2.83
/// assert!((distances[0] - 2.828).abs() < 0.01);
/// ```
pub fn distance_transform<F: Float>(grid: &[bool], width: usize, height: usize) -> Vec<F> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let inf = F::from(width + height).unwrap();

    // Phase 1: Compute 1D distance transform along rows
    let mut g = vec![inf; width * height];

    for y in 0..height {
        // Forward pass
        let mut dist = inf;
        for x in 0..width {
            let idx = y * width + x;
            if grid[idx] {
                dist = F::zero();
            } else if dist < inf {
                dist = dist + F::one();
            }
            g[idx] = dist;
        }

        // Backward pass
        dist = inf;
        for x in (0..width).rev() {
            let idx = y * width + x;
            if grid[idx] {
                dist = F::zero();
            } else if dist < inf {
                dist = dist + F::one();
            }
            if dist < g[idx] {
                g[idx] = dist;
            }
        }
    }

    // Phase 2: Compute 2D Euclidean distance using column scans
    let mut result = vec![F::zero(); width * height];
    let mut s = vec![0i64; height]; // Locations of parabola vertices
    let mut t = vec![F::zero(); height + 1]; // Range boundaries

    for x in 0..width {
        // Build lower envelope of parabolas
        let mut q = 0i64; // Current parabola index

        s[0] = 0;
        t[0] = F::from(-1e20).unwrap_or(F::neg_infinity());
        t[1] = F::from(1e20).unwrap_or(F::infinity());

        for u in 1..height {
            let u_i64 = u as i64;
            loop {
                let v = s[q as usize];
                // Intersection of parabolas at v and u
                let g_v = g[v as usize * width + x];
                let g_u = g[u * width + x];
                let v_f = F::from(v).unwrap();
                let u_f = F::from(u_i64).unwrap();

                // (u² + g[u]² - v² - g[v]²) / (2(u - v))
                let intersection = ((u_f * u_f + g_u * g_u) - (v_f * v_f + g_v * g_v))
                    / (F::from(2).unwrap() * (u_f - v_f));

                if intersection > t[q as usize] {
                    q += 1;
                    s[q as usize] = u_i64;
                    t[q as usize] = intersection;
                    t[q as usize + 1] = F::from(1e20).unwrap_or(F::infinity());
                    break;
                }
                q -= 1;
                if q < 0 {
                    q = 0;
                    s[0] = u_i64;
                    t[0] = F::from(-1e20).unwrap_or(F::neg_infinity());
                    t[1] = F::from(1e20).unwrap_or(F::infinity());
                    break;
                }
            }
        }

        // Fill in distance values
        q = 0;
        for u in 0..height {
            let u_f = F::from(u).unwrap();
            while t[q as usize + 1] < u_f {
                q += 1;
            }
            let v = s[q as usize];
            let v_f = F::from(v).unwrap();
            let g_v = g[v as usize * width + x];
            let dy = u_f - v_f;
            result[u * width + x] = (dy * dy + g_v * g_v).sqrt();
        }
    }

    result
}

/// Computes a signed distance transform from a binary grid.
///
/// Points where `grid[i]` is `true` are considered "inside".
/// Returns negative distances for inside points, positive for outside.
///
/// # Arguments
///
/// * `grid` - Binary grid where `true` represents inside pixels
/// * `width` - Grid width
/// * `height` - Grid height
///
/// # Returns
///
/// A vector of signed distances.
pub fn signed_distance_transform<F: Float>(grid: &[bool], width: usize, height: usize) -> Vec<F> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    // Find boundary pixels (pixels that are inside but adjacent to outside)
    let mut boundary = vec![false; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if grid[idx] {
                // Check if this inside pixel is adjacent to an outside pixel
                let has_outside_neighbor = (x > 0 && !grid[idx - 1])
                    || (x + 1 < width && !grid[idx + 1])
                    || (y > 0 && !grid[idx - width])
                    || (y + 1 < height && !grid[idx + width]);

                if has_outside_neighbor {
                    boundary[idx] = true;
                }
            } else {
                // Check if this outside pixel is adjacent to an inside pixel
                let has_inside_neighbor = (x > 0 && grid[idx - 1])
                    || (x + 1 < width && grid[idx + 1])
                    || (y > 0 && grid[idx - width])
                    || (y + 1 < height && grid[idx + width]);

                if has_inside_neighbor {
                    boundary[idx] = true;
                }
            }
        }
    }

    // Compute distance to boundary
    let distances = distance_transform::<F>(&boundary, width, height);

    // Apply sign based on inside/outside
    distances
        .iter()
        .enumerate()
        .map(|(i, &d)| if grid[i] { -d } else { d })
        .collect()
}

/// Computes a chamfer distance transform (faster approximation).
///
/// Uses 3-4 chamfer weights for a good approximation of Euclidean distance.
///
/// # Arguments
///
/// * `grid` - Binary grid where `true` represents feature pixels
/// * `width` - Grid width
/// * `height` - Grid height
///
/// # Returns
///
/// Approximate distances (multiply by 1/3 for normalized result).
pub fn chamfer_distance_transform<F: Float>(grid: &[bool], width: usize, height: usize) -> Vec<F> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let inf = F::from((width + height) * 4).unwrap();
    let three = F::from(3).unwrap();
    let four = F::from(4).unwrap();

    let mut dist = vec![inf; width * height];

    // Initialize feature pixels
    for (i, &is_feature) in grid.iter().enumerate() {
        if is_feature {
            dist[i] = F::zero();
        }
    }

    // Forward pass (top-left to bottom-right)
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let mut d = dist[idx];

            // Check neighbors: NW, N, NE, W
            if y > 0 {
                if x > 0 {
                    d = d.min(dist[(y - 1) * width + (x - 1)] + four);
                }
                d = d.min(dist[(y - 1) * width + x] + three);
                if x + 1 < width {
                    d = d.min(dist[(y - 1) * width + (x + 1)] + four);
                }
            }
            if x > 0 {
                d = d.min(dist[y * width + (x - 1)] + three);
            }

            dist[idx] = d;
        }
    }

    // Backward pass (bottom-right to top-left)
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let idx = y * width + x;
            let mut d = dist[idx];

            // Check neighbors: E, SW, S, SE
            if x + 1 < width {
                d = d.min(dist[y * width + (x + 1)] + three);
            }
            if y + 1 < height {
                if x > 0 {
                    d = d.min(dist[(y + 1) * width + (x - 1)] + four);
                }
                d = d.min(dist[(y + 1) * width + x] + three);
                if x + 1 < width {
                    d = d.min(dist[(y + 1) * width + (x + 1)] + four);
                }
            }

            dist[idx] = d;
        }
    }

    // Normalize (divide by 3)
    dist.iter().map(|&d| d / three).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_transform_single_point() {
        // 5x5 grid with single point in center
        let mut grid = vec![false; 25];
        grid[12] = true; // (2, 2)

        let dist: Vec<f64> = distance_transform(&grid, 5, 5);

        // Center should be 0
        assert!(dist[12] < 1e-10);

        // Adjacent cells should be 1
        assert!((dist[7] - 1.0).abs() < 1e-10); // (2, 1)
        assert!((dist[11] - 1.0).abs() < 1e-10); // (1, 2)
        assert!((dist[13] - 1.0).abs() < 1e-10); // (3, 2)
        assert!((dist[17] - 1.0).abs() < 1e-10); // (2, 3)

        // Diagonal cells should be sqrt(2)
        let sqrt2 = 2.0_f64.sqrt();
        assert!((dist[6] - sqrt2).abs() < 1e-10); // (1, 1)
        assert!((dist[8] - sqrt2).abs() < 1e-10); // (3, 1)

        // Corners should be sqrt(8)
        let sqrt8 = 8.0_f64.sqrt();
        assert!((dist[0] - sqrt8).abs() < 1e-10); // (0, 0)
        assert!((dist[4] - sqrt8).abs() < 1e-10); // (4, 0)
        assert!((dist[20] - sqrt8).abs() < 1e-10); // (0, 4)
        assert!((dist[24] - sqrt8).abs() < 1e-10); // (4, 4)
    }

    #[test]
    fn test_distance_transform_line() {
        // Vertical line at x=2
        let mut grid = vec![false; 25];
        for y in 0..5 {
            grid[y * 5 + 2] = true;
        }

        let dist: Vec<f64> = distance_transform(&grid, 5, 5);

        // Points on the line should be 0
        for y in 0..5 {
            assert!(dist[y * 5 + 2] < 1e-10);
        }

        // Points one away should be 1
        for y in 0..5 {
            assert!((dist[y * 5 + 1] - 1.0).abs() < 1e-10);
            assert!((dist[y * 5 + 3] - 1.0).abs() < 1e-10);
        }

        // Points two away should be 2
        for y in 0..5 {
            assert!((dist[y * 5 + 0] - 2.0).abs() < 1e-10);
            assert!((dist[y * 5 + 4] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_distance_transform_all_features() {
        // All cells are features
        let grid = vec![true; 25];
        let dist: Vec<f64> = distance_transform(&grid, 5, 5);

        // All distances should be 0
        for d in dist {
            assert!(d < 1e-10);
        }
    }

    #[test]
    fn test_signed_distance_transform() {
        // Create a filled square in the middle
        let mut grid = vec![false; 49]; // 7x7
        for y in 2..5 {
            for x in 2..5 {
                grid[y * 7 + x] = true;
            }
        }

        let dist: Vec<f64> = signed_distance_transform(&grid, 7, 7);

        // Inside the square should be negative
        assert!(dist[3 * 7 + 3] < 0.0); // Center

        // Outside should be positive
        assert!(dist[0] > 0.0); // Corner
        assert!(dist[6] > 0.0); // Corner
    }

    #[test]
    fn test_chamfer_distance() {
        // Single point in center
        let mut grid = vec![false; 25];
        grid[12] = true;

        let dist: Vec<f64> = chamfer_distance_transform(&grid, 5, 5);

        // Center should be 0
        assert!(dist[12] < 1e-10);

        // Adjacent should be ~1
        assert!((dist[7] - 1.0).abs() < 0.1);
        assert!((dist[11] - 1.0).abs() < 0.1);

        // Diagonal should be ~sqrt(2) ≈ 1.41, chamfer gives 4/3 ≈ 1.33
        let expected_diag = 4.0 / 3.0;
        assert!((dist[6] - expected_diag).abs() < 0.1);
    }

    #[test]
    fn test_empty_grid() {
        let dist: Vec<f64> = distance_transform(&[], 0, 0);
        assert!(dist.is_empty());
    }

    #[test]
    fn test_f32_distance_transform() {
        let mut grid = vec![false; 9];
        grid[4] = true; // Center at (1, 1)

        let dist: Vec<f32> = distance_transform(&grid, 3, 3);

        // Center should be 0
        assert!(dist[4] < 1e-5);
        // Corner at (0, 0) is sqrt(1² + 1²) = sqrt(2) from center
        assert!((dist[0] - 2.0_f32.sqrt()).abs() < 1e-5);
    }
}
