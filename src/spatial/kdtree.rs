//! KD-tree for efficient spatial point queries.
//!
//! A KD-tree is a space-partitioning data structure for organizing points
//! in k-dimensional space. This implementation is specialized for 2D points
//! and provides efficient nearest neighbor and range queries.
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, spatial::KdTree};
//!
//! let points: Vec<Point2<f64>> = vec![
//!     Point2::new(2.0, 3.0),
//!     Point2::new(5.0, 4.0),
//!     Point2::new(9.0, 6.0),
//!     Point2::new(4.0, 7.0),
//!     Point2::new(8.0, 1.0),
//!     Point2::new(7.0, 2.0),
//! ];
//!
//! let tree = KdTree::build(&points);
//!
//! // Find nearest neighbor
//! let query = Point2::new(5.0, 5.0);
//! if let Some((idx, dist)) = tree.nearest(&points, query) {
//!     println!("Nearest point: {:?} at distance {}", points[idx], dist);
//! }
//!
//! // Find k nearest neighbors
//! let neighbors = tree.k_nearest(&points, query, 3);
//! ```

use crate::bounds::Aabb2;
use crate::primitives::Point2;
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A node in the KD-tree.
#[derive(Debug, Clone)]
enum KdNode {
    /// A leaf node containing a single point index.
    Leaf {
        /// Index of the point in the original array.
        index: usize,
    },
    /// An internal node that splits space along an axis.
    Internal {
        /// Index of the point at this node (median point).
        index: usize,
        /// The axis along which this node splits (0 = x, 1 = y).
        axis: u8,
        /// Left child (points with smaller coordinate on split axis).
        left: Option<Box<KdNode>>,
        /// Right child (points with larger coordinate on split axis).
        right: Option<Box<KdNode>>,
    },
}

/// A 2D KD-tree for efficient spatial queries on point sets.
///
/// The KD-tree stores indices into an external point array, allowing
/// the original data to remain in place.
///
/// # Construction
///
/// The tree is built using the median-of-medians approach for balanced
/// construction, alternating split axes at each level.
///
/// # Complexity
///
/// - Construction: O(n log n)
/// - Nearest neighbor: O(log n) average, O(n) worst case
/// - K-nearest neighbors: O(k log n) average
/// - Range query: O(√n + k) where k is the number of results
#[derive(Debug, Clone)]
pub struct KdTree {
    root: Option<Box<KdNode>>,
    size: usize,
}

impl KdTree {
    /// Builds a KD-tree from a slice of points.
    ///
    /// Returns an empty tree if the input is empty.
    pub fn build<F: Float>(points: &[Point2<F>]) -> Self {
        if points.is_empty() {
            return KdTree {
                root: None,
                size: 0,
            };
        }

        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build_recursive(points, &mut indices, 0);

        KdTree {
            root: Some(root),
            size: points.len(),
        }
    }

    /// Recursively builds the tree.
    fn build_recursive<F: Float>(
        points: &[Point2<F>],
        indices: &mut [usize],
        depth: usize,
    ) -> Box<KdNode> {
        let axis = (depth % 2) as u8;

        if indices.len() == 1 {
            return Box::new(KdNode::Leaf { index: indices[0] });
        }

        // Sort indices by the current axis
        indices.sort_by(|&a, &b| {
            let val_a = if axis == 0 {
                points[a].x
            } else {
                points[a].y
            };
            let val_b = if axis == 0 {
                points[b].x
            } else {
                points[b].y
            };
            val_a.partial_cmp(&val_b).unwrap_or(Ordering::Equal)
        });

        let median = indices.len() / 2;
        let median_index = indices[median];

        let left = if median > 0 {
            Some(Self::build_recursive(
                points,
                &mut indices[..median],
                depth + 1,
            ))
        } else {
            None
        };

        let right = if median + 1 < indices.len() {
            Some(Self::build_recursive(
                points,
                &mut indices[median + 1..],
                depth + 1,
            ))
        } else {
            None
        };

        Box::new(KdNode::Internal {
            index: median_index,
            axis,
            left,
            right,
        })
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Finds the nearest neighbor to a query point.
    ///
    /// Returns the index of the nearest point and its distance, or None if empty.
    pub fn nearest<F: Float>(&self, points: &[Point2<F>], query: Point2<F>) -> Option<(usize, F)> {
        let root = self.root.as_ref()?;
        let mut best: Option<(usize, F)> = None;
        Self::nearest_recursive(points, root, query, &mut best);
        best
    }

    /// Recursive nearest neighbor search.
    fn nearest_recursive<F: Float>(
        points: &[Point2<F>],
        node: &KdNode,
        query: Point2<F>,
        best: &mut Option<(usize, F)>,
    ) {
        match node {
            KdNode::Leaf { index } => {
                let dist = points[*index].distance(query);
                match best {
                    None => *best = Some((*index, dist)),
                    Some((_, best_dist)) if dist < *best_dist => *best = Some((*index, dist)),
                    _ => {}
                }
            }
            KdNode::Internal {
                index,
                axis,
                left,
                right,
            } => {
                // Check the current node's point
                let point = points[*index];
                let dist = point.distance(query);
                match best {
                    None => *best = Some((*index, dist)),
                    Some((_, best_dist)) if dist < *best_dist => *best = Some((*index, dist)),
                    _ => {}
                }

                // Determine which side to search first
                let (query_val, point_val) = if *axis == 0 {
                    (query.x, point.x)
                } else {
                    (query.y, point.y)
                };

                let (first, second) = if query_val < point_val {
                    (left, right)
                } else {
                    (right, left)
                };

                // Search the closer side first
                if let Some(child) = first {
                    Self::nearest_recursive(points, child, query, best);
                }

                // Check if we need to search the other side
                let axis_dist = (query_val - point_val).abs();
                let should_search_other = match best {
                    None => true,
                    Some((_, best_dist)) => axis_dist < *best_dist,
                };

                if should_search_other {
                    if let Some(child) = second {
                        Self::nearest_recursive(points, child, query, best);
                    }
                }
            }
        }
    }

    /// Finds the k nearest neighbors to a query point.
    ///
    /// Returns a vector of (index, distance) pairs sorted by distance (closest first).
    pub fn k_nearest<F: Float>(
        &self,
        points: &[Point2<F>],
        query: Point2<F>,
        k: usize,
    ) -> Vec<(usize, F)> {
        if k == 0 || self.root.is_none() {
            return Vec::new();
        }

        let root = self.root.as_ref().unwrap();
        let mut heap: BinaryHeap<HeapEntry<F>> = BinaryHeap::new();
        Self::k_nearest_recursive(points, root, query, k, &mut heap);

        // Extract results and sort by distance
        let mut results: Vec<(usize, F)> = heap.into_iter().map(|e| (e.index, e.dist)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Recursive k-nearest neighbor search using a max-heap.
    fn k_nearest_recursive<F: Float>(
        points: &[Point2<F>],
        node: &KdNode,
        query: Point2<F>,
        k: usize,
        heap: &mut BinaryHeap<HeapEntry<F>>,
    ) {
        match node {
            KdNode::Leaf { index } => {
                let dist = points[*index].distance(query);
                Self::heap_insert(heap, *index, dist, k);
            }
            KdNode::Internal {
                index,
                axis,
                left,
                right,
            } => {
                // Check current point
                let point = points[*index];
                let dist = point.distance(query);
                Self::heap_insert(heap, *index, dist, k);

                // Determine search order
                let (query_val, point_val) = if *axis == 0 {
                    (query.x, point.x)
                } else {
                    (query.y, point.y)
                };

                let (first, second) = if query_val < point_val {
                    (left, right)
                } else {
                    (right, left)
                };

                // Search closer side first
                if let Some(child) = first {
                    Self::k_nearest_recursive(points, child, query, k, heap);
                }

                // Check if other side could contain closer points
                let axis_dist = (query_val - point_val).abs();
                let should_search = heap.len() < k
                    || heap
                        .peek()
                        .map(|e| axis_dist < e.dist)
                        .unwrap_or(true);

                if should_search {
                    if let Some(child) = second {
                        Self::k_nearest_recursive(points, child, query, k, heap);
                    }
                }
            }
        }
    }

    /// Inserts into the max-heap, maintaining size k.
    fn heap_insert<F: Float>(heap: &mut BinaryHeap<HeapEntry<F>>, index: usize, dist: F, k: usize) {
        if heap.len() < k {
            heap.push(HeapEntry { index, dist });
        } else if let Some(max) = heap.peek() {
            if dist < max.dist {
                heap.pop();
                heap.push(HeapEntry { index, dist });
            }
        }
    }

    /// Finds all points within a given distance of a query point.
    ///
    /// Returns indices of all points within the radius.
    pub fn within_radius<F: Float>(
        &self,
        points: &[Point2<F>],
        query: Point2<F>,
        radius: F,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::radius_recursive(points, root, query, radius, &mut results);
        }
        results
    }

    /// Recursive radius search.
    fn radius_recursive<F: Float>(
        points: &[Point2<F>],
        node: &KdNode,
        query: Point2<F>,
        radius: F,
        results: &mut Vec<usize>,
    ) {
        match node {
            KdNode::Leaf { index } => {
                if points[*index].distance(query) <= radius {
                    results.push(*index);
                }
            }
            KdNode::Internal {
                index,
                axis,
                left,
                right,
            } => {
                // Check current point
                let point = points[*index];
                if point.distance(query) <= radius {
                    results.push(*index);
                }

                let (query_val, point_val) = if *axis == 0 {
                    (query.x, point.x)
                } else {
                    (query.y, point.y)
                };

                // Search left if it could contain points within radius
                if let Some(child) = left {
                    if query_val - radius <= point_val {
                        Self::radius_recursive(points, child, query, radius, results);
                    }
                }

                // Search right if it could contain points within radius
                if let Some(child) = right {
                    if query_val + radius >= point_val {
                        Self::radius_recursive(points, child, query, radius, results);
                    }
                }
            }
        }
    }

    /// Finds all points within an axis-aligned bounding box.
    ///
    /// Returns indices of all points inside the box.
    pub fn within_aabb<F: Float>(&self, points: &[Point2<F>], aabb: Aabb2<F>) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            Self::aabb_recursive(points, root, aabb, &mut results, 0);
        }
        results
    }

    /// Recursive AABB range search.
    fn aabb_recursive<F: Float>(
        points: &[Point2<F>],
        node: &KdNode,
        aabb: Aabb2<F>,
        results: &mut Vec<usize>,
        depth: usize,
    ) {
        match node {
            KdNode::Leaf { index } => {
                if aabb.contains_point(points[*index]) {
                    results.push(*index);
                }
            }
            KdNode::Internal {
                index,
                axis,
                left,
                right,
            } => {
                // Check current point
                let point = points[*index];
                if aabb.contains_point(point) {
                    results.push(*index);
                }

                let (point_val, min_val, max_val) = if *axis == 0 {
                    (point.x, aabb.min.x, aabb.max.x)
                } else {
                    (point.y, aabb.min.y, aabb.max.y)
                };

                // Search left if range overlaps
                if let Some(child) = left {
                    if min_val <= point_val {
                        Self::aabb_recursive(points, child, aabb, results, depth + 1);
                    }
                }

                // Search right if range overlaps
                if let Some(child) = right {
                    if max_val >= point_val {
                        Self::aabb_recursive(points, child, aabb, results, depth + 1);
                    }
                }
            }
        }
    }

    /// Finds all pairs of points that are within a given distance of each other.
    ///
    /// This is more efficient than checking all O(n²) pairs.
    /// Returns pairs of indices (i, j) where i < j.
    pub fn pairs_within_distance<F: Float>(
        &self,
        points: &[Point2<F>],
        distance: F,
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        for (i, point) in points.iter().enumerate() {
            let neighbors = self.within_radius(points, *point, distance);
            for j in neighbors {
                if i < j {
                    pairs.push((i, j));
                }
            }
        }

        pairs
    }
}

/// A heap entry for k-nearest neighbor search.
/// Uses reverse ordering so BinaryHeap acts as a max-heap on distance.
struct HeapEntry<F> {
    index: usize,
    dist: F,
}

impl<F: Float> PartialEq for HeapEntry<F> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<F: Float> Eq for HeapEntry<F> {}

impl<F: Float> PartialOrd for HeapEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<F: Float> Ord for HeapEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points() -> Vec<Point2<f64>> {
        vec![
            Point2::new(2.0, 3.0),
            Point2::new(5.0, 4.0),
            Point2::new(9.0, 6.0),
            Point2::new(4.0, 7.0),
            Point2::new(8.0, 1.0),
            Point2::new(7.0, 2.0),
        ]
    }

    #[test]
    fn test_build_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let tree = KdTree::build(&points);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_build_single() {
        let points = vec![Point2::new(1.0, 2.0)];
        let tree = KdTree::build(&points);
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_build_multiple() {
        let points = sample_points();
        let tree = KdTree::build(&points);
        assert_eq!(tree.len(), 6);
    }

    #[test]
    fn test_nearest_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let tree = KdTree::build(&points);
        assert!(tree.nearest(&points, Point2::new(0.0, 0.0)).is_none());
    }

    #[test]
    fn test_nearest_single() {
        let points = vec![Point2::new(5.0, 5.0)];
        let tree = KdTree::build(&points);

        let (idx, dist) = tree.nearest(&points, Point2::new(3.0, 3.0)).unwrap();
        assert_eq!(idx, 0);
        assert!((dist - (8.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_exact_match() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        // Query at exact point location
        let (idx, dist) = tree.nearest(&points, Point2::new(5.0, 4.0)).unwrap();
        assert_eq!(points[idx], Point2::new(5.0, 4.0));
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_nearest_general() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        // Query near (7, 2) - should find that point
        let (idx, _) = tree.nearest(&points, Point2::new(7.1, 1.9)).unwrap();
        assert_eq!(points[idx], Point2::new(7.0, 2.0));
    }

    #[test]
    fn test_k_nearest_k_zero() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        let results = tree.k_nearest(&points, Point2::new(5.0, 5.0), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_k_nearest_k_one() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        let results = tree.k_nearest(&points, Point2::new(5.0, 5.0), 1);
        assert_eq!(results.len(), 1);

        // Should match nearest()
        let (nearest_idx, _) = tree.nearest(&points, Point2::new(5.0, 5.0)).unwrap();
        assert_eq!(results[0].0, nearest_idx);
    }

    #[test]
    fn test_k_nearest_multiple() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        let results = tree.k_nearest(&points, Point2::new(5.0, 5.0), 3);
        assert_eq!(results.len(), 3);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_k_nearest_all() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        // Request more than available
        let results = tree.k_nearest(&points, Point2::new(0.0, 0.0), 100);
        assert_eq!(results.len(), 6); // All points
    }

    #[test]
    fn test_within_radius_none() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        // Very small radius at a point far from any sample point
        let results = tree.within_radius(&points, Point2::new(0.0, 0.0), 0.5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_within_radius_some() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        // Radius 2 around (5, 4) should find (5, 4) at least
        let results = tree.within_radius(&points, Point2::new(5.0, 4.0), 2.0);
        assert!(!results.is_empty());

        // Verify all results are actually within radius
        for idx in &results {
            assert!(points[*idx].distance(Point2::new(5.0, 4.0)) <= 2.0);
        }
    }

    #[test]
    fn test_within_radius_all() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        // Very large radius should find all points
        let results = tree.within_radius(&points, Point2::new(5.0, 4.0), 100.0);
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_within_aabb_none() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        let aabb = Aabb2::new(Point2::new(100.0, 100.0), Point2::new(200.0, 200.0));
        let results = tree.within_aabb(&points, aabb);
        assert!(results.is_empty());
    }

    #[test]
    fn test_within_aabb_some() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        // Box around lower-left region
        let aabb = Aabb2::new(Point2::new(1.0, 1.0), Point2::new(6.0, 5.0));
        let results = tree.within_aabb(&points, aabb);

        // Should include (2,3), (5,4)
        assert!(!results.is_empty());

        for idx in &results {
            assert!(aabb.contains_point(points[*idx]));
        }
    }

    #[test]
    fn test_within_aabb_all() {
        let points = sample_points();
        let tree = KdTree::build(&points);

        let aabb = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(20.0, 20.0));
        let results = tree.within_aabb(&points, aabb);
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_pairs_within_distance() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(10.0, 10.0),
        ];
        let tree = KdTree::build(&points);

        let pairs = tree.pairs_within_distance(&points, 1.5);

        // Only (0, 1) should be within distance 1.5
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1));
    }

    #[test]
    fn test_f32_support() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(1.0, 2.0),
            Point2::new(3.0, 4.0),
            Point2::new(5.0, 6.0),
        ];
        let tree = KdTree::build(&points);

        let (idx, _) = tree.nearest(&points, Point2::new(3.0, 4.0)).unwrap();
        assert_eq!(points[idx], Point2::new(3.0, 4.0));
    }

    #[test]
    fn test_grid_points() {
        // Test with a regular grid
        let mut points = Vec::new();
        for x in 0..10 {
            for y in 0..10 {
                points.push(Point2::new(x as f64, y as f64));
            }
        }
        let tree = KdTree::build(&points);

        // Nearest to center
        let (idx, _) = tree.nearest(&points, Point2::new(4.5, 4.5)).unwrap();
        let nearest = points[idx];
        assert!(nearest.x >= 4.0 && nearest.x <= 5.0);
        assert!(nearest.y >= 4.0 && nearest.y <= 5.0);

        // k-nearest around center
        let results = tree.k_nearest(&points, Point2::new(4.5, 4.5), 4);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_collinear_points() {
        // All points on a line
        let points: Vec<Point2<f64>> = (0..10).map(|i| Point2::new(i as f64, 0.0)).collect();
        let tree = KdTree::build(&points);

        let (idx, _) = tree.nearest(&points, Point2::new(5.5, 0.0)).unwrap();
        assert!(points[idx].x == 5.0 || points[idx].x == 6.0);
    }

    #[test]
    fn test_duplicate_points() {
        let points = vec![
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(5.0, 5.0),
        ];
        let tree = KdTree::build(&points);

        let (idx, dist) = tree.nearest(&points, Point2::new(1.0, 1.0)).unwrap();
        assert!(dist < 1e-10);
        assert!(points[idx] == Point2::new(1.0, 1.0));
    }

    #[test]
    fn test_stress_large() {
        // Build a larger tree
        let mut points = Vec::new();
        for i in 0..1000 {
            let x = (i * 7 % 100) as f64;
            let y = (i * 13 % 100) as f64;
            points.push(Point2::new(x, y));
        }
        let tree = KdTree::build(&points);

        // Verify nearest is actually nearest (brute force check)
        let query = Point2::new(50.0, 50.0);
        let (_kd_idx, kd_dist) = tree.nearest(&points, query).unwrap();

        let brute_min = points
            .iter()
            .map(|p| p.distance(query))
            .fold(f64::MAX, |a, b| a.min(b));

        assert!((kd_dist - brute_min).abs() < 1e-10);
    }
}
