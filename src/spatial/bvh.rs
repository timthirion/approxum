//! Bounding Volume Hierarchy for spatial queries.
//!
//! A BVH is a tree structure where each node contains an axis-aligned bounding
//! box (AABB) that encloses all primitives in its subtree. This enables efficient
//! spatial queries like intersection tests and nearest-neighbor searches.

use crate::bounds::Aabb2;
use crate::primitives::Point2;
use num_traits::Float;

/// A trait for objects that can be bounded and stored in a BVH.
pub trait Bounded<F: Float> {
    /// Returns the axis-aligned bounding box of this object.
    fn bounds(&self) -> Aabb2<F>;

    /// Returns the centroid of this object (used for partitioning).
    fn centroid(&self) -> Point2<F> {
        self.bounds().center()
    }
}

// Implement Bounded for Point2
impl<F: Float> Bounded<F> for Point2<F> {
    fn bounds(&self) -> Aabb2<F> {
        Aabb2::from_point(*self)
    }

    fn centroid(&self) -> Point2<F> {
        *self
    }
}

// Implement Bounded for Aabb2
impl<F: Float> Bounded<F> for Aabb2<F> {
    fn bounds(&self) -> Aabb2<F> {
        *self
    }
}

/// A node in the BVH tree.
#[derive(Debug, Clone)]
pub enum BvhNode<F> {
    /// A leaf node containing indices into the primitive array.
    Leaf {
        /// Bounding box of all primitives in this leaf.
        bounds: Aabb2<F>,
        /// Starting index in the primitive index array.
        first: usize,
        /// Number of primitives in this leaf.
        count: usize,
    },
    /// An internal node with two children.
    Internal {
        /// Bounding box of all primitives in this subtree.
        bounds: Aabb2<F>,
        /// Index of the left child node.
        left: usize,
        /// Index of the right child node.
        right: usize,
    },
}

impl<F: Float> BvhNode<F> {
    /// Returns the bounding box of this node.
    pub fn bounds(&self) -> Aabb2<F> {
        match self {
            BvhNode::Leaf { bounds, .. } => *bounds,
            BvhNode::Internal { bounds, .. } => *bounds,
        }
    }
}

/// A Bounding Volume Hierarchy for efficient spatial queries.
///
/// The BVH stores references to primitives by index, allowing the original
/// data to remain in place while enabling fast spatial queries.
///
/// # Example
///
/// ```
/// use approxum::{Point2, spatial::Bvh};
///
/// let points: Vec<Point2<f64>> = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(5.0, 5.0),
///     Point2::new(6.0, 6.0),
/// ];
///
/// let bvh = Bvh::build(&points, 1);
///
/// // Query for points in a region
/// let query_box = approxum::Aabb2::new(Point2::new(0.0, 0.0), Point2::new(2.0, 2.0));
/// let results = bvh.query_aabb(&points, query_box);
/// assert_eq!(results.len(), 2); // Points at (0,0) and (1,1)
/// ```
#[derive(Debug, Clone)]
pub struct Bvh<F> {
    /// The tree nodes.
    nodes: Vec<BvhNode<F>>,
    /// Indices into the original primitive array, reordered for the BVH.
    indices: Vec<usize>,
    /// Index of the root node.
    root: usize,
}

impl<F: Float> Bvh<F> {
    /// Builds a BVH from a slice of bounded primitives.
    ///
    /// # Arguments
    ///
    /// * `primitives` - The primitives to index
    /// * `max_leaf_size` - Maximum number of primitives per leaf node
    ///
    /// # Returns
    ///
    /// A new BVH, or an empty BVH if primitives is empty.
    pub fn build<T: Bounded<F>>(primitives: &[T], max_leaf_size: usize) -> Self {
        let n = primitives.len();

        if n == 0 {
            return Self {
                nodes: vec![],
                indices: vec![],
                root: 0,
            };
        }

        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(2 * n);

        let root = build_recursive(primitives, &mut indices, 0, n, max_leaf_size, &mut nodes);

        Self {
            nodes,
            indices,
            root,
        }
    }

    /// Returns true if the BVH is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the number of primitives in the BVH.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns the bounding box of the entire BVH.
    pub fn bounds(&self) -> Option<Aabb2<F>> {
        if self.nodes.is_empty() {
            None
        } else {
            Some(self.nodes[self.root].bounds())
        }
    }

    /// Queries for all primitives whose bounds intersect the given AABB.
    ///
    /// Returns indices into the original primitive array.
    pub fn query_aabb<T: Bounded<F>>(&self, primitives: &[T], query: Aabb2<F>) -> Vec<usize> {
        let mut results = Vec::new();
        if !self.nodes.is_empty() {
            self.query_aabb_recursive(primitives, self.root, query, &mut results);
        }
        results
    }

    fn query_aabb_recursive<T: Bounded<F>>(
        &self,
        primitives: &[T],
        node_idx: usize,
        query: Aabb2<F>,
        results: &mut Vec<usize>,
    ) {
        let node = &self.nodes[node_idx];

        if !node.bounds().intersects(query) {
            return;
        }

        match node {
            BvhNode::Leaf { first, count, .. } => {
                for i in *first..(*first + *count) {
                    let prim_idx = self.indices[i];
                    if primitives[prim_idx].bounds().intersects(query) {
                        results.push(prim_idx);
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                self.query_aabb_recursive(primitives, *left, query, results);
                self.query_aabb_recursive(primitives, *right, query, results);
            }
        }
    }

    /// Queries for all primitives whose bounds contain the given point.
    pub fn query_point<T: Bounded<F>>(&self, primitives: &[T], point: Point2<F>) -> Vec<usize> {
        let mut results = Vec::new();
        if !self.nodes.is_empty() {
            self.query_point_recursive(primitives, self.root, point, &mut results);
        }
        results
    }

    fn query_point_recursive<T: Bounded<F>>(
        &self,
        primitives: &[T],
        node_idx: usize,
        point: Point2<F>,
        results: &mut Vec<usize>,
    ) {
        let node = &self.nodes[node_idx];

        if !node.bounds().contains_point(point) {
            return;
        }

        match node {
            BvhNode::Leaf { first, count, .. } => {
                for i in *first..(*first + *count) {
                    let prim_idx = self.indices[i];
                    if primitives[prim_idx].bounds().contains_point(point) {
                        results.push(prim_idx);
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                self.query_point_recursive(primitives, *left, point, results);
                self.query_point_recursive(primitives, *right, point, results);
            }
        }
    }

    /// Finds the nearest primitive to a query point.
    ///
    /// Returns the index of the nearest primitive and the squared distance,
    /// or None if the BVH is empty.
    pub fn nearest<T: Bounded<F>>(&self, primitives: &[T], point: Point2<F>) -> Option<(usize, F)> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_dist_sq = F::infinity();

        self.nearest_recursive(
            primitives,
            self.root,
            point,
            &mut best_idx,
            &mut best_dist_sq,
        );

        if best_dist_sq.is_infinite() {
            None
        } else {
            Some((best_idx, best_dist_sq))
        }
    }

    fn nearest_recursive<T: Bounded<F>>(
        &self,
        primitives: &[T],
        node_idx: usize,
        point: Point2<F>,
        best_idx: &mut usize,
        best_dist_sq: &mut F,
    ) {
        let node = &self.nodes[node_idx];

        // Early exit if this node can't possibly contain a closer point
        let node_dist_sq = node.bounds().distance_squared_to_point(point);
        if node_dist_sq >= *best_dist_sq {
            return;
        }

        match node {
            BvhNode::Leaf { first, count, .. } => {
                for i in *first..(*first + *count) {
                    let prim_idx = self.indices[i];
                    let prim_center = primitives[prim_idx].centroid();
                    let dist_sq = point.distance_squared(prim_center);
                    if dist_sq < *best_dist_sq {
                        *best_dist_sq = dist_sq;
                        *best_idx = prim_idx;
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                // Visit closer child first for better pruning
                let left_dist = self.nodes[*left].bounds().distance_squared_to_point(point);
                let right_dist = self.nodes[*right].bounds().distance_squared_to_point(point);

                if left_dist < right_dist {
                    self.nearest_recursive(primitives, *left, point, best_idx, best_dist_sq);
                    self.nearest_recursive(primitives, *right, point, best_idx, best_dist_sq);
                } else {
                    self.nearest_recursive(primitives, *right, point, best_idx, best_dist_sq);
                    self.nearest_recursive(primitives, *left, point, best_idx, best_dist_sq);
                }
            }
        }
    }

    /// Calls a function for each primitive whose bounds intersect the query AABB.
    ///
    /// The callback receives the primitive index. If it returns `false`, the
    /// traversal stops early.
    pub fn for_each_intersecting<T: Bounded<F>, C>(
        &self,
        primitives: &[T],
        query: Aabb2<F>,
        mut callback: C,
    ) where
        C: FnMut(usize) -> bool,
    {
        if !self.nodes.is_empty() {
            self.for_each_intersecting_recursive(primitives, self.root, query, &mut callback);
        }
    }

    fn for_each_intersecting_recursive<T: Bounded<F>, C>(
        &self,
        primitives: &[T],
        node_idx: usize,
        query: Aabb2<F>,
        callback: &mut C,
    ) -> bool
    where
        C: FnMut(usize) -> bool,
    {
        let node = &self.nodes[node_idx];

        if !node.bounds().intersects(query) {
            return true; // Continue
        }

        match node {
            BvhNode::Leaf { first, count, .. } => {
                for i in *first..(*first + *count) {
                    let prim_idx = self.indices[i];
                    if primitives[prim_idx].bounds().intersects(query) && !callback(prim_idx) {
                        return false; // Stop
                    }
                }
                true
            }
            BvhNode::Internal { left, right, .. } => {
                if !self.for_each_intersecting_recursive(primitives, *left, query, callback) {
                    return false;
                }
                self.for_each_intersecting_recursive(primitives, *right, query, callback)
            }
        }
    }
}

/// Recursively builds the BVH tree.
fn build_recursive<F: Float, T: Bounded<F>>(
    primitives: &[T],
    indices: &mut [usize],
    start: usize,
    end: usize,
    max_leaf_size: usize,
    nodes: &mut Vec<BvhNode<F>>,
) -> usize {
    let count = end - start;

    // Compute bounds for this node
    let bounds = compute_bounds(primitives, &indices[start..end]);

    // Create leaf if small enough
    if count <= max_leaf_size {
        let node_idx = nodes.len();
        nodes.push(BvhNode::Leaf {
            bounds,
            first: start,
            count,
        });
        return node_idx;
    }

    // Find best split using SAH
    let (split_axis, split_pos) = find_best_split(primitives, &indices[start..end], bounds);

    // Partition primitives
    let mid = partition(primitives, &mut indices[start..end], split_axis, split_pos);
    let mid = start + mid;

    // Handle degenerate cases where partition fails
    let mid = if mid == start || mid == end {
        start + count / 2
    } else {
        mid
    };

    // Reserve space for this internal node
    let node_idx = nodes.len();
    nodes.push(BvhNode::Internal {
        bounds,
        left: 0,  // Placeholder
        right: 0, // Placeholder
    });

    // Recursively build children
    let left = build_recursive(primitives, indices, start, mid, max_leaf_size, nodes);
    let right = build_recursive(primitives, indices, mid, end, max_leaf_size, nodes);

    // Update this node with correct child indices
    nodes[node_idx] = BvhNode::Internal {
        bounds,
        left,
        right,
    };

    node_idx
}

/// Computes the bounding box of a set of primitives.
fn compute_bounds<F: Float, T: Bounded<F>>(primitives: &[T], indices: &[usize]) -> Aabb2<F> {
    let first_bounds = primitives[indices[0]].bounds();
    indices[1..].iter().fold(first_bounds, |acc, &idx| {
        acc.union(primitives[idx].bounds())
    })
}

/// Finds the best split using Surface Area Heuristic (SAH).
fn find_best_split<F: Float, T: Bounded<F>>(
    primitives: &[T],
    indices: &[usize],
    bounds: Aabb2<F>,
) -> (usize, F) {
    let size = bounds.size();

    // Choose axis with largest extent
    let axis = if size.x > size.y { 0 } else { 1 };

    // Use median of centroids as split position
    let mut centroids: Vec<F> = indices
        .iter()
        .map(|&idx| {
            let c = primitives[idx].centroid();
            if axis == 0 {
                c.x
            } else {
                c.y
            }
        })
        .collect();

    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = centroids.len() / 2;
    let split_pos = centroids[mid];

    (axis, split_pos)
}

/// Partitions indices around a split position.
/// Returns the number of elements in the left partition.
fn partition<F: Float, T: Bounded<F>>(
    primitives: &[T],
    indices: &mut [usize],
    axis: usize,
    split_pos: F,
) -> usize {
    let mut left = 0;
    let mut right = indices.len();

    while left < right {
        let c = primitives[indices[left]].centroid();
        let pos = if axis == 0 { c.x } else { c.y };

        if pos < split_pos {
            left += 1;
        } else {
            right -= 1;
            indices.swap(left, right);
        }
    }

    left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bvh_empty() {
        let points: Vec<Point2<f64>> = vec![];
        let bvh = Bvh::build(&points, 1);
        assert!(bvh.is_empty());
        assert_eq!(bvh.len(), 0);
        assert!(bvh.bounds().is_none());
    }

    #[test]
    fn test_bvh_single() {
        let points = vec![Point2::new(1.0, 2.0)];
        let bvh = Bvh::build(&points, 1);
        assert!(!bvh.is_empty());
        assert_eq!(bvh.len(), 1);
    }

    #[test]
    fn test_bvh_build() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(5.0, 5.0),
            Point2::new(6.0, 6.0),
        ];

        let bvh = Bvh::build(&points, 1);
        assert_eq!(bvh.len(), 4);

        let bounds = bvh.bounds().unwrap();
        assert_eq!(bounds.min.x, 0.0);
        assert_eq!(bounds.min.y, 0.0);
        assert_eq!(bounds.max.x, 6.0);
        assert_eq!(bounds.max.y, 6.0);
    }

    #[test]
    fn test_bvh_query_aabb() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(5.0, 5.0),
            Point2::new(6.0, 6.0),
        ];

        let bvh = Bvh::build(&points, 1);

        // Query for points in bottom-left region
        let query = Aabb2::new(Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0));
        let results = bvh.query_aabb(&points, query);

        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
    }

    #[test]
    fn test_bvh_query_point() {
        let boxes: Vec<Aabb2<f64>> = vec![
            Aabb2::new(Point2::new(0.0, 0.0), Point2::new(2.0, 2.0)),
            Aabb2::new(Point2::new(1.0, 1.0), Point2::new(3.0, 3.0)),
            Aabb2::new(Point2::new(5.0, 5.0), Point2::new(7.0, 7.0)),
        ];

        let bvh = Bvh::build(&boxes, 1);

        // Point in the overlap of first two boxes
        let results = bvh.query_point(&boxes, Point2::new(1.5, 1.5));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));

        // Point only in third box
        let results = bvh.query_point(&boxes, Point2::new(6.0, 6.0));
        assert_eq!(results.len(), 1);
        assert!(results.contains(&2));

        // Point outside all boxes
        let results = bvh.query_point(&boxes, Point2::new(10.0, 10.0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_bvh_nearest() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];

        let bvh = Bvh::build(&points, 1);

        // Query near first point
        let (idx, dist_sq) = bvh.nearest(&points, Point2::new(0.1, 0.1)).unwrap();
        assert_eq!(idx, 0);
        assert!(dist_sq < 0.1);

        // Query near center - should find one of the corners
        let (idx, _) = bvh.nearest(&points, Point2::new(5.0, 5.0)).unwrap();
        assert!(idx < 4); // Any corner is valid
    }

    #[test]
    fn test_bvh_for_each() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
            Point2::new(5.0, 5.0),
        ];

        let bvh = Bvh::build(&points, 1);

        let query = Aabb2::new(Point2::new(-1.0, -1.0), Point2::new(3.0, 3.0));
        let mut count = 0;
        bvh.for_each_intersecting(&points, query, |_| {
            count += 1;
            true // Continue
        });

        assert_eq!(count, 3);
    }

    #[test]
    fn test_bvh_for_each_early_exit() {
        let points: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
            Point2::new(3.0, 3.0),
        ];

        let bvh = Bvh::build(&points, 1);

        let query = Aabb2::new(Point2::new(-1.0, -1.0), Point2::new(10.0, 10.0));
        let mut count = 0;
        bvh.for_each_intersecting(&points, query, |_| {
            count += 1;
            count < 2 // Stop after 2
        });

        assert_eq!(count, 2);
    }

    #[test]
    fn test_bvh_larger_dataset() {
        // Create a grid of points
        let mut points: Vec<Point2<f64>> = Vec::new();
        for x in 0..10 {
            for y in 0..10 {
                points.push(Point2::new(x as f64, y as f64));
            }
        }

        let bvh = Bvh::build(&points, 4);
        assert_eq!(bvh.len(), 100);

        // Query for points in a region
        let query = Aabb2::new(Point2::new(2.0, 2.0), Point2::new(5.0, 5.0));
        let results = bvh.query_aabb(&points, query);

        // Should find 4x4 = 16 points
        assert_eq!(results.len(), 16);
    }

    #[test]
    fn test_bvh_f32() {
        let points: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
        ];

        let bvh = Bvh::build(&points, 1);
        assert_eq!(bvh.len(), 3);

        let query = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(1.5, 1.5));
        let results = bvh.query_aabb(&points, query);
        assert_eq!(results.len(), 2);
    }
}
