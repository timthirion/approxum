//! Visibility polygon computation.
//!
//! Computes the region visible from a viewpoint inside a polygon or scene.
//!
//! # Example
//!
//! ```
//! use approxum::polygon::{Polygon, visibility_polygon};
//! use approxum::Point2;
//!
//! // A simple room
//! let room = Polygon::new(vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(10.0, 0.0),
//!     Point2::new(10.0, 10.0),
//!     Point2::new(0.0, 10.0),
//! ]);
//!
//! // Viewpoint in the center
//! let viewpoint = Point2::new(5.0, 5.0);
//!
//! let visible = visibility_polygon(&room, viewpoint);
//! // In an empty room, the entire room is visible
//! ```

use super::core::Polygon;
use crate::primitives::Point2;
use num_traits::Float;
use std::cmp::Ordering;

/// Computes the visibility polygon from a viewpoint inside a simple polygon.
///
/// The visibility polygon contains all points that are visible from the viewpoint,
/// meaning a straight line from the viewpoint to that point doesn't cross any edge.
///
/// # Arguments
///
/// * `boundary` - The bounding polygon (must contain the viewpoint)
/// * `viewpoint` - The point from which visibility is computed
///
/// # Returns
///
/// A polygon representing the visible region. If the viewpoint is outside
/// the boundary or the boundary is invalid, returns an empty polygon.
pub fn visibility_polygon<F: Float>(boundary: &Polygon<F>, viewpoint: Point2<F>) -> Polygon<F> {
    if boundary.vertices.len() < 3 {
        return Polygon::new(vec![]);
    }

    // Check if viewpoint is inside the boundary
    if !super::core::polygon_contains(&boundary.vertices, viewpoint) {
        return Polygon::new(vec![]);
    }

    compute_visibility(&boundary.vertices, &[], viewpoint)
}

/// Computes visibility polygon with obstacles inside the boundary.
///
/// # Arguments
///
/// * `boundary` - The outer bounding polygon
/// * `obstacles` - Interior obstacles that block visibility
/// * `viewpoint` - The point from which visibility is computed
pub fn visibility_polygon_with_obstacles<F: Float>(
    boundary: &Polygon<F>,
    obstacles: &[Polygon<F>],
    viewpoint: Point2<F>,
) -> Polygon<F> {
    if boundary.vertices.len() < 3 {
        return Polygon::new(vec![]);
    }

    if !super::core::polygon_contains(&boundary.vertices, viewpoint) {
        return Polygon::new(vec![]);
    }

    // Check viewpoint is not inside any obstacle
    for obstacle in obstacles {
        if super::core::polygon_contains(&obstacle.vertices, viewpoint) {
            return Polygon::new(vec![]);
        }
    }

    let obstacle_verts: Vec<&[Point2<F>]> =
        obstacles.iter().map(|o| o.vertices.as_slice()).collect();
    compute_visibility(&boundary.vertices, &obstacle_verts, viewpoint)
}

/// Internal visibility computation.
fn compute_visibility<F: Float>(
    boundary: &[Point2<F>],
    obstacles: &[&[Point2<F>]],
    viewpoint: Point2<F>,
) -> Polygon<F> {
    // Collect all edges
    let mut edges: Vec<(Point2<F>, Point2<F>)> = Vec::new();

    // Boundary edges
    let n = boundary.len();
    for i in 0..n {
        edges.push((boundary[i], boundary[(i + 1) % n]));
    }

    // Obstacle edges
    for obstacle in obstacles {
        let m = obstacle.len();
        for i in 0..m {
            edges.push((obstacle[i], obstacle[(i + 1) % m]));
        }
    }

    // Collect all vertices for ray casting
    let mut ray_targets: Vec<Point2<F>> = Vec::new();

    for &v in boundary {
        ray_targets.push(v);
    }

    for obstacle in obstacles {
        for &v in *obstacle {
            ray_targets.push(v);
        }
    }

    // For each vertex, cast rays slightly to each side to catch edges
    let epsilon = F::from(1e-6).unwrap();
    let mut rays: Vec<RayInfo<F>> = Vec::new();

    for target in &ray_targets {
        let dx = target.x - viewpoint.x;
        let dy = target.y - viewpoint.y;
        let angle = dy.atan2(dx);

        // Cast three rays: to the vertex and slightly to each side
        rays.push(RayInfo { angle });
        rays.push(RayInfo {
            angle: angle - epsilon,
        });
        rays.push(RayInfo {
            angle: angle + epsilon,
        });
    }

    // Sort rays by angle
    rays.sort_by(|a, b| a.angle.partial_cmp(&b.angle).unwrap_or(Ordering::Equal));

    // Remove duplicate angles
    rays.dedup_by(|a, b| (a.angle - b.angle).abs() < epsilon * F::from(0.1).unwrap());

    // Cast each ray and find closest intersection
    let mut visibility_points: Vec<Point2<F>> = Vec::new();

    for ray in &rays {
        let dir = Point2::new(ray.angle.cos(), ray.angle.sin());

        if let Some(hit) = cast_ray(viewpoint, dir, &edges) {
            // Avoid duplicate points
            if visibility_points.is_empty()
                || visibility_points
                    .last()
                    .map(|p| p.distance(hit) > epsilon)
                    .unwrap_or(true)
            {
                visibility_points.push(hit);
            }
        }
    }

    // Close the polygon by checking if first and last points are the same
    if visibility_points.len() >= 2 {
        let first = visibility_points[0];
        let last = *visibility_points.last().unwrap();
        if first.distance(last) < epsilon {
            visibility_points.pop();
        }
    }

    Polygon::new(visibility_points)
}

/// Ray information for sorting.
#[derive(Clone, Copy)]
struct RayInfo<F> {
    angle: F,
}

/// Casts a ray and finds the closest intersection with any edge.
fn cast_ray<F: Float>(
    origin: Point2<F>,
    direction: Point2<F>,
    edges: &[(Point2<F>, Point2<F>)],
) -> Option<Point2<F>> {
    let mut closest: Option<(Point2<F>, F)> = None;

    for &(p1, p2) in edges {
        if let Some((point, t)) = ray_segment_intersection(origin, direction, p1, p2) {
            if t > F::epsilon() {
                match closest {
                    None => closest = Some((point, t)),
                    Some((_, closest_t)) if t < closest_t => closest = Some((point, t)),
                    _ => {}
                }
            }
        }
    }

    closest.map(|(p, _)| p)
}

/// Computes ray-segment intersection.
///
/// Ray: origin + t * direction (t >= 0)
/// Segment: p1 to p2
fn ray_segment_intersection<F: Float>(
    origin: Point2<F>,
    direction: Point2<F>,
    p1: Point2<F>,
    p2: Point2<F>,
) -> Option<(Point2<F>, F)> {
    let seg_dir = Point2::new(p2.x - p1.x, p2.y - p1.y);

    // Cross product of directions
    let cross = direction.x * seg_dir.y - direction.y * seg_dir.x;

    if cross.abs() < F::epsilon() {
        // Parallel
        return None;
    }

    let delta = Point2::new(p1.x - origin.x, p1.y - origin.y);

    let t_ray = (delta.x * seg_dir.y - delta.y * seg_dir.x) / cross;
    let t_seg = (delta.x * direction.y - delta.y * direction.x) / cross;

    // Ray must go forward (t >= 0) and intersection must be on segment [0, 1]
    if t_ray >= F::zero() && t_seg >= F::zero() && t_seg <= F::one() {
        let point = Point2::new(
            origin.x + t_ray * direction.x,
            origin.y + t_ray * direction.y,
        );
        Some((point, t_ray))
    } else {
        None
    }
}

/// Computes the visible area from a viewpoint.
///
/// This is a convenience function that returns the area of the visibility polygon.
pub fn visible_area<F: Float>(boundary: &Polygon<F>, viewpoint: Point2<F>) -> F {
    let vis = visibility_polygon(boundary, viewpoint);
    super::core::polygon_area(&vis.vertices).abs()
}

/// Checks if a target point is visible from the viewpoint within the boundary.
///
/// Returns true if a straight line from viewpoint to target doesn't cross any boundary edge
/// (except at endpoints).
pub fn is_visible<F: Float>(
    boundary: &Polygon<F>,
    viewpoint: Point2<F>,
    target: Point2<F>,
) -> bool {
    if boundary.vertices.len() < 3 {
        return false;
    }

    // Both points must be inside or on the boundary
    if !super::core::polygon_contains(&boundary.vertices, viewpoint) {
        return false;
    }
    if !super::core::polygon_contains(&boundary.vertices, target) {
        return false;
    }

    // Check if the line segment from viewpoint to target crosses any edge
    let n = boundary.vertices.len();
    for i in 0..n {
        let p1 = boundary.vertices[i];
        let p2 = boundary.vertices[(i + 1) % n];

        if segments_properly_intersect(viewpoint, target, p1, p2) {
            return false;
        }
    }

    true
}

/// Checks if a target is visible considering obstacles.
pub fn is_visible_with_obstacles<F: Float>(
    boundary: &Polygon<F>,
    obstacles: &[Polygon<F>],
    viewpoint: Point2<F>,
    target: Point2<F>,
) -> bool {
    // First check boundary visibility
    if !is_visible(boundary, viewpoint, target) {
        return false;
    }

    // Check if target is inside any obstacle
    for obstacle in obstacles {
        if super::core::polygon_contains(&obstacle.vertices, target) {
            return false;
        }
    }

    // Check if line of sight crosses any obstacle edge
    for obstacle in obstacles {
        let m = obstacle.vertices.len();
        for i in 0..m {
            let p1 = obstacle.vertices[i];
            let p2 = obstacle.vertices[(i + 1) % m];

            if segments_properly_intersect(viewpoint, target, p1, p2) {
                return false;
            }
        }
    }

    true
}

/// Checks if two segments properly intersect (cross each other, not just touch).
fn segments_properly_intersect<F: Float>(
    a1: Point2<F>,
    a2: Point2<F>,
    b1: Point2<F>,
    b2: Point2<F>,
) -> bool {
    let d1 = cross_product_sign(b1, b2, a1);
    let d2 = cross_product_sign(b1, b2, a2);
    let d3 = cross_product_sign(a1, a2, b1);
    let d4 = cross_product_sign(a1, a2, b2);

    // Proper intersection if signs differ on both checks
    if d1 * d2 < F::zero() && d3 * d4 < F::zero() {
        return true;
    }

    false
}

/// Returns the sign of the cross product (p2-p1) × (p3-p1).
fn cross_product_sign<F: Float>(p1: Point2<F>, p2: Point2<F>, p3: Point2<F>) -> F {
    (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn square<F: Float>(size: F) -> Polygon<F> {
        Polygon::new(vec![
            Point2::new(F::zero(), F::zero()),
            Point2::new(size, F::zero()),
            Point2::new(size, size),
            Point2::new(F::zero(), size),
        ])
    }

    #[test]
    fn test_visibility_empty_room() {
        let room: Polygon<f64> = square(10.0);
        let viewpoint = Point2::new(5.0, 5.0);

        let vis = visibility_polygon(&room, viewpoint);

        // Visibility polygon should cover the entire room
        assert!(!vis.vertices.is_empty());

        // Area should be approximately equal to room area
        let vis_area = super::super::core::polygon_area(&vis.vertices).abs();
        let room_area = super::super::core::polygon_area(&room.vertices).abs();
        assert_relative_eq!(vis_area, room_area, epsilon = 0.1);
    }

    #[test]
    fn test_visibility_corner_viewpoint() {
        let room: Polygon<f64> = square(10.0);
        let viewpoint = Point2::new(1.0, 1.0); // Near corner

        let vis = visibility_polygon(&room, viewpoint);

        assert!(!vis.vertices.is_empty());
        // Should still see entire room from corner
        let vis_area = super::super::core::polygon_area(&vis.vertices).abs();
        assert!(vis_area > 90.0); // Most of the room
    }

    #[test]
    fn test_visibility_outside_boundary() {
        let room: Polygon<f64> = square(10.0);
        let viewpoint = Point2::new(20.0, 20.0); // Outside

        let vis = visibility_polygon(&room, viewpoint);

        // Should return empty polygon
        assert!(vis.vertices.is_empty());
    }

    #[test]
    fn test_visibility_with_obstacle() {
        let room: Polygon<f64> = square(10.0);
        let obstacle = Polygon::new(vec![
            Point2::new(4.0, 4.0),
            Point2::new(6.0, 4.0),
            Point2::new(6.0, 6.0),
            Point2::new(4.0, 6.0),
        ]);
        let viewpoint = Point2::new(2.0, 5.0);

        let vis = visibility_polygon_with_obstacles(&room, &[obstacle], viewpoint);

        // Should have reduced visibility due to obstacle
        assert!(!vis.vertices.is_empty());

        let vis_area = super::super::core::polygon_area(&vis.vertices).abs();
        let room_area = super::super::core::polygon_area(&room.vertices).abs();

        // Visibility should be less than full room
        assert!(vis_area < room_area);
    }

    #[test]
    fn test_visibility_viewpoint_in_obstacle() {
        let room: Polygon<f64> = square(10.0);
        let obstacle = Polygon::new(vec![
            Point2::new(4.0, 4.0),
            Point2::new(6.0, 4.0),
            Point2::new(6.0, 6.0),
            Point2::new(4.0, 6.0),
        ]);
        let viewpoint = Point2::new(5.0, 5.0); // Inside obstacle

        let vis = visibility_polygon_with_obstacles(&room, &[obstacle], viewpoint);

        // Should return empty since viewpoint is in obstacle
        assert!(vis.vertices.is_empty());
    }

    #[test]
    fn test_is_visible_direct_line() {
        let room: Polygon<f64> = square(10.0);

        let viewpoint = Point2::new(2.0, 2.0);
        let target = Point2::new(8.0, 8.0);

        assert!(is_visible(&room, viewpoint, target));
    }

    #[test]
    fn test_is_visible_target_outside() {
        let room: Polygon<f64> = square(10.0);

        let viewpoint = Point2::new(5.0, 5.0);
        let target = Point2::new(20.0, 20.0); // Outside

        assert!(!is_visible(&room, viewpoint, target));
    }

    #[test]
    fn test_is_visible_with_obstacle_blocking() {
        let room: Polygon<f64> = square(10.0);
        let obstacle = Polygon::new(vec![
            Point2::new(4.0, 0.0),
            Point2::new(6.0, 0.0),
            Point2::new(6.0, 10.0),
            Point2::new(4.0, 10.0),
        ]);

        let viewpoint = Point2::new(2.0, 5.0);
        let target = Point2::new(8.0, 5.0); // On other side of wall

        assert!(!is_visible_with_obstacles(
            &room,
            &[obstacle],
            viewpoint,
            target
        ));
    }

    #[test]
    fn test_is_visible_with_obstacle_not_blocking() {
        let room: Polygon<f64> = square(10.0);
        let obstacle = Polygon::new(vec![
            Point2::new(7.0, 7.0),
            Point2::new(9.0, 7.0),
            Point2::new(9.0, 9.0),
            Point2::new(7.0, 9.0),
        ]);

        let viewpoint = Point2::new(2.0, 2.0);
        let target = Point2::new(2.0, 8.0); // Not blocked by obstacle

        assert!(is_visible_with_obstacles(
            &room,
            &[obstacle],
            viewpoint,
            target
        ));
    }

    #[test]
    fn test_visible_area() {
        let room: Polygon<f64> = square(10.0);
        let viewpoint = Point2::new(5.0, 5.0);

        let area = visible_area(&room, viewpoint);

        // Should be approximately 100 (full room)
        assert_relative_eq!(area, 100.0, epsilon = 1.0);
    }

    #[test]
    fn test_visibility_l_shaped_room() {
        // L-shaped room
        let room: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 5.0),
            Point2::new(5.0, 5.0),
            Point2::new(5.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let viewpoint = Point2::new(2.0, 2.0);
        let vis = visibility_polygon(&room, viewpoint);

        // Should have a visibility polygon
        assert!(!vis.vertices.is_empty());

        // Area should be less than if it were a full square
        let vis_area = super::super::core::polygon_area(&vis.vertices).abs();
        assert!(vis_area > 0.0);
    }

    #[test]
    fn test_visibility_triangle() {
        let triangle: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(5.0, 10.0),
        ]);

        let viewpoint = Point2::new(5.0, 3.0);
        let vis = visibility_polygon(&triangle, viewpoint);

        assert!(!vis.vertices.is_empty());

        // Should see entire triangle from inside
        let vis_area = super::super::core::polygon_area(&vis.vertices).abs();
        let tri_area = super::super::core::polygon_area(&triangle.vertices).abs();
        assert_relative_eq!(vis_area, tri_area, epsilon = 1.0);
    }

    #[test]
    fn test_f32_support() {
        let room: Polygon<f32> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ]);

        let viewpoint = Point2::new(5.0, 5.0);
        let vis = visibility_polygon(&room, viewpoint);

        assert!(!vis.vertices.is_empty());
    }

    #[test]
    fn test_multiple_obstacles() {
        let room: Polygon<f64> = square(20.0);

        let obstacles = vec![
            Polygon::new(vec![
                Point2::new(5.0, 5.0),
                Point2::new(7.0, 5.0),
                Point2::new(7.0, 7.0),
                Point2::new(5.0, 7.0),
            ]),
            Polygon::new(vec![
                Point2::new(13.0, 13.0),
                Point2::new(15.0, 13.0),
                Point2::new(15.0, 15.0),
                Point2::new(13.0, 15.0),
            ]),
        ];

        let viewpoint = Point2::new(10.0, 10.0);
        let vis = visibility_polygon_with_obstacles(&room, &obstacles, viewpoint);

        assert!(!vis.vertices.is_empty());

        // Should be less than full room due to obstacles
        let vis_area = super::super::core::polygon_area(&vis.vertices).abs();
        assert!(vis_area < 400.0);
        assert!(vis_area > 300.0); // But still most of the room
    }
}
