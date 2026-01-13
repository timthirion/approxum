//! Polygon clipping algorithms.

use crate::polygon::core::Polygon;
use crate::primitives::Point2;
use num_traits::Float;

/// Clips a polygon against a convex clipping polygon using Sutherland-Hodgman algorithm.
///
/// This algorithm efficiently clips any polygon against a convex clipping region.
/// The result is a single polygon (possibly empty if entirely clipped).
///
/// # Arguments
///
/// * `subject` - The polygon to be clipped
/// * `clip` - The convex clipping polygon (must be convex)
///
/// # Returns
///
/// The clipped polygon. May be empty if the subject is entirely outside the clip region.
///
/// # Example
///
/// ```
/// use approxum::polygon::{Polygon, sutherland_hodgman};
/// use approxum::Point2;
///
/// // A square to clip
/// let subject = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(3.0, 0.0),
///     Point2::new(3.0, 3.0),
///     Point2::new(0.0, 3.0),
/// ]);
///
/// // Clip against another square
/// let clip = Polygon::new(vec![
///     Point2::new(1.0, 1.0),
///     Point2::new(2.0, 1.0),
///     Point2::new(2.0, 2.0),
///     Point2::new(1.0, 2.0),
/// ]);
///
/// let result = sutherland_hodgman(&subject, &clip);
/// assert!(!result.is_empty());
/// ```
pub fn sutherland_hodgman<F: Float>(subject: &Polygon<F>, clip: &Polygon<F>) -> Polygon<F> {
    if subject.is_empty() || clip.is_empty() {
        return Polygon::empty();
    }

    let mut output = subject.vertices.clone();

    let clip_n = clip.vertices.len();
    for i in 0..clip_n {
        if output.is_empty() {
            break;
        }

        let edge_start = clip.vertices[i];
        let edge_end = clip.vertices[(i + 1) % clip_n];

        let input = output;
        output = Vec::new();

        let input_n = input.len();
        for j in 0..input_n {
            let current = input[j];
            let next = input[(j + 1) % input_n];

            let current_inside = is_inside(current, edge_start, edge_end);
            let next_inside = is_inside(next, edge_start, edge_end);

            if current_inside {
                output.push(current);
                if !next_inside {
                    // Exiting: add intersection
                    if let Some(intersection) = line_intersection(current, next, edge_start, edge_end)
                    {
                        output.push(intersection);
                    }
                }
            } else if next_inside {
                // Entering: add intersection
                if let Some(intersection) = line_intersection(current, next, edge_start, edge_end) {
                    output.push(intersection);
                }
            }
        }
    }

    Polygon::new(output)
}

/// Clips a polygon against a convex clipping polygon.
///
/// This is an alias for `sutherland_hodgman` with a more descriptive name.
pub fn clip_polygon_by_convex<F: Float>(subject: &Polygon<F>, convex_clip: &Polygon<F>) -> Polygon<F> {
    sutherland_hodgman(subject, convex_clip)
}

/// Tests if a point is on the "inside" (left side) of a directed edge.
#[inline]
fn is_inside<F: Float>(point: Point2<F>, edge_start: Point2<F>, edge_end: Point2<F>) -> bool {
    // Cross product: positive means left side (inside for CCW polygon)
    let cross = (edge_end.x - edge_start.x) * (point.y - edge_start.y)
        - (edge_end.y - edge_start.y) * (point.x - edge_start.x);
    cross >= F::zero()
}

/// Computes the intersection of two infinite lines.
fn line_intersection<F: Float>(
    p1: Point2<F>,
    p2: Point2<F>,
    p3: Point2<F>,
    p4: Point2<F>,
) -> Option<Point2<F>> {
    let d1x = p2.x - p1.x;
    let d1y = p2.y - p1.y;
    let d2x = p4.x - p3.x;
    let d2y = p4.y - p3.y;

    let denom = d1x * d2y - d1y * d2x;

    if denom.abs() < F::epsilon() {
        return None; // Parallel
    }

    let t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;

    Some(Point2::new(p1.x + t * d1x, p1.y + t * d1y))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_sutherland_hodgman_inside() {
        // Subject entirely inside clip
        let subject = Polygon::new(vec![
            Point2::new(1.0_f64, 1.0),
            Point2::new(2.0, 1.0),
            Point2::new(2.0, 2.0),
            Point2::new(1.0, 2.0),
        ]);

        let clip = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(3.0, 0.0),
            Point2::new(3.0, 3.0),
            Point2::new(0.0, 3.0),
        ]);

        let result = sutherland_hodgman(&subject, &clip);
        assert_eq!(result.len(), 4);
        assert!(approx_eq(result.area(), 1.0, 1e-10));
    }

    #[test]
    fn test_sutherland_hodgman_outside() {
        // Subject entirely outside clip
        let subject = Polygon::new(vec![
            Point2::new(10.0_f64, 10.0),
            Point2::new(11.0, 10.0),
            Point2::new(11.0, 11.0),
            Point2::new(10.0, 11.0),
        ]);

        let clip = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);

        let result = sutherland_hodgman(&subject, &clip);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sutherland_hodgman_partial_overlap() {
        // Two overlapping squares
        let subject = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let clip = Polygon::new(vec![
            Point2::new(1.0, 1.0),
            Point2::new(3.0, 1.0),
            Point2::new(3.0, 3.0),
            Point2::new(1.0, 3.0),
        ]);

        let result = sutherland_hodgman(&subject, &clip);
        assert!(!result.is_empty());
        // Intersection should be a 1x1 square
        assert!(approx_eq(result.area(), 1.0, 1e-10));
    }

    #[test]
    fn test_sutherland_hodgman_clip_triangle() {
        // Clip a square with a triangle
        let subject = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let clip = Polygon::new(vec![
            Point2::new(1.0, 0.0),
            Point2::new(3.0, 1.0),
            Point2::new(1.0, 2.0),
        ]);

        let result = sutherland_hodgman(&subject, &clip);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_sutherland_hodgman_empty_subject() {
        let subject: Polygon<f64> = Polygon::empty();
        let clip = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
        ]);

        let result = sutherland_hodgman(&subject, &clip);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sutherland_hodgman_empty_clip() {
        let subject = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
        ]);
        let clip: Polygon<f64> = Polygon::empty();

        let result = sutherland_hodgman(&subject, &clip);
        assert!(result.is_empty());
    }

    #[test]
    fn test_is_inside() {
        // Edge from (0,0) to (1,0) - horizontal
        let edge_start = Point2::new(0.0_f64, 0.0);
        let edge_end = Point2::new(1.0, 0.0);

        // Point above (left side for CCW) should be inside
        assert!(is_inside(Point2::new(0.5, 1.0), edge_start, edge_end));
        // Point below (right side) should be outside
        assert!(!is_inside(Point2::new(0.5, -1.0), edge_start, edge_end));
    }

    #[test]
    fn test_clip_polygon_by_convex_alias() {
        let subject = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let clip = Polygon::new(vec![
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(1.0, 2.0),
        ]);

        let result = clip_polygon_by_convex(&subject, &clip);
        assert!(approx_eq(result.area(), 2.0, 1e-10));
    }

    #[test]
    fn test_f32() {
        let subject: Polygon<f32> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);

        let clip: Polygon<f32> = Polygon::new(vec![
            Point2::new(1.0, 1.0),
            Point2::new(3.0, 1.0),
            Point2::new(3.0, 3.0),
            Point2::new(1.0, 3.0),
        ]);

        let result = sutherland_hodgman(&subject, &clip);
        assert!((result.area() - 1.0).abs() < 0.01);
    }
}
