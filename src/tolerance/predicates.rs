//! Geometric predicates with explicit tolerance.

use crate::primitives::{Point2, Segment2};
use num_traits::Float;

/// Result of an orientation test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Points are counter-clockwise (positive area).
    CounterClockwise,
    /// Points are clockwise (negative area).
    Clockwise,
    /// Points are collinear (within tolerance).
    Collinear,
}

/// Computes the orientation of three points with tolerance.
///
/// Returns the orientation of the triangle formed by points `a`, `b`, `c`:
/// - `CounterClockwise` if `c` is to the left of the line from `a` to `b`
/// - `Clockwise` if `c` is to the right of the line from `a` to `b`
/// - `Collinear` if `c` is on the line (within `eps` tolerance)
///
/// The test is based on the signed area of the triangle. If the absolute
/// value of twice the signed area is less than `eps`, the points are
/// considered collinear.
///
/// # Arguments
///
/// * `a`, `b`, `c` - The three points to test
/// * `eps` - Tolerance for collinearity. This is compared against the absolute
///   value of the cross product (twice the signed area).
#[inline]
pub fn orient2d<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>, eps: F) -> Orientation {
    // Cross product of (b - a) and (c - a)
    // This equals twice the signed area of triangle ABC
    let ab = b - a;
    let ac = c - a;
    let cross = ab.cross(ac);

    if cross > eps {
        Orientation::CounterClockwise
    } else if cross < -eps {
        Orientation::Clockwise
    } else {
        Orientation::Collinear
    }
}

/// Checks if a point lies on a line segment within tolerance.
///
/// Returns `true` if the point `p` is within distance `eps` of the segment.
///
/// # Arguments
///
/// * `p` - The point to test
/// * `segment` - The line segment
/// * `eps` - Distance tolerance
#[inline]
pub fn point_on_segment<F: Float>(p: Point2<F>, segment: Segment2<F>, eps: F) -> bool {
    segment.distance_squared_to_point(p) <= eps * eps
}

/// Result of a segment intersection test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentIntersection<F> {
    /// Segments do not intersect.
    None,
    /// Segments intersect at a single point.
    Point {
        /// The intersection point.
        point: Point2<F>,
        /// Parameter along first segment (0 = start, 1 = end).
        t1: F,
        /// Parameter along second segment (0 = start, 1 = end).
        t2: F,
    },
    /// Segments overlap along a line (collinear and overlapping).
    Overlapping {
        /// Start of the overlapping region.
        start: Point2<F>,
        /// End of the overlapping region.
        end: Point2<F>,
    },
}

/// Tests if two line segments intersect, with tolerance.
///
/// Returns information about the intersection:
/// - `None` if segments don't intersect
/// - `Point` if they intersect at a single point (within tolerance)
/// - `Overlapping` if they are collinear and overlap
///
/// # Arguments
///
/// * `s1` - First segment
/// * `s2` - Second segment
/// * `eps` - Tolerance for collinearity and endpoint tests
pub fn segments_intersect<F: Float>(
    s1: Segment2<F>,
    s2: Segment2<F>,
    eps: F,
) -> SegmentIntersection<F> {
    let d1 = s1.direction();
    let d2 = s2.direction();
    let cross = d1.cross(d2);

    // Vector from s1.start to s2.start
    let d = s2.start - s1.start;

    let cross_abs = cross.abs();
    let eps_sq = eps * eps;

    // Check if segments are nearly parallel
    if cross_abs <= eps {
        // Parallel segments - check if collinear
        let dist_sq = s1.distance_squared_to_point(s2.start);
        if dist_sq > eps_sq {
            // Parallel but not collinear
            return SegmentIntersection::None;
        }

        // Collinear - check for overlap
        // Project all points onto the line defined by s1
        let len_sq = d1.magnitude_squared();
        if len_sq <= eps_sq {
            // s1 is degenerate (a point)
            if point_on_segment(s1.start, s2, eps) {
                return SegmentIntersection::Point {
                    point: s1.start,
                    t1: F::zero(),
                    t2: project_point_to_segment_param(s1.start, s2),
                };
            }
            return SegmentIntersection::None;
        }

        // Project s2 endpoints onto s1's line
        let t2_start = (s2.start - s1.start).dot(d1) / len_sq;
        let t2_end = (s2.end - s1.start).dot(d1) / len_sq;

        // Normalize direction of t2 range
        let (t2_min, t2_max) = if t2_start <= t2_end {
            (t2_start, t2_end)
        } else {
            (t2_end, t2_start)
        };

        // Find overlap with [0, 1]
        let overlap_start = t2_min.max(F::zero());
        let overlap_end = t2_max.min(F::one());

        if overlap_start > overlap_end + eps {
            // No overlap
            return SegmentIntersection::None;
        }

        if (overlap_end - overlap_start).abs() <= eps {
            // Single point overlap
            let point = s1.point_at(overlap_start);
            return SegmentIntersection::Point {
                point,
                t1: overlap_start,
                t2: if t2_start <= t2_end {
                    overlap_start
                } else {
                    F::one() - overlap_start
                },
            };
        }

        // Range overlap
        return SegmentIntersection::Overlapping {
            start: s1.point_at(overlap_start),
            end: s1.point_at(overlap_end),
        };
    }

    // Non-parallel segments - find intersection point
    // Solve: s1.start + t1 * d1 = s2.start + t2 * d2
    // Using Cramer's rule:
    // t1 = (d x d2) / (d1 x d2)
    // t2 = (d x d1) / (d1 x d2)
    // where d = s2.start - s1.start

    let t1 = d.cross(d2) / cross;
    let t2 = d.cross(d1) / cross;

    // Check if intersection is within both segments (with tolerance)
    let neg_eps = -eps;
    let one_plus_eps = F::one() + eps;

    if t1 >= neg_eps && t1 <= one_plus_eps && t2 >= neg_eps && t2 <= one_plus_eps {
        // Clamp to [0, 1] for the actual point
        let t1_clamped = t1.max(F::zero()).min(F::one());
        let point = s1.point_at(t1_clamped);

        SegmentIntersection::Point {
            point,
            t1: t1_clamped,
            t2: t2.max(F::zero()).min(F::one()),
        }
    } else {
        SegmentIntersection::None
    }
}

/// Helper: project a point onto a segment and return the parameter.
fn project_point_to_segment_param<F: Float>(p: Point2<F>, seg: Segment2<F>) -> F {
    let (_, t) = seg.closest_point(p);
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // orient2d tests

    #[test]
    fn test_orient2d_ccw() {
        let a: Point2<f64> = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 1.0);
        assert_eq!(orient2d(a, b, c, 1e-10), Orientation::CounterClockwise);
    }

    #[test]
    fn test_orient2d_cw() {
        let a: Point2<f64> = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, -1.0);
        assert_eq!(orient2d(a, b, c, 1e-10), Orientation::Clockwise);
    }

    #[test]
    fn test_orient2d_collinear() {
        let a: Point2<f64> = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(2.0, 0.0);
        assert_eq!(orient2d(a, b, c, 1e-10), Orientation::Collinear);
    }

    #[test]
    fn test_orient2d_nearly_collinear() {
        let a: Point2<f64> = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 1e-12); // Very slightly above the line
        assert_eq!(orient2d(a, b, c, 1e-10), Orientation::Collinear);
    }

    #[test]
    fn test_orient2d_just_above_tolerance() {
        let a: Point2<f64> = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        let c = Point2::new(0.5, 1e-8); // Above tolerance
        assert_eq!(orient2d(a, b, c, 1e-10), Orientation::CounterClockwise);
    }

    // point_on_segment tests

    #[test]
    fn test_point_on_segment_at_start() {
        let seg: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let p = Point2::new(0.0, 0.0);
        assert!(point_on_segment(p, seg, 1e-10));
    }

    #[test]
    fn test_point_on_segment_at_end() {
        let seg: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let p = Point2::new(10.0, 0.0);
        assert!(point_on_segment(p, seg, 1e-10));
    }

    #[test]
    fn test_point_on_segment_middle() {
        let seg: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let p = Point2::new(5.0, 0.0);
        assert!(point_on_segment(p, seg, 1e-10));
    }

    #[test]
    fn test_point_on_segment_near() {
        let seg: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let p = Point2::new(5.0, 0.5);
        assert!(point_on_segment(p, seg, 1.0)); // Within tolerance
        assert!(!point_on_segment(p, seg, 0.1)); // Outside tolerance
    }

    #[test]
    fn test_point_on_segment_beyond_end() {
        let seg: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let p = Point2::new(15.0, 0.0);
        assert!(!point_on_segment(p, seg, 1e-10));
    }

    // segments_intersect tests

    #[test]
    fn test_segments_intersect_crossing() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 10.0);
        let s2 = Segment2::from_coords(0.0, 10.0, 10.0, 0.0);

        match segments_intersect(s1, s2, 1e-10) {
            SegmentIntersection::Point { point, t1, t2 } => {
                assert_relative_eq!(point.x, 5.0, epsilon = 1e-10);
                assert_relative_eq!(point.y, 5.0, epsilon = 1e-10);
                assert_relative_eq!(t1, 0.5, epsilon = 1e-10);
                assert_relative_eq!(t2, 0.5, epsilon = 1e-10);
            }
            _ => panic!("Expected point intersection"),
        }
    }

    #[test]
    fn test_segments_intersect_t_junction() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let s2 = Segment2::from_coords(5.0, -5.0, 5.0, 5.0);

        match segments_intersect(s1, s2, 1e-10) {
            SegmentIntersection::Point { point, t1, t2 } => {
                assert_relative_eq!(point.x, 5.0, epsilon = 1e-10);
                assert_relative_eq!(point.y, 0.0, epsilon = 1e-10);
                assert_relative_eq!(t1, 0.5, epsilon = 1e-10);
                assert_relative_eq!(t2, 0.5, epsilon = 1e-10);
            }
            _ => panic!("Expected point intersection"),
        }
    }

    #[test]
    fn test_segments_intersect_at_endpoint() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 5.0, 5.0);
        let s2 = Segment2::from_coords(5.0, 5.0, 10.0, 0.0);

        match segments_intersect(s1, s2, 1e-10) {
            SegmentIntersection::Point { point, t1, t2 } => {
                assert_relative_eq!(point.x, 5.0, epsilon = 1e-10);
                assert_relative_eq!(point.y, 5.0, epsilon = 1e-10);
                assert_relative_eq!(t1, 1.0, epsilon = 1e-10);
                assert_relative_eq!(t2, 0.0, epsilon = 1e-10);
            }
            _ => panic!("Expected point intersection"),
        }
    }

    #[test]
    fn test_segments_no_intersection() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 1.0, 0.0);
        let s2 = Segment2::from_coords(0.0, 1.0, 1.0, 1.0);

        assert_eq!(segments_intersect(s1, s2, 1e-10), SegmentIntersection::None);
    }

    #[test]
    fn test_segments_parallel_no_intersection() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let s2 = Segment2::from_coords(0.0, 1.0, 10.0, 1.0);

        assert_eq!(segments_intersect(s1, s2, 1e-10), SegmentIntersection::None);
    }

    #[test]
    fn test_segments_collinear_no_overlap() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 5.0, 0.0);
        let s2 = Segment2::from_coords(10.0, 0.0, 15.0, 0.0);

        assert_eq!(segments_intersect(s1, s2, 1e-10), SegmentIntersection::None);
    }

    #[test]
    fn test_segments_collinear_overlapping() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let s2 = Segment2::from_coords(5.0, 0.0, 15.0, 0.0);

        match segments_intersect(s1, s2, 1e-10) {
            SegmentIntersection::Overlapping { start, end } => {
                assert_relative_eq!(start.x, 5.0, epsilon = 1e-10);
                assert_relative_eq!(end.x, 10.0, epsilon = 1e-10);
            }
            _ => panic!("Expected overlapping intersection"),
        }
    }

    #[test]
    fn test_segments_collinear_contained() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 10.0, 0.0);
        let s2 = Segment2::from_coords(2.0, 0.0, 8.0, 0.0);

        match segments_intersect(s1, s2, 1e-10) {
            SegmentIntersection::Overlapping { start, end } => {
                assert_relative_eq!(start.x, 2.0, epsilon = 1e-10);
                assert_relative_eq!(end.x, 8.0, epsilon = 1e-10);
            }
            _ => panic!("Expected overlapping intersection"),
        }
    }

    #[test]
    fn test_segments_collinear_touching() {
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 5.0, 0.0);
        let s2 = Segment2::from_coords(5.0, 0.0, 10.0, 0.0);

        match segments_intersect(s1, s2, 1e-10) {
            SegmentIntersection::Point { point, .. } => {
                assert_relative_eq!(point.x, 5.0, epsilon = 1e-10);
            }
            _ => panic!("Expected point intersection at touching point"),
        }
    }

    #[test]
    fn test_segments_almost_intersecting() {
        // Segments that would intersect if extended, but don't quite touch
        let s1: Segment2<f64> = Segment2::from_coords(0.0, 0.0, 4.0, 4.0);
        let s2 = Segment2::from_coords(6.0, 4.0, 10.0, 0.0);

        assert_eq!(segments_intersect(s1, s2, 1e-10), SegmentIntersection::None);
    }
}
