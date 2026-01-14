//! Path stroke to outline conversion.
//!
//! Converts a path (polyline) into a filled polygon representing the stroke
//! with a given width. Supports various line cap and join styles.
//!
//! # Example
//!
//! ```
//! use approxum::{Point2, polygon::{stroke_polyline, LineCap, LineJoin}};
//!
//! let path = vec![
//!     Point2::new(0.0, 0.0),
//!     Point2::new(10.0, 0.0),
//!     Point2::new(10.0, 10.0),
//! ];
//!
//! let outline = stroke_polyline(&path, 2.0, LineCap::Round, LineJoin::Round, 0.1);
//! ```

use super::core::Polygon;
use crate::primitives::Point2;
use num_traits::Float;

/// Line cap style for path endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineCap {
    /// Flat cap at the exact endpoint.
    Butt,
    /// Semicircular cap extending beyond the endpoint.
    Round,
    /// Square cap extending beyond the endpoint by half the stroke width.
    Square,
}

/// Line join style for path corners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineJoin {
    /// Sharp corner (may be clipped by miter limit).
    Miter,
    /// Rounded corner.
    Round,
    /// Beveled (flat) corner.
    Bevel,
}

/// Options for stroke generation.
#[derive(Debug, Clone, Copy)]
pub struct StrokeOptions<F> {
    /// Stroke width (total width, not half-width).
    pub width: F,
    /// Style for line endpoints.
    pub line_cap: LineCap,
    /// Style for line joins at corners.
    pub line_join: LineJoin,
    /// Miter limit (ratio of miter length to stroke width).
    /// When exceeded, miter joins fall back to bevel.
    pub miter_limit: F,
    /// Tolerance for curve approximation (round caps/joins).
    pub tolerance: F,
}

impl<F: Float> Default for StrokeOptions<F> {
    fn default() -> Self {
        Self {
            width: F::one(),
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: F::from(4.0).unwrap(),
            tolerance: F::from(0.1).unwrap(),
        }
    }
}

impl<F: Float> StrokeOptions<F> {
    /// Creates stroke options with the given width.
    pub fn with_width(width: F) -> Self {
        Self {
            width,
            ..Default::default()
        }
    }

    /// Sets the line cap style.
    pub fn line_cap(mut self, cap: LineCap) -> Self {
        self.line_cap = cap;
        self
    }

    /// Sets the line join style.
    pub fn line_join(mut self, join: LineJoin) -> Self {
        self.line_join = join;
        self
    }

    /// Sets the miter limit.
    pub fn miter_limit(mut self, limit: F) -> Self {
        self.miter_limit = limit;
        self
    }

    /// Sets the tolerance for curve approximation.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Converts an open polyline to a stroked outline polygon.
///
/// # Arguments
///
/// * `points` - The polyline vertices
/// * `width` - Stroke width
/// * `cap` - Line cap style for endpoints
/// * `join` - Line join style for corners
/// * `tolerance` - Tolerance for round caps/joins approximation
///
/// # Returns
///
/// A polygon representing the outline of the stroked path.
///
/// # Example
///
/// ```
/// use approxum::{Point2, polygon::{stroke_polyline, LineCap, LineJoin}};
///
/// let path = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(5.0, 0.0),
///     Point2::new(5.0, 5.0),
/// ];
///
/// let outline = stroke_polyline(&path, 1.0, LineCap::Butt, LineJoin::Miter, 0.1);
/// assert!(!outline.vertices.is_empty());
/// ```
pub fn stroke_polyline<F: Float>(
    points: &[Point2<F>],
    width: F,
    cap: LineCap,
    join: LineJoin,
    tolerance: F,
) -> Polygon<F> {
    let options = StrokeOptions {
        width,
        line_cap: cap,
        line_join: join,
        tolerance,
        ..Default::default()
    };
    stroke_polyline_with_options(points, &options)
}

/// Converts an open polyline to a stroked outline with full options.
pub fn stroke_polyline_with_options<F: Float>(
    points: &[Point2<F>],
    options: &StrokeOptions<F>,
) -> Polygon<F> {
    if points.len() < 2 {
        return Polygon::empty();
    }

    let half_width = options.width / (F::one() + F::one());

    // Build the outline by going forward on one side, then backward on the other
    let mut outline: Vec<Point2<F>> = Vec::new();

    // Forward pass (left side)
    let left_side = build_offset_side(points, half_width, options, false);

    // Backward pass (right side)
    let right_side = build_offset_side(points, -half_width, options, true);

    // Start cap
    add_start_cap(&mut outline, points, half_width, options);

    // Left side (forward)
    outline.extend(left_side);

    // End cap
    add_end_cap(&mut outline, points, half_width, options);

    // Right side (backward, already reversed)
    outline.extend(right_side);

    Polygon::new(outline)
}

/// Converts a closed polyline to a stroked outline polygon.
///
/// For closed paths, no end caps are added and the path is treated as a loop.
///
/// # Example
///
/// ```
/// use approxum::{Point2, polygon::{stroke_closed_polyline, LineJoin}};
///
/// // A triangle
/// let path = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(10.0, 0.0),
///     Point2::new(5.0, 8.0),
/// ];
///
/// let outline = stroke_closed_polyline(&path, 1.0, LineJoin::Miter, 0.1);
/// ```
pub fn stroke_closed_polyline<F: Float>(
    points: &[Point2<F>],
    width: F,
    join: LineJoin,
    tolerance: F,
) -> Polygon<F> {
    let options = StrokeOptions {
        width,
        line_join: join,
        tolerance,
        ..Default::default()
    };
    stroke_closed_polyline_with_options(points, &options)
}

/// Converts a closed polyline to a stroked outline with full options.
pub fn stroke_closed_polyline_with_options<F: Float>(
    points: &[Point2<F>],
    options: &StrokeOptions<F>,
) -> Polygon<F> {
    if points.len() < 3 {
        return Polygon::empty();
    }

    let half_width = options.width / (F::one() + F::one());

    // For closed paths, we generate outer and inner outlines
    let outer = build_closed_offset(points, half_width, options);
    let _inner = build_closed_offset(points, -half_width, options);

    // The result is the outer polygon with a hole (inner polygon reversed)
    // For simplicity, we return just the outer polygon
    // A proper implementation would return a polygon with holes

    Polygon::new(outer)
}

// ============================================================================
// Internal implementation
// ============================================================================

/// Builds one side of the offset path.
fn build_offset_side<F: Float>(
    points: &[Point2<F>],
    offset: F,
    options: &StrokeOptions<F>,
    reverse: bool,
) -> Vec<Point2<F>> {
    let n = points.len();
    if n < 2 {
        return vec![];
    }

    let mut result = Vec::new();

    // Compute normals for each segment
    let mut normals: Vec<Point2<F>> = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = points[i + 1].x - points[i].x;
        let dy = points[i + 1].y - points[i].y;
        let len = (dx * dx + dy * dy).sqrt();
        if len > F::epsilon() {
            // Normal is perpendicular to direction (rotate 90 CCW)
            normals.push(Point2::new(-dy / len, dx / len));
        } else {
            // Degenerate segment, use previous normal or default
            if let Some(&prev) = normals.last() {
                normals.push(prev);
            } else {
                normals.push(Point2::new(F::zero(), F::one()));
            }
        }
    }

    // First point
    let first_offset = Point2::new(
        points[0].x + offset * normals[0].x,
        points[0].y + offset * normals[0].y,
    );
    result.push(first_offset);

    // Interior points with joins
    for i in 1..n - 1 {
        let prev_normal = normals[i - 1];
        let next_normal = normals[i];

        let join_points = compute_join(
            points[i],
            prev_normal,
            next_normal,
            offset,
            options,
        );

        result.extend(join_points);
    }

    // Last point
    let last_offset = Point2::new(
        points[n - 1].x + offset * normals[n - 2].x,
        points[n - 1].y + offset * normals[n - 2].y,
    );
    result.push(last_offset);

    if reverse {
        result.reverse();
    }

    result
}

/// Builds the offset for a closed path.
fn build_closed_offset<F: Float>(
    points: &[Point2<F>],
    offset: F,
    options: &StrokeOptions<F>,
) -> Vec<Point2<F>> {
    let n = points.len();
    if n < 3 {
        return vec![];
    }

    let mut result = Vec::new();

    // Compute normals for each segment (including closing segment)
    let mut normals: Vec<Point2<F>> = Vec::with_capacity(n);
    for i in 0..n {
        let next = (i + 1) % n;
        let dx = points[next].x - points[i].x;
        let dy = points[next].y - points[i].y;
        let len = (dx * dx + dy * dy).sqrt();
        if len > F::epsilon() {
            normals.push(Point2::new(-dy / len, dx / len));
        } else {
            if let Some(&prev) = normals.last() {
                normals.push(prev);
            } else {
                normals.push(Point2::new(F::zero(), F::one()));
            }
        }
    }

    // All points with joins
    for i in 0..n {
        let prev_idx = if i == 0 { n - 1 } else { i - 1 };
        let prev_normal = normals[prev_idx];
        let next_normal = normals[i];

        let join_points = compute_join(
            points[i],
            prev_normal,
            next_normal,
            offset,
            options,
        );

        result.extend(join_points);
    }

    result
}

/// Computes the join points at a corner.
fn compute_join<F: Float>(
    point: Point2<F>,
    prev_normal: Point2<F>,
    next_normal: Point2<F>,
    offset: F,
    options: &StrokeOptions<F>,
) -> Vec<Point2<F>> {
    // Check if normals are nearly parallel
    let cross = prev_normal.x * next_normal.y - prev_normal.y * next_normal.x;
    let _dot = prev_normal.x * next_normal.x + prev_normal.y * next_normal.y;

    // If nearly parallel, just use one offset point
    if cross.abs() < F::from(1e-6).unwrap() {
        let avg_normal = Point2::new(
            (prev_normal.x + next_normal.x) / (F::one() + F::one()),
            (prev_normal.y + next_normal.y) / (F::one() + F::one()),
        );
        let len = (avg_normal.x * avg_normal.x + avg_normal.y * avg_normal.y).sqrt();
        let normalized = if len > F::epsilon() {
            Point2::new(avg_normal.x / len, avg_normal.y / len)
        } else {
            prev_normal
        };
        return vec![Point2::new(
            point.x + offset * normalized.x,
            point.y + offset * normalized.y,
        )];
    }

    // Determine if this is an inner or outer corner for this offset side
    let is_outer = (cross * offset) > F::zero();

    match options.line_join {
        LineJoin::Miter => {
            compute_miter_join(point, prev_normal, next_normal, offset, options, is_outer)
        }
        LineJoin::Round => {
            compute_round_join(point, prev_normal, next_normal, offset, options, is_outer)
        }
        LineJoin::Bevel => {
            compute_bevel_join(point, prev_normal, next_normal, offset, is_outer)
        }
    }
}

/// Computes a miter join.
fn compute_miter_join<F: Float>(
    point: Point2<F>,
    prev_normal: Point2<F>,
    next_normal: Point2<F>,
    offset: F,
    options: &StrokeOptions<F>,
    is_outer: bool,
) -> Vec<Point2<F>> {
    if !is_outer {
        // Inner corner: use the intersection point
        if let Some(miter) = compute_miter_point(point, prev_normal, next_normal, offset) {
            return vec![miter];
        }
    }

    // Compute miter point
    let miter_point = compute_miter_point(point, prev_normal, next_normal, offset);

    if let Some(miter) = miter_point {
        // Check miter limit
        let miter_length = miter.distance(point);
        let half_width = offset.abs();

        if half_width > F::epsilon() && miter_length / half_width > options.miter_limit {
            // Exceeded miter limit, fall back to bevel
            return compute_bevel_join(point, prev_normal, next_normal, offset, is_outer);
        }

        vec![miter]
    } else {
        // Fallback to bevel
        compute_bevel_join(point, prev_normal, next_normal, offset, is_outer)
    }
}

/// Computes the miter intersection point.
fn compute_miter_point<F: Float>(
    point: Point2<F>,
    prev_normal: Point2<F>,
    next_normal: Point2<F>,
    offset: F,
) -> Option<Point2<F>> {
    // The miter point is where the two offset lines intersect
    // Line 1: point + offset * prev_normal + t * prev_tangent
    // Line 2: point + offset * next_normal + s * next_tangent

    // Tangent is perpendicular to normal
    let prev_tangent = Point2::new(prev_normal.y, -prev_normal.x);
    let next_tangent = Point2::new(next_normal.y, -next_normal.x);

    // Solve for intersection
    let cross = prev_tangent.x * next_tangent.y - prev_tangent.y * next_tangent.x;

    if cross.abs() < F::epsilon() {
        return None;
    }

    let p1 = Point2::new(
        point.x + offset * prev_normal.x,
        point.y + offset * prev_normal.y,
    );
    let p2 = Point2::new(
        point.x + offset * next_normal.x,
        point.y + offset * next_normal.y,
    );

    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;

    let t = (dx * next_tangent.y - dy * next_tangent.x) / cross;

    Some(Point2::new(
        p1.x + t * prev_tangent.x,
        p1.y + t * prev_tangent.y,
    ))
}

/// Computes a bevel join.
fn compute_bevel_join<F: Float>(
    point: Point2<F>,
    prev_normal: Point2<F>,
    next_normal: Point2<F>,
    offset: F,
    is_outer: bool,
) -> Vec<Point2<F>> {
    let p1 = Point2::new(
        point.x + offset * prev_normal.x,
        point.y + offset * prev_normal.y,
    );
    let p2 = Point2::new(
        point.x + offset * next_normal.x,
        point.y + offset * next_normal.y,
    );

    if is_outer {
        vec![p1, p2]
    } else {
        // Inner corner: just use one point (average or intersection)
        if let Some(miter) = compute_miter_point(point, prev_normal, next_normal, offset) {
            vec![miter]
        } else {
            vec![Point2::new(
                (p1.x + p2.x) / (F::one() + F::one()),
                (p1.y + p2.y) / (F::one() + F::one()),
            )]
        }
    }
}

/// Computes a round join.
fn compute_round_join<F: Float>(
    point: Point2<F>,
    prev_normal: Point2<F>,
    next_normal: Point2<F>,
    offset: F,
    options: &StrokeOptions<F>,
    is_outer: bool,
) -> Vec<Point2<F>> {
    if !is_outer {
        // Inner corner: use miter point
        if let Some(miter) = compute_miter_point(point, prev_normal, next_normal, offset) {
            return vec![miter];
        }
    }

    // Generate arc from prev_normal to next_normal
    let start_angle = prev_normal.y.atan2(prev_normal.x);
    let end_angle = next_normal.y.atan2(next_normal.x);

    let radius = offset.abs();
    let mut arc_points = generate_arc(point, radius, start_angle, end_angle, options.tolerance);

    // Apply offset sign
    if offset < F::zero() {
        arc_points.reverse();
    }

    arc_points
}

/// Generates arc points between two angles.
fn generate_arc<F: Float>(
    center: Point2<F>,
    radius: F,
    start_angle: F,
    end_angle: F,
    tolerance: F,
) -> Vec<Point2<F>> {
    let mut result = Vec::new();

    // Determine angle sweep
    let pi = F::from(std::f64::consts::PI).unwrap();
    let two_pi = pi + pi;

    let mut sweep = end_angle - start_angle;

    // Normalize sweep to [-π, π]
    while sweep > pi {
        sweep = sweep - two_pi;
    }
    while sweep < -pi {
        sweep = sweep + two_pi;
    }

    // Number of segments based on tolerance
    let _arc_length = radius * sweep.abs();
    let num_segments = if tolerance > F::epsilon() {
        let step_angle = (F::one() + F::one()) * (F::one() - tolerance / radius).acos();
        ((sweep.abs() / step_angle).ceil()).max(F::one())
    } else {
        F::from(8.0).unwrap()
    };

    let n = num_segments.to_usize().unwrap_or(8).max(1);
    let angle_step = sweep / F::from(n).unwrap();

    for i in 0..=n {
        let angle = start_angle + F::from(i).unwrap() * angle_step;
        result.push(Point2::new(
            center.x + radius * angle.cos(),
            center.y + radius * angle.sin(),
        ));
    }

    result
}

/// Adds the start cap to the outline.
fn add_start_cap<F: Float>(
    outline: &mut Vec<Point2<F>>,
    points: &[Point2<F>],
    half_width: F,
    options: &StrokeOptions<F>,
) {
    if points.len() < 2 {
        return;
    }

    let p0 = points[0];
    let p1 = points[1];

    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let len = (dx * dx + dy * dy).sqrt();

    if len < F::epsilon() {
        return;
    }

    // Direction from start to next point
    let dir = Point2::new(dx / len, dy / len);
    // Normal (perpendicular, pointing left)
    let normal = Point2::new(-dir.y, dir.x);

    match options.line_cap {
        LineCap::Butt => {
            // No cap needed, offset points are sufficient
        }
        LineCap::Square => {
            // Extend backward by half_width
            let back = Point2::new(p0.x - half_width * dir.x, p0.y - half_width * dir.y);
            let corner1 = Point2::new(
                back.x - half_width * normal.x,
                back.y - half_width * normal.y,
            );
            let corner2 = Point2::new(
                back.x + half_width * normal.x,
                back.y + half_width * normal.y,
            );
            outline.push(corner1);
            outline.push(corner2);
        }
        LineCap::Round => {
            // Semicircle from right side to left side
            let start_angle = (-normal.y).atan2(-normal.x);
            let end_angle = normal.y.atan2(normal.x);

            let arc = generate_arc(p0, half_width, start_angle, end_angle, options.tolerance);
            outline.extend(arc);
        }
    }
}

/// Adds the end cap to the outline.
fn add_end_cap<F: Float>(
    outline: &mut Vec<Point2<F>>,
    points: &[Point2<F>],
    half_width: F,
    options: &StrokeOptions<F>,
) {
    let n = points.len();
    if n < 2 {
        return;
    }

    let p0 = points[n - 2];
    let p1 = points[n - 1];

    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let len = (dx * dx + dy * dy).sqrt();

    if len < F::epsilon() {
        return;
    }

    let dir = Point2::new(dx / len, dy / len);
    let normal = Point2::new(-dir.y, dir.x);

    match options.line_cap {
        LineCap::Butt => {
            // No cap needed
        }
        LineCap::Square => {
            let forward = Point2::new(p1.x + half_width * dir.x, p1.y + half_width * dir.y);
            let corner1 = Point2::new(
                forward.x + half_width * normal.x,
                forward.y + half_width * normal.y,
            );
            let corner2 = Point2::new(
                forward.x - half_width * normal.x,
                forward.y - half_width * normal.y,
            );
            outline.push(corner1);
            outline.push(corner2);
        }
        LineCap::Round => {
            let start_angle = normal.y.atan2(normal.x);
            let end_angle = (-normal.y).atan2(-normal.x);

            let arc = generate_arc(p1, half_width, start_angle, end_angle, options.tolerance);
            outline.extend(arc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn horizontal_line() -> Vec<Point2<f64>> {
        vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)]
    }

    fn right_angle() -> Vec<Point2<f64>> {
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
        ]
    }

    fn triangle() -> Vec<Point2<f64>> {
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(5.0, 8.0),
        ]
    }

    #[test]
    fn test_stroke_horizontal_line_butt() {
        let path = horizontal_line();
        let outline = stroke_polyline(&path, 2.0, LineCap::Butt, LineJoin::Miter, 0.1);

        assert!(!outline.vertices.is_empty());
        assert!(outline.vertices.len() >= 4);

        // Should be a rectangle
        let area = outline.area();
        assert_relative_eq!(area, 20.0, epsilon = 0.5); // 10 x 2
    }

    #[test]
    fn test_stroke_horizontal_line_square() {
        let path = horizontal_line();
        let outline = stroke_polyline(&path, 2.0, LineCap::Square, LineJoin::Miter, 0.1);

        assert!(!outline.vertices.is_empty());

        // Should extend by 1 on each side (half width)
        let area = outline.area();
        assert_relative_eq!(area, 24.0, epsilon = 0.5); // 12 x 2
    }

    #[test]
    fn test_stroke_horizontal_line_round() {
        let path = horizontal_line();
        let outline = stroke_polyline(&path, 2.0, LineCap::Round, LineJoin::Miter, 0.1);

        assert!(!outline.vertices.is_empty());

        // Area should be at least as big as butt cap rectangle
        let area = outline.area();
        assert!(area >= 19.0); // At least the rectangle area
    }

    #[test]
    fn test_stroke_right_angle_miter() {
        let path = right_angle();
        let outline = stroke_polyline(&path, 2.0, LineCap::Butt, LineJoin::Miter, 0.1);

        assert!(!outline.vertices.is_empty());
        assert!(outline.area() > 0.0);
    }

    #[test]
    fn test_stroke_right_angle_bevel() {
        let path = right_angle();
        let outline = stroke_polyline(&path, 2.0, LineCap::Butt, LineJoin::Bevel, 0.1);

        assert!(!outline.vertices.is_empty());
        assert!(outline.area() > 0.0);
    }

    #[test]
    fn test_stroke_right_angle_round() {
        let path = right_angle();
        let outline = stroke_polyline(&path, 2.0, LineCap::Butt, LineJoin::Round, 0.1);

        assert!(!outline.vertices.is_empty());
        assert!(outline.area() > 0.0);
    }

    #[test]
    fn test_stroke_closed_triangle() {
        let path = triangle();
        let outline = stroke_closed_polyline(&path, 2.0, LineJoin::Miter, 0.1);

        assert!(!outline.vertices.is_empty());
        assert!(outline.area() > 0.0);
    }

    #[test]
    fn test_stroke_single_point() {
        let path = vec![Point2::new(0.0, 0.0)];
        let outline = stroke_polyline(&path, 2.0, LineCap::Round, LineJoin::Miter, 0.1);

        assert!(outline.vertices.is_empty());
    }

    #[test]
    fn test_stroke_two_points() {
        let path = vec![Point2::new(0.0, 0.0), Point2::new(5.0, 0.0)];
        let outline = stroke_polyline(&path, 2.0, LineCap::Butt, LineJoin::Miter, 0.1);

        assert!(!outline.vertices.is_empty());
        assert!(outline.vertices.len() >= 4);
    }

    #[test]
    fn test_stroke_options_builder() {
        let options = StrokeOptions::with_width(3.0)
            .line_cap(LineCap::Round)
            .line_join(LineJoin::Bevel)
            .miter_limit(2.0)
            .tolerance(0.05);

        assert_eq!(options.width, 3.0);
        assert_eq!(options.line_cap, LineCap::Round);
        assert_eq!(options.line_join, LineJoin::Bevel);
        assert_eq!(options.miter_limit, 2.0);
        assert_eq!(options.tolerance, 0.05);
    }

    #[test]
    fn test_stroke_with_options() {
        let path = right_angle();
        let options = StrokeOptions::with_width(2.0)
            .line_cap(LineCap::Round)
            .line_join(LineJoin::Round);

        let outline = stroke_polyline_with_options(&path, &options);
        assert!(!outline.vertices.is_empty());
    }

    #[test]
    fn test_stroke_zigzag() {
        let path = vec![
            Point2::new(0.0, 0.0),
            Point2::new(5.0, 5.0),
            Point2::new(10.0, 0.0),
            Point2::new(15.0, 5.0),
        ];

        let outline = stroke_polyline(&path, 1.0, LineCap::Butt, LineJoin::Miter, 0.1);
        assert!(!outline.vertices.is_empty());
        assert!(outline.area() > 0.0);
    }

    #[test]
    fn test_stroke_acute_angle_miter_limit() {
        // Very acute angle that should trigger miter limit
        let path = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(0.0, 1.0), // Sharp turn
        ];

        let options = StrokeOptions {
            width: 2.0,
            miter_limit: 2.0, // Low miter limit
            ..Default::default()
        };

        let outline = stroke_polyline_with_options(&path, &options);
        assert!(!outline.vertices.is_empty());
    }

    #[test]
    fn test_f32_support() {
        let path: Vec<Point2<f32>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
        ];

        let outline = stroke_polyline(&path, 1.0, LineCap::Butt, LineJoin::Miter, 0.1);
        assert!(!outline.vertices.is_empty());
    }

    #[test]
    fn test_stroke_closed_square() {
        let path = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];

        let outline = stroke_closed_polyline(&path, 2.0, LineJoin::Miter, 0.1);
        assert!(!outline.vertices.is_empty());

        // Should have a reasonable area (outer square expanded by stroke width)
        assert!(outline.area() > 50.0);
    }

    #[test]
    fn test_stroke_degenerate_segment() {
        // Path with a zero-length segment
        let path = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 0.0), // Degenerate
            Point2::new(10.0, 0.0),
        ];

        let outline = stroke_polyline(&path, 2.0, LineCap::Butt, LineJoin::Miter, 0.1);
        assert!(!outline.vertices.is_empty());
    }

    #[test]
    fn test_line_cap_enum() {
        assert_ne!(LineCap::Butt, LineCap::Round);
        assert_ne!(LineCap::Round, LineCap::Square);
        assert_eq!(LineCap::Butt, LineCap::Butt);
    }

    #[test]
    fn test_line_join_enum() {
        assert_ne!(LineJoin::Miter, LineJoin::Round);
        assert_ne!(LineJoin::Round, LineJoin::Bevel);
        assert_eq!(LineJoin::Miter, LineJoin::Miter);
    }
}
