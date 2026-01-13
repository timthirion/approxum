//! Core polygon types and basic operations.

use crate::primitives::Point2;
use num_traits::Float;

/// A simple polygon represented as a sequence of vertices.
///
/// Vertices are stored in counter-clockwise order. The polygon is implicitly
/// closed (the last vertex connects to the first).
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon<F> {
    /// The vertices of the polygon in CCW order.
    pub vertices: Vec<Point2<F>>,
}

impl<F: Float> Polygon<F> {
    /// Creates a new polygon from vertices.
    ///
    /// The vertices should be in counter-clockwise order for a positive area.
    /// If provided in clockwise order, area calculations will be negative.
    #[inline]
    pub fn new(vertices: Vec<Point2<F>>) -> Self {
        Self { vertices }
    }

    /// Creates an empty polygon.
    #[inline]
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
        }
    }

    /// Returns true if the polygon has no vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the signed area of the polygon using the shoelace formula.
    ///
    /// Positive for CCW winding, negative for CW winding.
    pub fn signed_area(&self) -> F {
        polygon_signed_area(&self.vertices)
    }

    /// Returns the absolute area of the polygon.
    pub fn area(&self) -> F {
        self.signed_area().abs()
    }

    /// Returns the centroid (center of mass) of the polygon.
    pub fn centroid(&self) -> Option<Point2<F>> {
        polygon_centroid(&self.vertices)
    }

    /// Tests if a point is inside the polygon.
    pub fn contains(&self, point: Point2<F>) -> bool {
        polygon_contains(&self.vertices, point)
    }

    /// Tests if the polygon is convex.
    pub fn is_convex(&self) -> bool {
        polygon_is_convex(&self.vertices)
    }

    /// Returns the bounding box as (min, max) points.
    pub fn bounding_box(&self) -> Option<(Point2<F>, Point2<F>)> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min = self.vertices[0];
        let mut max = self.vertices[0];

        for v in &self.vertices[1..] {
            if v.x < min.x {
                min.x = v.x;
            }
            if v.y < min.y {
                min.y = v.y;
            }
            if v.x > max.x {
                max.x = v.x;
            }
            if v.y > max.y {
                max.y = v.y;
            }
        }

        Some((min, max))
    }

    /// Ensures the polygon has CCW winding order.
    pub fn ensure_ccw(&mut self) {
        if self.signed_area() < F::zero() {
            self.vertices.reverse();
        }
    }

    /// Returns a polygon with reversed winding order.
    pub fn reversed(&self) -> Self {
        let mut vertices = self.vertices.clone();
        vertices.reverse();
        Self { vertices }
    }

    /// Returns the perimeter of the polygon.
    pub fn perimeter(&self) -> F {
        if self.vertices.len() < 2 {
            return F::zero();
        }

        let mut perimeter = F::zero();
        let n = self.vertices.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let dx = self.vertices[j].x - self.vertices[i].x;
            let dy = self.vertices[j].y - self.vertices[i].y;
            perimeter = perimeter + (dx * dx + dy * dy).sqrt();
        }

        perimeter
    }
}

/// Computes the signed area of a polygon using the shoelace formula.
///
/// Positive for CCW winding, negative for CW winding.
pub fn polygon_signed_area<F: Float>(vertices: &[Point2<F>]) -> F {
    if vertices.len() < 3 {
        return F::zero();
    }

    let mut area = F::zero();
    let n = vertices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        area = area + vertices[i].x * vertices[j].y;
        area = area - vertices[j].x * vertices[i].y;
    }

    area / F::from(2.0).unwrap()
}

/// Computes the absolute area of a polygon.
pub fn polygon_area<F: Float>(vertices: &[Point2<F>]) -> F {
    polygon_signed_area(vertices).abs()
}

/// Computes the centroid of a polygon.
///
/// Returns None for degenerate polygons (fewer than 3 vertices or zero area).
pub fn polygon_centroid<F: Float>(vertices: &[Point2<F>]) -> Option<Point2<F>> {
    if vertices.len() < 3 {
        return None;
    }

    let area = polygon_signed_area(vertices);
    if area.abs() < F::epsilon() {
        return None;
    }

    let mut cx = F::zero();
    let mut cy = F::zero();
    let n = vertices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        cx = cx + (vertices[i].x + vertices[j].x) * cross;
        cy = cy + (vertices[i].y + vertices[j].y) * cross;
    }

    let six = F::from(6.0).unwrap();
    Some(Point2::new(cx / (six * area), cy / (six * area)))
}

/// Tests if a point is inside a polygon using the ray casting algorithm.
///
/// Points on the boundary may return either true or false.
pub fn polygon_contains<F: Float>(vertices: &[Point2<F>], point: Point2<F>) -> bool {
    if vertices.len() < 3 {
        return false;
    }

    let mut inside = false;
    let n = vertices.len();

    let mut j = n - 1;
    for i in 0..n {
        let vi = vertices[i];
        let vj = vertices[j];

        if ((vi.y > point.y) != (vj.y > point.y))
            && (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x)
        {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// Tests if a polygon is convex.
///
/// Returns true if all cross products of consecutive edges have the same sign.
pub fn polygon_is_convex<F: Float>(vertices: &[Point2<F>]) -> bool {
    if vertices.len() < 3 {
        return true; // Degenerate cases are considered convex
    }

    let n = vertices.len();
    let mut sign: Option<bool> = None;

    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        let c = vertices[(i + 2) % n];

        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);

        if cross.abs() > F::epsilon() {
            let is_positive = cross > F::zero();
            match sign {
                None => sign = Some(is_positive),
                Some(s) if s != is_positive => return false,
                _ => {}
            }
        }
    }

    true
}

/// Computes the cross product of vectors (b-a) and (c-a).
#[inline]
#[allow(dead_code)] // Kept for potential future use
pub(crate) fn cross<F: Float>(a: Point2<F>, b: Point2<F>, c: Point2<F>) -> F {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Computes the intersection point of two line segments, if any.
///
/// Returns Some((point, t, u)) where t and u are parameters along each segment.
pub(crate) fn segment_intersection<F: Float>(
    p1: Point2<F>,
    p2: Point2<F>,
    p3: Point2<F>,
    p4: Point2<F>,
) -> Option<(Point2<F>, F, F)> {
    let d1x = p2.x - p1.x;
    let d1y = p2.y - p1.y;
    let d2x = p4.x - p3.x;
    let d2y = p4.y - p3.y;

    let denom = d1x * d2y - d1y * d2x;

    if denom.abs() < F::epsilon() {
        return None; // Parallel or collinear
    }

    let t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;
    let u = ((p3.x - p1.x) * d1y - (p3.y - p1.y) * d1x) / denom;

    let point = Point2::new(p1.x + t * d1x, p1.y + t * d1y);

    Some((point, t, u))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_polygon_new() {
        let poly: Polygon<f64> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
        ]);
        assert_eq!(poly.len(), 3);
        assert!(!poly.is_empty());
    }

    #[test]
    fn test_polygon_empty() {
        let poly: Polygon<f64> = Polygon::empty();
        assert!(poly.is_empty());
        assert_eq!(poly.len(), 0);
    }

    #[test]
    fn test_polygon_area_square() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);
        assert!(approx_eq(poly.area(), 4.0, 1e-10));
    }

    #[test]
    fn test_polygon_area_triangle() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(1.0, 2.0),
        ]);
        assert!(approx_eq(poly.area(), 2.0, 1e-10));
    }

    #[test]
    fn test_polygon_signed_area_ccw() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);
        assert!(poly.signed_area() > 0.0); // CCW is positive
    }

    #[test]
    fn test_polygon_signed_area_cw() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 0.0),
        ]);
        assert!(poly.signed_area() < 0.0); // CW is negative
    }

    #[test]
    fn test_polygon_centroid_square() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);
        let centroid = poly.centroid().unwrap();
        assert!(approx_eq(centroid.x, 1.0, 1e-10));
        assert!(approx_eq(centroid.y, 1.0, 1e-10));
    }

    #[test]
    fn test_polygon_contains_inside() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);
        assert!(poly.contains(Point2::new(1.0, 1.0)));
        assert!(poly.contains(Point2::new(0.5, 0.5)));
    }

    #[test]
    fn test_polygon_contains_outside() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);
        assert!(!poly.contains(Point2::new(3.0, 3.0)));
        assert!(!poly.contains(Point2::new(-1.0, 1.0)));
    }

    #[test]
    fn test_polygon_is_convex_square() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);
        assert!(poly.is_convex());
    }

    #[test]
    fn test_polygon_is_convex_concave() {
        // L-shaped polygon (concave)
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);
        assert!(!poly.is_convex());
    }

    #[test]
    fn test_polygon_bounding_box() {
        let poly = Polygon::new(vec![
            Point2::new(1.0_f64, 2.0),
            Point2::new(3.0, 1.0),
            Point2::new(4.0, 3.0),
            Point2::new(2.0, 4.0),
        ]);
        let (min, max) = poly.bounding_box().unwrap();
        assert!(approx_eq(min.x, 1.0, 1e-10));
        assert!(approx_eq(min.y, 1.0, 1e-10));
        assert!(approx_eq(max.x, 4.0, 1e-10));
        assert!(approx_eq(max.y, 4.0, 1e-10));
    }

    #[test]
    fn test_polygon_perimeter() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);
        assert!(approx_eq(poly.perimeter(), 4.0, 1e-10));
    }

    #[test]
    fn test_polygon_ensure_ccw() {
        let mut poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 0.0),
        ]);
        assert!(poly.signed_area() < 0.0); // CW
        poly.ensure_ccw();
        assert!(poly.signed_area() > 0.0); // Now CCW
    }

    #[test]
    fn test_polygon_reversed() {
        let poly = Polygon::new(vec![
            Point2::new(0.0_f64, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
        ]);
        let rev = poly.reversed();
        assert!(poly.signed_area() > 0.0);
        assert!(rev.signed_area() < 0.0);
    }

    #[test]
    fn test_segment_intersection() {
        let p1 = Point2::new(0.0_f64, 0.0);
        let p2 = Point2::new(2.0, 2.0);
        let p3 = Point2::new(0.0, 2.0);
        let p4 = Point2::new(2.0, 0.0);

        let (point, t, u) = segment_intersection(p1, p2, p3, p4).unwrap();
        assert!(approx_eq(point.x, 1.0, 1e-10));
        assert!(approx_eq(point.y, 1.0, 1e-10));
        assert!(approx_eq(t, 0.5, 1e-10));
        assert!(approx_eq(u, 0.5, 1e-10));
    }

    #[test]
    fn test_segment_intersection_parallel() {
        let p1 = Point2::new(0.0_f64, 0.0);
        let p2 = Point2::new(1.0, 0.0);
        let p3 = Point2::new(0.0, 1.0);
        let p4 = Point2::new(1.0, 1.0);

        assert!(segment_intersection(p1, p2, p3, p4).is_none());
    }

    #[test]
    fn test_polygon_f32() {
        let poly: Polygon<f32> = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]);
        assert!((poly.area() - 1.0).abs() < 0.001);
    }
}
