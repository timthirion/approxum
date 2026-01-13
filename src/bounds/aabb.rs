//! Axis-aligned bounding box.

use crate::primitives::{Point2, Vec2};
use num_traits::Float;

/// A 2D axis-aligned bounding box.
///
/// Defined by minimum and maximum corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb2<F> {
    /// Minimum corner (smallest x and y values).
    pub min: Point2<F>,
    /// Maximum corner (largest x and y values).
    pub max: Point2<F>,
}

impl<F: Float> Aabb2<F> {
    /// Creates a new AABB from min and max corners.
    ///
    /// Does not validate that min <= max.
    #[inline]
    pub fn new(min: Point2<F>, max: Point2<F>) -> Self {
        Self { min, max }
    }

    /// Creates an AABB from two arbitrary corners.
    ///
    /// Correctly handles corners in any orientation.
    #[inline]
    pub fn from_corners(a: Point2<F>, b: Point2<F>) -> Self {
        Self {
            min: Point2::new(a.x.min(b.x), a.y.min(b.y)),
            max: Point2::new(a.x.max(b.x), a.y.max(b.y)),
        }
    }

    /// Creates an AABB containing a single point.
    #[inline]
    pub fn from_point(p: Point2<F>) -> Self {
        Self { min: p, max: p }
    }

    /// Creates an AABB from an iterator of points.
    ///
    /// Returns `None` if the iterator is empty.
    pub fn from_points<I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = Point2<F>>,
    {
        let mut iter = points.into_iter();
        let first = iter.next()?;

        let mut aabb = Self::from_point(first);
        for p in iter {
            aabb = aabb.expand_to_include(p);
        }
        Some(aabb)
    }

    /// Returns the width of the AABB.
    #[inline]
    pub fn width(self) -> F {
        self.max.x - self.min.x
    }

    /// Returns the height of the AABB.
    #[inline]
    pub fn height(self) -> F {
        self.max.y - self.min.y
    }

    /// Returns the size as a vector (width, height).
    #[inline]
    pub fn size(self) -> Vec2<F> {
        Vec2::new(self.width(), self.height())
    }

    /// Returns the center point of the AABB.
    #[inline]
    pub fn center(self) -> Point2<F> {
        self.min.midpoint(self.max)
    }

    /// Returns the area of the AABB.
    #[inline]
    pub fn area(self) -> F {
        self.width() * self.height()
    }

    /// Returns a new AABB expanded to include the given point.
    #[inline]
    pub fn expand_to_include(self, p: Point2<F>) -> Self {
        Self {
            min: Point2::new(self.min.x.min(p.x), self.min.y.min(p.y)),
            max: Point2::new(self.max.x.max(p.x), self.max.y.max(p.y)),
        }
    }

    /// Returns the union of two AABBs (smallest AABB containing both).
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: Point2::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y)),
            max: Point2::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y)),
        }
    }

    /// Returns the intersection of two AABBs, if they overlap.
    #[inline]
    pub fn intersection(self, other: Self) -> Option<Self> {
        let min = Point2::new(self.min.x.max(other.min.x), self.min.y.max(other.min.y));
        let max = Point2::new(self.max.x.min(other.max.x), self.max.y.min(other.max.y));

        if min.x <= max.x && min.y <= max.y {
            Some(Self { min, max })
        } else {
            None
        }
    }

    /// Returns `true` if this AABB contains the given point.
    #[inline]
    pub fn contains_point(self, p: Point2<F>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Returns `true` if this AABB intersects another AABB.
    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Returns the squared distance from a point to this AABB.
    ///
    /// Returns 0 if the point is inside the AABB.
    pub fn distance_squared_to_point(self, p: Point2<F>) -> F {
        let dx = if p.x < self.min.x {
            self.min.x - p.x
        } else if p.x > self.max.x {
            p.x - self.max.x
        } else {
            F::zero()
        };

        let dy = if p.y < self.min.y {
            self.min.y - p.y
        } else if p.y > self.max.y {
            p.y - self.max.y
        } else {
            F::zero()
        };

        dx * dx + dy * dy
    }

    /// Returns the distance from a point to this AABB.
    ///
    /// Returns 0 if the point is inside the AABB.
    #[inline]
    pub fn distance_to_point(self, p: Point2<F>) -> F {
        self.distance_squared_to_point(p).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let aabb: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));
        assert_eq!(aabb.min.x, 0.0);
        assert_eq!(aabb.max.x, 10.0);
    }

    #[test]
    fn test_from_corners() {
        // Corners in "wrong" order
        let aabb: Aabb2<f64> = Aabb2::from_corners(Point2::new(10.0, 10.0), Point2::new(0.0, 0.0));
        assert_eq!(aabb.min.x, 0.0);
        assert_eq!(aabb.min.y, 0.0);
        assert_eq!(aabb.max.x, 10.0);
        assert_eq!(aabb.max.y, 10.0);
    }

    #[test]
    fn test_from_points() {
        let points = vec![
            Point2::new(1.0, 2.0),
            Point2::new(-3.0, 5.0),
            Point2::new(4.0, -1.0),
        ];
        let aabb: Aabb2<f64> = Aabb2::from_points(points).unwrap();
        assert_eq!(aabb.min.x, -3.0);
        assert_eq!(aabb.min.y, -1.0);
        assert_eq!(aabb.max.x, 4.0);
        assert_eq!(aabb.max.y, 5.0);
    }

    #[test]
    fn test_from_points_empty() {
        let points: Vec<Point2<f64>> = vec![];
        assert!(Aabb2::from_points(points).is_none());
    }

    #[test]
    fn test_dimensions() {
        let aabb: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 5.0));
        assert_eq!(aabb.width(), 10.0);
        assert_eq!(aabb.height(), 5.0);
        assert_eq!(aabb.area(), 50.0);
    }

    #[test]
    fn test_center() {
        let aabb: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));
        let c = aabb.center();
        assert_eq!(c.x, 5.0);
        assert_eq!(c.y, 5.0);
    }

    #[test]
    fn test_contains_point() {
        let aabb: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));

        assert!(aabb.contains_point(Point2::new(5.0, 5.0)));
        assert!(aabb.contains_point(Point2::new(0.0, 0.0))); // On boundary
        assert!(aabb.contains_point(Point2::new(10.0, 10.0))); // On boundary
        assert!(!aabb.contains_point(Point2::new(-1.0, 5.0)));
        assert!(!aabb.contains_point(Point2::new(5.0, 11.0)));
    }

    #[test]
    fn test_intersects() {
        let a: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));
        let b = Aabb2::new(Point2::new(5.0, 5.0), Point2::new(15.0, 15.0));
        let c = Aabb2::new(Point2::new(20.0, 20.0), Point2::new(30.0, 30.0));

        assert!(a.intersects(b));
        assert!(b.intersects(a));
        assert!(!a.intersects(c));
    }

    #[test]
    fn test_union() {
        let a: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(5.0, 5.0));
        let b = Aabb2::new(Point2::new(3.0, 3.0), Point2::new(10.0, 10.0));
        let u = a.union(b);

        assert_eq!(u.min.x, 0.0);
        assert_eq!(u.min.y, 0.0);
        assert_eq!(u.max.x, 10.0);
        assert_eq!(u.max.y, 10.0);
    }

    #[test]
    fn test_intersection() {
        let a: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));
        let b = Aabb2::new(Point2::new(5.0, 5.0), Point2::new(15.0, 15.0));
        let i = a.intersection(b).unwrap();

        assert_eq!(i.min.x, 5.0);
        assert_eq!(i.min.y, 5.0);
        assert_eq!(i.max.x, 10.0);
        assert_eq!(i.max.y, 10.0);
    }

    #[test]
    fn test_intersection_none() {
        let a: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(5.0, 5.0));
        let b = Aabb2::new(Point2::new(10.0, 10.0), Point2::new(15.0, 15.0));

        assert!(a.intersection(b).is_none());
    }

    #[test]
    fn test_distance_to_point() {
        let aabb: Aabb2<f64> = Aabb2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));

        // Inside
        assert_eq!(aabb.distance_to_point(Point2::new(5.0, 5.0)), 0.0);

        // Outside, aligned with edge
        assert_eq!(aabb.distance_to_point(Point2::new(15.0, 5.0)), 5.0);

        // Outside, diagonal (3-4-5 triangle)
        assert_eq!(aabb.distance_to_point(Point2::new(13.0, 14.0)), 5.0);
    }
}
