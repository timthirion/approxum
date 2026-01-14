//! Curve discretization, fitting, and intersection.

mod arc;
mod bezier;
mod bspline;
mod catmull_rom;
mod fit;
mod hermite;
mod intersect;
mod nurbs;

pub use arc::Arc2;
pub use bezier::{CubicBezier2, QuadraticBezier2};
pub use bspline::BSpline2;
pub use catmull_rom::CatmullRom2;
pub use fit::{fit_cubic, fit_cubic_iterative, FitResult};
pub use hermite::HermiteSpline2;
pub use intersect::{
    cubic_self_intersection, cubics_intersect, intersect_cubic_cubic, intersect_cubic_segment,
    intersect_quadratic_quadratic, intersect_quadratic_segment, quadratics_intersect,
    CurveIntersection,
};
pub use nurbs::Nurbs2;
