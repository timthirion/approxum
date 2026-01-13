//! Curve discretization and fitting.

mod arc;
mod bezier;
mod bspline;
mod fit;
mod nurbs;

pub use arc::Arc2;
pub use bezier::{CubicBezier2, QuadraticBezier2};
pub use bspline::BSpline2;
pub use fit::{fit_cubic, fit_cubic_iterative, FitResult};
pub use nurbs::Nurbs2;
