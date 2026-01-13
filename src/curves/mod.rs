//! Curve discretization and fitting.

mod arc;
mod bezier;
mod bspline;
mod catmull_rom;
mod fit;
mod hermite;
mod nurbs;

pub use arc::Arc2;
pub use bezier::{CubicBezier2, QuadraticBezier2};
pub use bspline::BSpline2;
pub use catmull_rom::CatmullRom2;
pub use fit::{fit_cubic, fit_cubic_iterative, FitResult};
pub use hermite::HermiteSpline2;
pub use nurbs::Nurbs2;
