//! Curve discretization and fitting.

mod arc;
mod bezier;
mod fit;

pub use arc::Arc2;
pub use bezier::{CubicBezier2, QuadraticBezier2};
pub use fit::{fit_cubic, fit_cubic_iterative, FitResult};
