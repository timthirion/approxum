//! Error types for approxum operations.

use thiserror::Error;

/// Errors that can occur during approximate geometric operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ApproxError {
    /// Points are too close together for reliable computation.
    #[error("degenerate input: points too close together")]
    DegenerateInput,

    /// Lines are nearly parallel.
    #[error("lines are nearly parallel")]
    NearParallel,

    /// Tolerance is too small for the input scale.
    #[error("tolerance too small for input scale")]
    ToleranceTooSmall,

    /// Algorithm did not converge within the iteration limit.
    #[error("convergence failed after {iterations} iterations")]
    ConvergenceFailed {
        /// Number of iterations attempted.
        iterations: usize,
    },
}
