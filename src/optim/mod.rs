//! Optimisation and root-finding routines
/// The errors which can be raised when doing numerical optimisation
#[derive(Clone, Debug)]
pub enum Error {
    /// A numerical error (typically, division by zero) prevents the algorithm to run,
    /// along with its details
    NumericalError(String),
    /// The maximum number of iterations has been reached
    MaxIterations,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::NumericalError(s) => write!(f, "Numerical error: {}", s),
            Error::MaxIterations => write!(f, "Maximum number of iterations reached"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

pub mod oned;
pub use oned::{newton_raphson, halley, RootFinding1d};