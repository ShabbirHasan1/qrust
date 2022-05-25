//! Optimisation and root-finding routines
pub mod oned;

pub use oned::{newton_raphson, halley, RootFinding1d};