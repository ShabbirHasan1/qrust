//! Tools for iterative methods

/// A simple iterative method
pub trait Method {
    /// Performs a single iteration
    fn iterate(&mut self) -> ();
}

pub trait Iterative {}
