//! Tools for iterative methods

/// A simple iterative method
pub trait Iterative {
    /// Performs `n` iterations
    fn iterate_n(&mut self, n: usize) -> ();
    /// Performs a single iteration
    fn iterate(&mut self) -> () {
        self.iterate_n(1);
    }
    fn iter<F>(&mut self, f: F) -> ErrorIterator<Self, F>
    where
        Self: Sized,
    {
        ErrorIterator {
            error: f,
            iterative: self,
        }
    }
}

pub struct ErrorIterator<'a, I, F> {
    error: F,
    iterative: &'a mut I,
}
impl<'a, I, F, T> Iterator for ErrorIterator<'a, I, F>
where
    I: Iterative,
    F: Fn(&I) -> T,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.iterative.iterate();
        Some((&self.error)(&self.iterative))
    }
}
