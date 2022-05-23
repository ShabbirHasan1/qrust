//! Quantitative Rust tools
//! =======================
//!
//! This library is a collection of miscellaneous numerical/quantitative tools for Rust

pub mod misc;
pub mod pricing;
pub mod proba;

/// Internal utilities for testing
#[cfg(test)]
mod testing {
    pub use approx::AbsDiffEq;
    use ndarray::Array2;
    use num::traits::Pow;

    /// Rounds the provided number at `digits` number after the decimal point
    ///
    /// The operation done is: (x * 10^digits).round() / 10^digits
    ///
    /// Note that this is a very crude way to do this, and can lose precision on top
    /// of the desired rounding. Its main intent is for reporting and printing.
    ///
    /// # Arguments
    ///  * x - the value to round
    ///  * digits - the number of digits after the decimal point to keep
    ///
    /// # Returns
    /// rounded - a value rounded at the provided points.
    pub(crate) fn round_at(x: f64, digits: u8) -> f64 {
        let pow = 10.0f64.pow(digits);
        (x * pow).round() / pow
    }

    pub(crate) fn assert_identity(mat: &Array2<f64>, tol: f64) -> () {
        let n = mat.shape()[0];
        assert!(
            mat.abs_diff_eq(&Array2::eye(n), tol),
            "{} is not identity",
            mat
        );
    }
    pub(crate) fn assert_inverse(x: &Array2<f64>, y: &Array2<f64>, tol: f64) -> () {
        assert_identity(&x.dot(y), tol);
        assert_identity(&y.dot(x), tol);
    }
    pub(crate) fn assert_allclose<T>(x: &T, y: &T, tol: T::Epsilon) -> ()
        where
            T: AbsDiffEq + std::fmt::Debug,
            T::Epsilon: Copy + std::fmt::Display,
    {
        assert!(
            x.abs_diff_eq(y, tol),
            "{:?} != {:?} with tolerance {}",
            x,
            y,
            tol
        );
    }
}
