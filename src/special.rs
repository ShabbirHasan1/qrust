//! Special functions

use num::traits::Inv;

/// The `softplus` function x -> ln(1 + exp(x))
///
/// # Arguments
///  * `x` - input in R
///
/// # Returns
/// `ln(1 + exp(x))` - a 'soft' (C infinity) version of `(x)^+`
pub fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// The `logistic` function: x -> 1 / (1 + exp(-x))
///
/// # Arguments
///  * `x` - input in R
///
/// # Returns
/// `1 / (1 + exp(-x))` - output in [0; 1]
pub fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        ((-x).exp() + 1.0).inv()
    } else {
        let ex = x.exp();
        ex / (ex + 1.0)
    }
}
