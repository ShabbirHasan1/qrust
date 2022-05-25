//! Special functions

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

