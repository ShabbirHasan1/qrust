//! Definition of probabilities, distributions and utilities.
//! # Examples
//! ```
//! use qrust::proba::*;
//!
//! let n = normal(1.0, 4.5);
//! assert_eq!(n.std(), 4.5);
//! assert_eq!(n.mean(), 1.0);
//!
//! let u = uniform(3.0, 5.0);
//! assert_eq!(u.median(), 4.0);
//! ```
//!
//! The distributions also all come with computation of the both their PDF and CDF over R:
//! ```
//! use std::f64::consts::PI;
//! use qrust::proba::*;
//!
//! let n = normal(1.0, 1.0);
//! assert_eq!(n.cdf(1.0), 0.5);
//! // PDF of the normal variable is exp{-(x-mu)^2/sigma^2} / sqrt{2 * PI * sigma^2}
//! assert!((n.pdf(1.0) - (1.0 / (2. * PI * 1.0).sqrt())).abs() < 1e-12);
//!
//! let x = lognormal(0.0, 1.0);
//! let y = lognormal(0.0, 0.25);
//! assert_eq!(x.cdf(1.0), y.cdf(1.0));
//! assert!(x.cdf(0.5) > y.cdf(0.5));
//! assert!(x.cdf(1.5) < y.cdf(1.5));
//! ```
//!
//! The variables can also be shifted and scaled arbitrarily:
//! ```
//! use qrust::proba::*;
//!
//! let u = uniform(0.0, 1.0);
//! let m = u.mean();
//! let s = u.std();
//! // v := 3 * ((2 * u) - 1.5) + 2.0 -> 6. * u - 2.5
//! let v = u.scale(2.0).shift(-1.5).scale(3.0).shift(2.0);
//! assert_eq!(6. * m - 2.5, v.mean());
//! assert_eq!(6. * s, v.std());
//! ```
use std::cmp::Ordering;
use std::convert::TryFrom;

/// Represents a probability in [0; 1]
/// TODO: make this an integer or something
#[derive(Clone, Copy, Debug)]
pub struct Proba(f64);
impl Proba {
    /// The Half probability
    pub const HALF: Proba = Proba(0.5);
    /// The One (100%) probability
    pub const ONE: Proba = Proba(1.0);
    /// The Zero (0.0%) probability
    pub const ZERO: Proba = Proba(0.0);
    /// The complement to 1 of the probability
    pub fn complement(&self) -> Proba {
        Proba(1.0 - self.0)
    }
    /// Conversion to a float64
    pub fn as_f64(&self) -> f64 {
        self.0
    }
    pub fn from_f64(x: f64) -> Proba {
        Self::try_from(x).expect("Invalid value")
    }
}
impl TryFrom<f64> for Proba {
    type Error = String;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value >= 0.0 && value <= 1.0 {
            Ok(Proba(value))
        } else {
            Err(format!("Invalid probability: {}", value))
        }
    }
}
impl Into<f64> for Proba {
    fn into(self) -> f64 {
        self.as_f64()
    }
}
impl PartialEq<f64> for Proba {
    fn eq(&self, other: &f64) -> bool {
        self.0.eq(other)
    }
}
impl PartialEq<Proba> for Proba {
    fn eq(&self, other: &Proba) -> bool {
        self.eq(&other.0)
    }
}
impl PartialOrd<Proba> for Proba {
    fn partial_cmp(&self, other: &Proba) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

mod marginal;
pub use marginal::*;
