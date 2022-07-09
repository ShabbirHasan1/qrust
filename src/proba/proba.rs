//! Numerical representation of probability(ies)

use crate::special::logistic;
use std::cmp::Ordering;

/// Represents a probability in [0; 1]
/// A probability `p` is represented by a float `x` such that `logistic(x) = p`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Proba(f64);

impl Proba {
    /// The Half probability
    pub const HALF: Proba = Proba(0.0);
    /// The One (100%) probability
    pub const ONE: Proba = Proba(f64::INFINITY);
    /// The Zero (0.0%) probability
    pub const ZERO: Proba = Proba(f64::NEG_INFINITY);
    /// The complement to 1 of the probability
    pub fn complement(&self) -> Proba {
        Proba(-self.0)
    }
    /// Represents the probability as a floating-point precision number
    pub fn as_f64(&self) -> f64 {
        logistic(self.0)
    }
    /// Whether the probability is exactly one
    pub fn is_one(&self) -> bool {
        self.0.is_sign_positive() && self.0.is_infinite()
    }
    /// Whether the probability is exactly zero
    pub fn is_zero(&self) -> bool {
        self.0.is_sign_negative() && self.0.is_infinite()
    }
    /// Converts from a f64 value, clipping if necessary
    pub fn from_f64(p: f64) -> Self {
        if p <= 0.0 {
            Proba::ZERO
        } else if p >= 1.0 {
            Proba::ONE
        } else {
            Proba((p / (1. - p)).ln())
        }
    }
    /// Converts from a f64 as the natural logarithm of the proba (has to be negative)
    pub fn from_ln(f: f64) -> Self {
        if f >= 0.0 {
            Proba::ONE
        } else {
            // TODO: there has to be a better way...
            Self::from_f64(f.exp())
        }
    }
    /// Attempts to convert the floating point representation of the probability,
    /// returning the faulty value if it fails to do so
    pub fn try_from_f64(p: f64) -> Result<Self, f64> {
        if p < 0.0 || p > 1.0 {
            Err(p)
        } else {
            Ok(Self::from_f64(p))
        }
    }
}
/// Implements conversion from a floating point into a probability by CLIPPING
impl From<f64> for Proba {
    fn from(p: f64) -> Self {
        Proba::from_f64(p)
    }
}
impl Into<f64> for Proba {
    fn into(self) -> f64 {
        self.as_f64()
    }
}
impl PartialOrd<Proba> for Proba {
    fn partial_cmp(&self, other: &Proba) -> Option<Ordering> {
        // Logistic representation is strictly increasing
        self.0.partial_cmp(&other.0)
    }
}
impl PartialEq<f64> for Proba {
    fn eq(&self, other: &f64) -> bool {
        self.as_f64().eq(other)
    }
}
impl PartialOrd<f64> for Proba {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.as_f64().partial_cmp(other)
    }
}

/// A partition of the [0; 1] space representing a discrete distribution of probabilities.
/// The weights of the partition, proportional to the log of the probabilities
#[derive(Clone)]
pub struct Partition(Vec<f64>);
impl Partition {
    /// Generates a new uniform partition of [0; 1] with `n` different buckets
    pub fn uniform(n: usize) -> Self {
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(0.0);
        }
        Self(data).normalize()
    }
    pub fn from_weights(weights: Vec<f64>) -> Self {
        Self(weights.into_iter().map(|x| x.max(0.0).ln()).collect()).normalize()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn proba(&self, i: usize) -> Option<Proba> {
        if i < self.len() {
            Some(Proba::from_ln(self.0[i]))
        } else {
            None
        }
    }
    /// Returns an array of probabilities
    pub fn probas(&self) -> Vec<Proba> {
        let n = self.len();
        let mut probas = Vec::with_capacity(n);
        let mut diff = 1.0;
        let mut imin = 0;
        let mut mn = 1.0;
        for (i, ln_weight) in self.0.iter().enumerate() {
            let proba = ln_weight.exp();
            if proba < mn {
                mn = proba;
                imin = i;
            }
            probas.push(proba);
            diff -= proba;
        }
        probas[imin] += diff;
        probas.into_iter().map(|p| Proba::from_f64(p)).collect()
    }
    /// Translates the weights until it gets to be a proper partition
    fn normalize(mut self) -> Self {
        let n = self.len();
        let mut mx = self.0[0];
        for i in 1..n {
            if self.0[i] > mx {
                mx = self.0[i];
            }
        }
        let cte = -mx - self.0.iter().map(|wi| (*wi - mx).exp()).sum::<f64>().ln();
        for i in 0..n {
            self.0[i] += cte;
        }
        self
    }
    /// Performs a bayesian update of the partition, using the provided conditional probabilities
    pub fn bayes(mut self, conditional: Partition) -> Result<Self, (usize, usize)> {
        let n = self.len();
        if n != conditional.len() {
            return Err((n, conditional.len()));
        }
        for i in 0..n {
            self.0[i] += conditional.0[i];
        }
        Ok(self.normalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_allclose;
    #[test]
    pub fn test_uniform() {
        let part = Partition::uniform(10);
        let total = part.probas().into_iter().map(|p| -> f64 { p.into() }).sum();
        assert_allclose(&total, &1.0, 1e-100);
        assert_allclose(
            &part.proba(0).unwrap().as_f64(),
            &(part.proba(1).unwrap().as_f64()),
            1e-15,
        );
    }
    #[test]
    pub fn test_from_weights() {
        let part = Partition::from_weights(vec![10.0, 5.0, 2.0, 10.0]);
        let total = part.probas().into_iter().map(|p| -> f64 { p.into() }).sum();
        assert_allclose(&total, &1.0, 1e-100);
        assert_allclose(
            &part.proba(0).unwrap().as_f64(),
            &(part.proba(1).unwrap().as_f64() * 2.0),
            1e-15,
        );
    }
}
