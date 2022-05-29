//! Kalman filters
use ndarray::prelude::*;
use crate::cov::Covariance;

/// Updates the provided state to take into account the given observation
/// 
pub fn observe_linear(
    state_estimate: &mut Array1<f64>,
    state_noise: &mut Covariance,
    observation: &Array1<f64>,
    observation_model: &Array2<f64>,
    observation_noise: &Covariance,
) -> Result<(), ()> {

}
