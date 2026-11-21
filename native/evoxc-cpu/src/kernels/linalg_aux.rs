//! Matrix factorization / spectral kernels (`eigh`, `cholesky`, `qr`,
//! `matrix_rank`, `svd`, `matrix_exp`).
//!
//! These are the numpy backend's LAPACK-shaped kernels; they are grouped in
//! their own file because each is a self-contained algorithm rather than a
//! strided loop.
use super::KernelMap;

pub fn register(_map: &mut KernelMap) {}
