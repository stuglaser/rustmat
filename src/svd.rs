use base::{MatBase, Mat, SubAssign, BlockTrait};
use std::num::Float;

// SVD Decomposition

struct SVD {
    U: Mat,
    S: Mat,  // TODO: DiagMat
    V: Mat,
}

impl SVD {
    pub fn of(A: &Mat) -> SVD {
        SVD{U: Mat::ident(2), S: Mat::ident(2), V: Mat::ident(2)}
    }
}

#[test]
fn test_svd_simple() {
    let A = Mat::from_slice(
        2, 3, &[3., 2., 2.,
                2., 3., -2.]);
    let svd = SVD::of(&A);

    let rt2 = 1. / (2.0 as f32).sqrt();
    let rt18 =  1. / (18.0 as f32).sqrt();
    let U = Mat::from_slice(2, 2, &[rt2, rt2, rt2, -rt2]);
    let S = Mat::from_slice(2, 2, &[5., 0., 0., 0., 3., 0.]);
    let Vt = Mat::from_slice(3, 3, &[rt2, rt2, 0.,
                                     rt18, -rt18, 4. * rt18,
                                     2./3., -2./3., -1./3.]);
}
