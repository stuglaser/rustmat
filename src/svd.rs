use base::{MatBase, MatBaseMut, Mat, BlockTrait};
use householder::reflector_to_e1;
use std::num::Float;

// SVD Decomposition

struct SVD {
    U: Mat,
    S: Mat,  // TODO: DiagMat
    V: Mat,
}

fn is_bidiagonal<T:MatBase>(A: &T, eps: f32) -> bool {
    for coor in A.coor_iter() {
        let on_bidiag = coor.0 == coor.1 || coor.0 + 1 == coor.1;
        if !on_bidiag && A[coor].abs() > eps {
            return false
        }
    }
    true
}

fn bidiagonalize(A: Mat) -> (Mat, Mat, Mat) {
    let mut B = A.resolved();
    let mut U = Mat::ident(B.rows());
    let mut V = Mat::ident(B.cols());

    let rows = B.rows();  // Borrow checker can't handle simple things
    let cols = B.cols();

    for i in range(0, A.rows() - 1) {  // TODO: min(rows, cols)?
        let mut Bb = B.block_mut(i, i, rows, cols);

        // Reduce the column
        let Hcol = reflector_to_e1(&Bb.col(0));
        Hcol.lapply(&mut Bb);
        Hcol.rapply(&mut U.block_right_mut(i));

        // Reduce the row
        let Hrow = reflector_to_e1(&Bb.block(0, 1, 1, Bb.cols()).t());
        Hrow.rapply(&mut Bb.block_right_mut(1));
        Hrow.lapply(&mut V.block_bottom_mut(i + 1));
    }

    // Final element (just to make it positive)
    let last = (B.rows() - 1, B.cols() - 1);
    if B[last] < 0.0 {
        B[last] *= -1.0;
        // Negates the right column of U
        let uright = U.cols() - 1;
        for i in range(0, U.rows()) {
            U[(i, uright)] *= -1.0;
        }
    }

    (U, B, V)
}

impl SVD {
    pub fn of(A: &Mat) -> SVD {
        SVD{U: Mat::ident(2), S: Mat::ident(2), V: Mat::ident(2)}
    }
}

#[test]
fn test_bidiagonalize() {
    let A = Mat::from_slice(4, 4,
                            &[1., 3., 3., -2.,
                              2., -2., 4., 8.,
                              2., 5., 5., 1.,
                              -1., 1., 4., 3.]);
    let (Ub, B, Vb) = bidiagonalize(A.clone());

    if !is_bidiagonal(&B, 1e-4) {
        panic!("Isn't bidiagonal:\n{}", B);
    }

    let Ar = Ub * B * Vb;
    assert_mat_near!(Ar, A, 1e-4);
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
