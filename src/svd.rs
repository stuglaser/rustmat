use base::{MatBase, MatBaseMut, Mat};
use householder::reflector_to_e1;
use std::num::Float;

use qr::QR;

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

fn is_diagonal<T:MatBase>(A: &T, eps: f32) -> bool {
    for coor in A.coor_iter() {
        let on_diag = coor.0 == coor.1;
        if !on_diag && A[coor].abs() > eps {
            return false
        }
    }
    true
}

fn is_identity<T:MatBase>(A: &T, eps: f32) -> bool {
    if A.rows() != A.cols() {
        return false;
    }
    
    for coor in A.coor_iter() {
        let on_diag = coor.0 == coor.1;
        if on_diag {
            if (A[coor] - 1.0).abs() > eps {
                return false
            }
        }
        else {
            if A[coor].abs() > eps {
                return false
            }
        }
    }
    true
}

fn is_orthonormal<T:MatBase>(A: &T, eps: f32) -> bool {
    let m = A * A.t();
    if !is_identity(&m, eps) {
        return false;
    }
    for j in range(0, A.cols()) {
        if (A.col(j).norm() - 1.0).abs() > eps {
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

    for i in range(0, A.rows()) {  // TODO: min(rows, cols)?
        let mut Bb = B.block_mut(i, i, rows, cols);

        // Reduce the column
        let Hcol = reflector_to_e1(&Bb.col(0));
        Hcol.lapply(&mut Bb);
        Hcol.rapply(&mut U.block_right_mut(i));

        if i < A.cols() - 1 {
            // Reduce the row
            let Hrow = reflector_to_e1(&Bb.block(0, 1, 1, Bb.cols()).t());
            Hrow.rapply(&mut Bb.block_right_mut(1));
            Hrow.lapply(&mut V.block_bottom_mut(i + 1));
        }
    }

    (U, B, V)
}

impl SVD {
    pub fn of(A: Mat) -> SVD {
        let (mut U, mut S, mut V) = bidiagonalize(A);

        // QR procedure to diagonalize S.  This is a very inefficient
        // and poorly conditioned version, but it's the easiest to
        // write.
        for _ in range(0, 10) {
            let qr = QR::of(&S);
        }
        
        SVD{U: U, S: S, V: V}
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
fn test_bidiagonalize_rect() {
    let A = Mat::from_slice(
        2, 3, &[3., 2., 2.,
                2., 3., -2.]);
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
    let svd = SVD::of(A.clone());

    let rt2 = 1. / (2.0 as f32).sqrt();
    let rt18 =  1. / (18.0 as f32).sqrt();
    let U = Mat::from_slice(2, 2, &[rt2, rt2, rt2, -rt2]);
    let S = Mat::from_slice(2, 2, &[5., 0., 0., 0., 3., 0.]);
    let Vt = Mat::from_slice(3, 3, &[rt2, rt2, 0.,
                                     rt18, -rt18, 4. * rt18,
                                     2./3., -2./3., -1./3.]);

    println!("U = \n{}S=\n{}V=\n{}", svd.U, svd.S, svd.V);

    if !is_diagonal(&svd.S, 1e-4) {
        panic!("S matrix is not diagonal:\n{}", svd.S);
    }

    assert_mat_near!(svd.S, S, 1e-5);
    assert_mat_near!(svd.U, U, 1e-5);
    assert_mat_near!(svd.V, Vt.t(), 1e-5);
}
