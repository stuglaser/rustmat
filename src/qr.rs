use base::{MatBase, Mat, SubAssign, BlockTrait};
use std::num::Float;

// QR Decomposition
//
// TODO: We can optimize space usage by storing Q implicitly in the
// lower-tri part of R
//
// See: http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf

struct QR {
    Q: Mat,
    R: Mat,
}

impl QR {
    pub fn of(A: &Mat) -> QR {
        if !A.is_square() {
            panic!("QR must apply to square matrix");
        }
        let mut R = A.clone();
        let mut Q = Mat::ident(A.r);

        let mut e1 = Mat::zero(R.rows(), 1);  // Avoids reallocation

        for j in range(0, R.cols()) {
            //println!("Iteration {}", j);
            //println!("Q before:\n{}", Q);
            //println!("R before:\n{}", R);
            //println!("Q*R before:\n{}", &Q * &R);

            let rows = R.rows();  // Borrow checker limitations
            let cols = R.cols();
            let mut Rb = R.block_mut(j, j, rows, cols);

            // All I want for christmas is a smarter borrow checker
            let w = {
                let x = Rb.col(0);
                let norm = x.norm();
                e1[(0, 0)] = norm.signum() * norm;
                //println!("|x| * e = \n{}", e1);
                let mut v = x - &e1.block(0, 0, rows - j, 1);
                //println!("v = \n{}", v);
                if !v.is_zero() {
                    v.normalize();
                }
                v
            };

            //println!("w = \n{}", w);

            //let H = Mat::ident(Rb.rows()) - &w * &w.t() * 2.0;
            //println!("H = \n{}", H);

            // H := I - 2ww'
            // R := H * R
            let tmp = &w.t() * &Rb;  // Borrow checker limitations (for R)
            Rb.sub_assign(&w * tmp * 2.0);

            // Q := Q * H
            let mut Qb = Q.block_mut(0, j, rows, cols);
            let tmp = &Qb * &w;
            Qb.sub_assign(tmp * w.t() * 2.0);

        }

        //println!("Q final:\n{}", Q);
        //println!("R final:\n{}", R);
        //println!("Q*R final:\n{}", &Q * &R);

        QR{Q: Q, R: R}
    }
}

#[test]
fn test_qr_simple() {
    let A = Mat::from_slice(3, 3, &[0.0, 1.0, 1.0,
                                    1.0, 1.0, 2.0,
                                    0.0, 0.0, 3.0]);
    let qr = QR::of(&A);

    let Q = Mat::from_slice(3, 3, &[0.0, 1.0, 0.0,
                                    1.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0]);
    let R = Mat::from_slice(3, 3, &[1.0, 1.0, 2.0,
                                    0.0, 1.0, 1.0,
                                    0.0, 0.0, 3.0]);
    assert_mat_near!(qr.Q, Q, 1e-6);
    assert_mat_near!(qr.R, R, 1e-6);
}

#[test]
fn test_qr_wikipedia() {
    let A = Mat::from_slice(3, 3, &[12.0, -51.0, 4.0,
                                    6.0, 167.0, -68.0,
                                    -4.0, 24.0, -41.0]);
    let qr = QR::of(&A);

    let Q = Mat::from_slice(3, 3, &[6./7., -69./175., -58./175.,
                                    3./7., 158./175., 6./175.,
                                    -2./7., 6./35., -33./35.]);
    let R = Mat::from_slice(3, 3, &[14., 21., -14.,
                                    0., 175., -70.,
                                    0., 0., 35.]);
    assert_mat_near!(qr.Q, Q, 1e-4);
    assert_mat_near!(qr.R, R, 1e-4);
}
