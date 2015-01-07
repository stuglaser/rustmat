use base::{MatBase, MatBaseMut, Mat};
use householder::reflector_to_e1;

// QR Decomposition
//
// TODO: We can optimize space usage by storing Q implicitly in the
// lower-tri part of R
//
// See: http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf

pub struct QR {
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

        for j in range(0, R.cols()) {
            //println!("Iteration {}", j);
            //println!("Q before:\n{}", Q);
            //println!("R before:\n{}", R);
            //println!("Q*R before:\n{}", &Q * &R);

            let rows = R.rows();  // Borrow checker limitations
            let cols = R.cols();
            let mut Rb = R.block_mut(j, j, rows, cols);
            
            let H = reflector_to_e1(&Rb.col(0));

            // H := I - 2ww'
            // R := H * R
            H.lapply(&mut Rb);

            // Q := Q * H
            let mut Qb = Q.block_mut(0, j, rows, cols);
            H.rapply(&mut Qb);
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
    assert_mat_near!(qr.R, R, 1e-6);
    assert_mat_near!(qr.Q, Q, 1e-6);
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
    assert_mat_near!(qr.R, R, 1e-4);
    assert_mat_near!(qr.Q, Q, 1e-4);
}
