use base::{MatBase, Mat, SubAssign};

struct QR {
    // TODO: Store in a single mat
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

        let x = R.col(0);

        let e1 = Mat::zero(R.rows(), 1);


        let mut v = x - e1 * x.norm();
        v.normalize();

        // H := I - 2vv'
        // R := H * R

        //let bar = v * (v.t() * R) * 2.0;  // ICE
        //let foo: Mat = v * (v.t() * R) * 2.0;
        //R.sub_assign(foo);
        R.sub_assign(v * (v.t() * R) * 2.0);

        //R.block_lr() -= 2 * v * (v.t() * R.block_lr());
        
        

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
}
