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

        let mut v = x - x.norm() * e1;
        v.normalize();

        // H := I - 2vv'
        // R := H * R

        R -= 2 * v * (v.t() * R);

        //R.block_lr() -= 2 * v * (v.t() * R.block_lr());
        
        

        QR{Q: Q, R: R}
    }
}

