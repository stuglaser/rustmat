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
        let R = A.clone();
        let Q = Mat::ident(A.r);

        let x = A.col(0);
        //let v = ;

        QR{Q: Q, R: R}
    }
}

