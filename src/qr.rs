use base::{MatBase, Mat, SubAssign, BlockTrait};

struct QR {
    // TODO: Store in a single mat
    Q: Mat,
    R: Mat,
}

fn _is_zero<T:MatBase>(mat: &T) -> bool {
    for coor in mat.coor_iter() {
        if mat[coor] != 0.0 {
            return false;
        }
    }
    true
}

impl QR {
    pub fn of(A: &Mat) -> QR {
        if !A.is_square() {
            panic!("QR must apply to square matrix");
        }
        let mut R = A.clone();
        let mut Q = Mat::ident(A.r);

        println!("inital Q:\n{}", Q);
        println!("inital R:\n{}", R);


        let mut e1 = Mat::zero(R.rows(), 1);

        for j in range(0, R.cols() - 1) {
            println!("Iteration {}", j);
            println!("Q before:\n{}", Q);
            println!("R before:\n{}", R);
            println!("Q*R before:\n{}", &Q * &R);
 
            let rows = R.rows();
            let cols = R.cols();
            let mut Rb = R.block_mut(j, j, rows, cols);

            // Skips this column if it's already upper-tri
            if _is_zero(&Rb.block(1, 0, Rb.rows(), 1)) {
                println!("Column is already upper-tri");
                continue;
            }

            // All I want for christmas is a smarter borrow checker
            let w = {
                let x = Rb.col(0);
                e1[(0, 0)] = x.norm();
                println!("|x| * e = \n{}", e1);
                let mut v = x - &e1.block(0, 0, rows - j, 1);
                println!("v = \n{}", v);
                v.normalize();
                v
            };

            println!("w = \n{}", w);

            let H = Mat::ident(Rb.rows()) - &w * &w.t() * 2.0;
            println!("H = \n{}", H);

            // H := I - 2ww'
            // R := H * R
            let tmp = &w.t() * &Rb;  // Borrow checker limitations (for R)
            Rb.sub_assign(&w * tmp * 2.0);
            //R.block_lr() -= 2 * v * (v.t() * R.block_lr());

            // Q := Q * H
            let mut Qb = Q.block_mut(j, j, rows, cols);  // TODO: is this the correct block to take???
            let tmp = &Qb * &w;
            Qb.sub_assign(tmp * w.t() * 2.0);

        }

        println!("Q final:\n{}", Q);
        println!("R final:\n{}", R);
        println!("Q*R final:\n{}", &Q * &R);

        // A = QR
        // Q' * A = R
        
        
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

#[test]
fn test_qr_wikipedia() {
    let A = Mat::from_slice(3, 3, &[12.0, -51.0, 4.0,
                                    6.0, 167.0, -68.0,
                                    -4.0, 24.0, -41.0]);
    let qr = QR::of(&A);
}
