use base::{MatBase, Mat, Permutation};

pub struct LU {  // P * A = L * U
    // TODO: Store in a single mat
    L: Mat,
    U: Mat,
    P: Permutation,
}

impl LU {
    pub fn of(A: &Mat) -> LU {
        if A.rows() != A.cols() {
            panic!("Cannot take LU of a non-square matrix");
        }
        let mut L = Mat::ident(A.r);
        let mut U = A.clone();
        let mut P = Permutation::ident(A.r);

        // Reduce from row i, using pivot at (i, i)
        for i in range(0, A.c - 1) {
            if U(i, i) == 0.0 {
                // Finds a non-zero entry
                let mut nonzero = 0u;
                for j in range(i + 1, U.r) {
                    if U(j, i) != 0.0 {  // TODO: near, rather than equals
                        nonzero = j;
                        break
                    }
                }

                if nonzero == 0 {
                    panic!("Matrix is non-invertable.  Cannot take LU");
                }

                P.swap_left(i, nonzero);
                U.swap_row(i, nonzero);
            }

            // Reduces row j (from row i)
            for j in range(i + 1, A.r) {
                let x = U(j, i) / U(i, i);
                U.row_add(i, j, -x);
                L.set(j, i, x);
            }

            println!("After rref {}:\nP = {}\nL = \n{}\nU = \n{}", i, P, L, U);
        }

        LU{L: L, U: U, P: P}
    }

    pub fn solve(&self, b: &Vec<f32>) -> Vec<f32> {
        if self.L.r != b.len() {
            panic!("Vec has wrong length for LU solving");
        }

        let mut x = Vec::from_elem(b.len(), 0.0);

        // Propagate through L-inverse
        //
        // x := L^-1 * b
        // x(i) := b(i) - x(0)*L(i,0) - x(1)*L(i,1) - ... - x(i-1)*L(i,i-1)
        for i in range(0, x.len()) {
            x[i] = b[self.P[i]];
            for j in range(0, i) {
                x[i] -= x[j] * self.L.at(i, j);
            }
        }

        // Propagate through U-inverse
        //
        // x' := U^-1 * x
        // x'(i) := 1/U(i,i) * (x(i) - x'(i+1)*U(i,i+1) - x'(i+2)*U(i,i+2) - ...)
        for i in range(0, x.len()).rev() {
            x[i] = x[i];
            for j in range(i + 1, x.len()) {
                x[i] -= x[j] * self.U.at(i, j);
            }
            x[i] = x[i] / self.U.at(i, i);
        }

        x
    }

    pub fn resolve(&self) -> Mat {
        self.P.inv() * (self.L * self.U)
    }
}

#[test]
fn test_lu_3x3() {
    let A = Mat::from_slice(3, 3, &[1.0, -2.0, 3.0,
                                    2.0, -5.0, 12.0,
                                    -4.0, 2.0, -10.0]);
    let lu = A.lu();
    assert_mat_near!(lu.resolve(), A, 0.001);
}

#[test]
fn test_lu_2x2_perm() {
    let A = Mat::from_slice(
        2, 2,
        &[0.0, 1.0,
          -1.0, 0.0]);

    let lu = A.lu();

    println!("Result of LU:\nP = {}\nL = \n{}\nU = \n{}\n", lu.P, lu.L, lu.U);
    let LU = lu.L * lu.U;
    println!("L * U = \n{}", LU);

    println!("P = {}\nP^-1 = {}", lu.P, lu.P.inv());

    assert_mat_near!(lu.resolve(), A, 0.001);
}

#[test]
fn test_solve_2x2_ident() {
    // Ax = b
    let A = Mat::ident(2);
    let b : Vec<f32> = vec!{1.0, 2.0};

    let lu = A.lu();
    let x = lu.solve(&b);

    assert_eq!(x[0], b[0]);
    assert_eq!(x[1], b[1]);
}

#[test]
fn test_solve_2x2_upper() {
    // Ax = b
    let A = Mat::from_slice(2, 2, &[1.0, 2.0, 0.0, 1.0]);
    let b : Vec<f32> = vec!{1.0, 2.0};

    let lu = A.lu();
    let x = lu.solve(&b);

    assert_eq!(x[0], -3.0);
    assert_eq!(x[1], 2.0);
}

#[test]
fn test_solve_2x2_lower() {
    // Ax = b
    let A = Mat::from_slice(2, 2, &[1.0, 0.0, 2.0, 1.0]);
    let b : Vec<f32> = vec!{1.0, 2.0};

    let lu = A.lu();
    let x = lu.solve(&b);

    assert_eq!(x[0], 1.0);
    assert_eq!(x[1], 0.0);
}

#[test]
fn test_solve_3x3_simple() {
    let A = Mat::from_slice(3, 3, &[1.0, -2.0, 3.0,
                                    2.0, -5.0, 12.0,
                                    0.0, 2.0, -10.0]);
    let b = vec!{3.0, 2.0, 1.0};

    let lu = A.lu();
    let x = lu.solve(&b);
    println!("L = \n{}\nU = \n{}", lu.L, lu.U);

    assert_eq!(x[0], -20.5);
    assert_eq!(x[1], -17.0);
    assert_eq!(x[2], -3.5);
}

#[test]
fn test_2x2_solve_perm_simple() {
    let A = Mat::from_slice(
        2, 2,
        &[0.0, 1.0,
          -1.0, 0.0]);

    let b = vec!{2.0, -3.0};

    let lu = A.lu();
    let x = lu.solve(&b);
    assert_eq!(x[0], 3.0);
    assert_eq!(x[1], 2.0);
}
