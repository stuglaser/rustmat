#![macro_escape]

#[macro_export]
macro_rules! assert_vec_near(
    ($a_expr:expr, $b_expr:expr, $eps_expr:expr) => {
        match (&($a_expr), &($b_expr), &($eps_expr)) {
            (a, b, eps) => {
                use std::num::Float;
                if a.len() != b.len() {
                    panic!("Vec sizes don't match: {} vs {}", a.len(), b.len());
                }
                for i in range(0, a.len()) {
                    if (a[i] - b[i]).abs() > *eps {
                        panic!("Vec's aren't equal: {} and {}", a, b);
                    }
                }
            }
        }
    }
)

macro_rules! assert_mat_near(
    ($A_expr:expr, $B_expr:expr, $eps_expr:expr) => {
        match (&($A_expr), &($B_expr), &($eps_expr)) {
            (A, B, eps) => {
                use std::num::Float;
                if A.r != B.r || A.c != B.c {
                    panic!("Matrix sizes don't match: ({}, {}) vs ({}, {})",
                           A.r, A.c, B.r, B.c);
                }
                for i in range(0, A.r) {
                    for j in range(0, A.c) {
                        let x : f32 = A.at(i, j) - B.at(i, j);
                        if x.abs() > *eps {
                            panic!("Matrices aren't equal: \n{} and \n{}", A, B);
                        }
                    }
                }
            }
        }
    }
)

