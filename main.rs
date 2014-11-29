//#![feature(overloaded_calls)]
#![feature(unboxed_closures)]
#![allow(non_snake_case)]
#![allow(dead_code)]  // TODO: Remove eventually

use std::fmt;

struct Mat {
    r: uint,
    c: uint,
    data: Vec<f32>  // Column major
}

fn _idx(r: uint, i: uint, j: uint) -> uint {
    j * r + i
}

impl Mat {
    fn new() -> Mat {
        Mat{r: 0, c: 0, data: Vec::new()}
    }

    fn zero(r: uint, c: uint) -> Mat {
        let d = Vec::from_elem(r * c, 0.0);
        Mat{r: r, c: c, data: d}
    }

    fn ident(n: uint) -> Mat {
        let mut d = Vec::from_elem(n * n, 0.0);
        for i in range(0, n) {
            d[i * n + i] = 1.0;
        }
        Mat{r: n, c: n, data: d}
    }

    fn from_slice(r: uint, c: uint, rowmajor: &[f32]) -> Mat {
        // Data must be converted from row-major
        let mut data = Vec::with_capacity(r * c);
        for j in range(0, c) {
            for i in range(0, r) {
                data.push(rowmajor[i * c + j]);
            }
        }
        Mat{r: r, c: c, data: data}
    }

    fn ind(&self, i: uint, j: uint) -> uint {
        j * self.r + i
    }

    fn at(&self, i: uint, j: uint) -> f32 {
        self.data[self.ind(i, j)]
    }

    fn at_mut<'a>(&'a mut self, i: uint, j: uint) -> &'a mut f32 {
        let i = self.ind(i, j);
        self.data.index_mut(&i)
    }

    fn set(&mut self, i: uint, j: uint, x: f32) {
        let i = self.ind(i, j);
        self.data[i] = x;
    }

    fn lu(&self) -> LU {
        LU::of(self)
    }

    fn row_add(&mut self, src: uint, dst: uint, c: f32) {
        for j in range(0, self.c) {
            *self.at_mut(dst, j) += c * self.at(src, j);
        }
    }

    fn col_add(&mut self, src: uint, dst: uint, c: f32) {
        for i in range(0, self.r) {
            *self.at_mut(i, dst) += c * self.at(i, src);
        }
    }
}

impl Clone for Mat {
    fn clone(&self) -> Mat {
        Mat{r: self.r, c: self.c, data: self.data.clone()}
    }
}

impl Fn<(uint, uint), f32> for Mat {
    extern "rust-call" fn call(&self, args: (uint, uint)) -> f32 {
        let (i, j) = args;
        self.data[self.ind(i, j)]
    }
}

/*
impl<'a> FnMut<(uint, uint), &'a f32> for Mat {
    extern "rust-call" fn call_mut<'a>(&'a mut self, args: (uint, uint)) -> &'a f32 {
        let (i, j) = args;
        let foo: &mut f32 = &self.data[self.ind(i, j)];
        //&self.data[self.ind(i, j)]
        foo
    }
}
*/


impl fmt::Show for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in range(0, self.r) {
            for j in range(0, self.c) {
                try!(write!(f, "{}  ", self.at(i, j)))
            }
            try!(write!(f, "\n"))
        }
        write!(f, "")
    }
}

impl PartialEq for Mat {
    fn eq(&self, other: &Mat) -> bool {
        if self.r != other.r || self.c != other.c {
            return false;
        }

        range(0, self.data.len()).all(|i| self.data[i] == other.data[i])
    }
}

impl Add<Mat, Mat> for Mat {
    fn add(&self, rhs: &Mat) -> Mat {
        if self.r != rhs.r || self.c != rhs.c {
            panic!("Size mismatch in Add: ({}, {}) vs ({}, {})",
                   self.r, self.c, rhs.r, rhs.c);
        }

        Mat{r: self.r, c: self.c,
            data: Vec::from_fn(self.data.len(),
                               |i| self.data[i] + rhs.data[i])}
    }
}

impl Sub<Mat, Mat> for Mat {
    fn sub(&self, rhs: &Mat) -> Mat {
        if self.r != rhs.r || self.c != rhs.c {
            panic!("Size mismatch in Sub: ({}, {}) vs ({}, {})",
                   self.r, self.c, rhs.r, rhs.c);
        }

        Mat{r: self.r, c: self.c,
            data: Vec::from_fn(self.data.len(),
                               |i| self.data[i] - rhs.data[i])}
    }
}

impl Mul<Mat, Mat> for Mat {
    fn mul(&self, rhs: &Mat) -> Mat {
        if self.c != rhs.r {
            panic!("Size mismatch in Mul: ({}, {}) * ({}, {})",
                   self.r, self.c, rhs.r, rhs.c);
        }

        // Inefficient
        let mut data = Vec::from_elem(self.r * rhs.c, 0.0);
        for i in range(0, self.r) {
            for j in range(0, rhs.c) {
                data[_idx(self.r, i, j)] =
                    range(0, self.c).fold(0.0, |x, k| x + self.at(i, k) * rhs.at(k, j));
            }
        }

        Mat{r: self.r, c: rhs.c, data: data}
    }
}

struct Permutation {
    v: Vec<uint>,
}

impl Permutation {
    fn ident(n: uint) -> Permutation {
        let v = Vec::from_fn(n, |i| i);
        Permutation{v: v}
    }

    fn swap(n: uint, i: uint, j: uint) -> Permutation {
        let v = Vec::from_fn(
            n,
            |k|
            if k == i { j }
            else if k == j { i }
            else { k });
        Permutation{v: v}
    }

    fn len(&self) -> uint {
        self.v.len()
    }

    // Equavalent to Permutation::swap(i, j) * self
    fn swap_left(&mut self, i: uint, j: uint) {
        let val = self.v[i];
        self.v[i] = self.v[j];
        self.v[j] = val;
    }

    fn to_mat(&self) -> Mat {
        let mut m = Mat::zero(self.v.len(), self.v.len());
        for (i, j) in self.v.iter().enumerate() {
            m.set(i, *j, 1.0);
        }
        m
    }
}

impl Index<uint, uint> for Permutation {
    fn index<'a>(&'a self, _index: &uint) -> &'a uint {
        self.v.index(_index)
    }
}

impl Mul<Permutation, Permutation> for Permutation {
    fn mul(&self, rhs: &Permutation) -> Permutation {
        let v = Vec::from_fn(self.len(), |i| rhs[self[i]]);
        Permutation{v: v}
    }
}

impl Mul<Mat, Mat> for Permutation {
    fn mul(&self, rhs: &Mat) -> Mat {
        if self.v.len() != rhs.r {
            panic!("Cannot permute matrix of mismatched size");
        }
        let mut m = Mat::zero(rhs.r, rhs.c);  // TODO: slow
        for i in range(0, rhs.r) {
            for j in range(0, rhs.c) {
                *m.at_mut(i, j) = rhs.at(i, j);
            }
        }
        m
    }
}

struct LU {
    // TODO: Store in a single mat
    L: Mat,
    U: Mat,
}

impl LU {
    pub fn of(A: &Mat) -> LU {
        if A.r != A.c {
            panic!("Cannot take LU of a non-square matrix");
        }
        let mut L = Mat::ident(A.r);
        let mut U = A.clone();
        let mut P = Permutation::ident(A.r);

        // Reduce from row i, using pivot at (i, i)
        for i in range(0, A.c - 1) {
            if U(i, i) == 0.0 {
                panic!("TODO: entry is zero.  implement pivoting in LU");
                // Finds a non-zero entry
                let mut nonzero = i;
                for j in range(i + 1, U.r) {
                    if U(j, i) != 0.0 {  // TODO: near, rather than equals
                        nonzero = j;
                        break
                    }
                }

                if nonzero == i {
                    panic!("Matrix is non-invertable.  Cannot take LU");
                }

                
            }

            // Reduces row j (from row i)
            for j in range(i + 1, A.r) {
                let x = U(j, i) / U(i, i);
                U.row_add(i, j, -x);
                L.col_add(j, i, x);  // TODO: Only one entry is created
            }
        }

        LU{L: L, U: U}
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
        for i in range(0, b.len()) {
            x[i] = b[i];
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
}

fn main() {
    /*
    println!("Hello!");

    let mut a = Mat::zero(2, 2);

    println!("a = \n{}", a);

    println!("a[0, 0] = {}", a(0, 0));
    {
        let x: &mut f32 = a.atmut(0, 0);
        //a.atmut(0, 0) = 1.0;
        *x = 1.0;
    }
    *a.atmut(0, 0) = 2.0;
    //a(0, 0) = 1.0;
    println!("a[0, 0] = {}", a(0, 0));
    println!("a = \n{}", a);

    let a = Mat::from_slice(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let b = Mat::from_slice(2, 2, [5.0, 6.0, 7.0, 8.0]);

    println!("a = \n{}", a);
    println!("b = \n{}", b);

    let c = a + b;
    println!("c = \n{}", c);
     */
}

#[test]
fn test_add() {
    let a = Mat::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b = Mat::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
    let c = a + b;

    assert_eq!(c(0, 0), 6.0);
    assert_eq!(c(0, 1), 8.0);
    assert_eq!(c(1, 0), 10.0);
    assert_eq!(c(1, 1), 12.0);
}

#[test]
#[should_fail]
fn test_add_mismatched_dimensions () {
    let a = Mat::from_slice(2, 1, &[1.0, 2.0]);
    let b = Mat::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
    let c = a + b;
    drop(c);
}

#[test]
fn test_subtract() {
    let a = Mat::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
    let b = Mat::from_slice(2, 2, &[2.0, 4.0, 6.0, 8.0]);
    let c = a - b;

    assert_eq!(c(0, 0), 3.0);
    assert_eq!(c(0, 1), 2.0);
    assert_eq!(c(1, 0), 1.0);
    assert_eq!(c(1, 1), 0.0);
}

#[test]
fn test_mul_ident() {
    let i1 = Mat::ident(2);
    let i2 = Mat::ident(2);

    let x = i1 * i2;
    assert_eq!(x(0, 0), 1.0);
    assert_eq!(x(0, 1), 0.0);
    assert_eq!(x(1, 0), 0.0);
    assert_eq!(x(1, 1), 1.0);
}

#[test]
fn test_mul_simple() {
    let a = Mat::from_slice(2, 3, &[1.0, 2.0, 3.0,
                                    4.0, 5.0, 6.0]);
    let b = Mat::from_slice(3, 4, &[2.0, 4.0, 6.0, 8.0,
                                    10.0, 12.0, 14.0, 16.0,
                                    18.0, 20.0, 22.0, 24.0]);
    let c = a * b;

    let expect = Mat::from_slice(2, 4, &[76.0, 88.0, 100.0, 112.0,
                                         166.0, 196.0, 226.0, 256.0]);

    assert_eq!(c, expect);
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
fn test_perm_identity() {
    let p = Permutation::ident(3);
    let m = p.to_mat();
    for i in range(0, 3) {
        for j in range(0, 3) {
            if i == j {
                assert_eq!(m(i, j), 1.0);
            }
            else {
                assert_eq!(m(i, j), 0.0);
            }
        }
    }
}

#[test]
fn test_perm_swaps() {
    let mut p = Permutation::ident(3);
    p.swap_left(0, 1);
    p.swap_left(1, 2);  // 0 goes into 2, 2 goes into 1
    assert_eq!(p[0], 1);
    assert_eq!(p[1], 2);
    assert_eq!(p[2], 0);
}

#[test]
fn test_2x2_perm_simple() {
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
