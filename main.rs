#![feature(overloaded_calls)]

use std::fmt;

//#![feature(overloaded_calls)]

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

    fn atmut<'a>(&'a mut self, i: uint, j: uint) -> &'a mut f32 {
        let i = self.ind(i, j);
        self.data.index_mut(&i)
    }

    fn at(&self, i: uint, j: uint) -> f32 {
        self.data[self.ind(i, j)]
    }

    fn set(&mut self, i: uint, j: uint, x: f32) {
        let i = self.ind(i, j);
        self.data[i] = x;
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

fn main() {
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
}

#[test]
fn test_add() {
    let a = Mat::from_slice(2, 2, [1.0, 2.0, 3.0, 4.0]);
    let b = Mat::from_slice(2, 2, [5.0, 6.0, 7.0, 8.0]);
    let c = a + b;

    if c(0, 0) != 6.0 { panic!(); }
    if c(0, 1) != 8.0 { panic!(); }
    if c(1, 0) != 10.0 { panic!(); }
    if c(1, 1) != 12.0 { panic!(); }
}

#[test]
#[should_fail]
fn test_add_mismatched_dimensions () {
    let a = Mat::from_slice(2, 1, [1.0, 2.0]);
    let b = Mat::from_slice(2, 2, [5.0, 6.0, 7.0, 8.0]);
    let c = a + b;
}

#[test]
fn test_subtract() {
    let a = Mat::from_slice(2, 2, [5.0, 6.0, 7.0, 8.0]);
    let b = Mat::from_slice(2, 2, [2.0, 4.0, 6.0, 8.0]);
    let c = a - b;

    if c(0, 0) != 3.0 { panic!(); }
    if c(0, 1) != 2.0 { panic!(); }
    if c(1, 0) != 1.0 { panic!(); }
    if c(1, 1) != 0.0 { panic!(); }
}

#[test]
fn test_mul_ident() {
    let i1 = Mat::ident(2);
    let i2 = Mat::ident(2);

    let x = i1 * i2;
    if x(0, 0) != 1.0 { panic!(); }
    if x(0, 1) != 0.0 { panic!(); }
    if x(1, 0) != 0.0 { panic!(); }
    if x(1, 1) != 1.0 { panic!(); }
}

#[test]
fn test_mul_simple() {
    let a = Mat::from_slice(2, 3, [1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0]);
    let b = Mat::from_slice(3, 4, [2.0, 4.0, 6.0, 8.0,
                                   10.0, 12.0, 14.0, 16.0,
                                   18.0, 20.0, 22.0, 24.0]);
    let c = a * b;
    
    let expect = Mat::from_slice(2, 4, [76.0, 88.0, 100.0, 112.0,
                                        166.0, 196.0, 226.0, 256.0]);

    println!("c = \n{}", c);
    println!("expect = \n{}", expect);
    if c != expect { panic!(); }
}
