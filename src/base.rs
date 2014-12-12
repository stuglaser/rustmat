use std::fmt;
use std::ptr;

use lu::LU;

pub trait VecBase : Index<uint, f32> {
    fn len(&self) -> uint;
}

pub trait RowVec : VecBase {
    pub fn t(&self) -> ColVec;
}

pub trait ColVec : VecBase {
    pub fn t(&self) -> RowVec;
}

pub trait MatTrait : Index<(uint, uint), f32> {
    pub fn rows(&self) -> uint;
    pub fn cols(&self) -> uint;
    pub fn t<'a>(&'a self) -> Transposed<'a, Self>;

    pub fn row(&self, i: uint) -> RowVec;
    pub fn col(&self, j: uint) -> ColVec;
}

pub struct Mat {
    pub r: uint,
    pub c: uint,
    pub data: Vec<f32>  // Column major
}

impl Mat {
    pub fn new() -> Mat {
        Mat{r: 0, c: 0, data: Vec::new()}
    }

    pub fn zero(r: uint, c: uint) -> Mat {
        let d = Vec::from_elem(r * c, 0.0);
        Mat{r: r, c: c, data: d}
    }

    pub fn ident(n: uint) -> Mat {
        let mut d = Vec::from_elem(n * n, 0.0);
        for i in range(0, n) {
            d[i * n + i] = 1.0;
        }
        Mat{r: n, c: n, data: d}
    }

    pub fn from_slice(r: uint, c: uint, rowmajor: &[f32]) -> Mat {
        // Data must be converted from row-major
        let mut data = Vec::with_capacity(r * c);
        for j in range(0, c) {
            for i in range(0, r) {
                data.push(rowmajor[i * c + j]);
            }
        }
        Mat{r: r, c: c, data: data}
    }

    pub fn ind(&self, i: uint, j: uint) -> uint {
        j * self.r + i
    }

    pub fn at(&self, i: uint, j: uint) -> f32 {
        self.data[self.ind(i, j)]
    }

    pub fn at_mut<'a>(&'a mut self, i: uint, j: uint) -> &'a mut f32 {
        let i = self.ind(i, j);
        self.data.index_mut(&i)
    }

    pub fn set(&mut self, i: uint, j: uint, x: f32) {
        let i = self.ind(i, j);
        self.data[i] = x;
    }

    pub fn is_square(&self) -> bool {
        self.r == self.c
    }

    pub fn lu(&self) -> LU {
        LU::of(self)
    }

    pub fn row_add(&mut self, src: uint, dst: uint, c: f32) {
        for j in range(0, self.c) {
            *self.at_mut(dst, j) += c * self.at(src, j);
        }
    }

    pub fn col_add(&mut self, src: uint, dst: uint, c: f32) {
        for i in range(0, self.r) {
            *self.at_mut(i, dst) += c * self.at(i, src);
        }
    }

    pub fn swap_row(&mut self, a: uint, b: uint) {
        for j in range(0, self.c) {
            unsafe {
                ptr::swap(self.at_mut(a, j), self.at_mut(b, j));
            }
        }
    }
}

impl MatTrait for Mat {
    fn rows(&self) -> uint {
        self.r
    }

    fn cols(&self) -> uint {
        self.c
    }
    
    fn t<'a>(&'a self) -> Transposed<'a, Mat> {
        Transposed{m: self}
    }

    pub fn row(&self, i: uint) -> RowView<Mat> {
        if i >= self.r {
            panic!("Row index out of bounds");
        }
        RowView{m: self, row: i}
    }

    pub fn col(&self, j: uint) -> ColView<Mat> {
        if j >= self.c {
            panic!("Column index out of bounds");
        }
        ColView{m: self, col: j}
    }
}

struct Transposed<'a, T: MatTrait + 'a> {
    m: &'a T
}

impl<'a, T: MatTrait + 'a> MatTrait for Transposed<'a, T> {
    fn rows(&self) -> uint {
        self.m.cols()
    }

    fn cols(&self) -> uint {
        self.m.rows()
    }
    
    fn t<'r>(&'r self) -> Transposed<'r, Transposed<'a, T>> {
        Transposed{m: self}
    }

    pub fn row(&self, i: uint) -> RowView<Transposed<'a, T>> {
        if i >= self.r {
            panic!("Row index out of bounds");
        }
        RowView{m: self, row: i}
    }

    pub fn col(&self, j: uint) -> ColView<Transposed<'a, T>> {
        if j >= self.c {
            panic!("Column index out of bounds");
        }
        ColView{m: self, col: j}
    }
}

impl<'a, T:MatTrait + 'a> Index<(uint, uint), f32> for Transposed<'a, T> {
    fn index<'r>(&'r self, &(i, j): &(uint, uint)) -> &f32 {
        self.m.index(&(j, i))
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

impl Index<(uint, uint), f32> for Mat {
    fn index(&self, &(i, j): &(uint, uint)) -> &f32 {
        self.data.index(&self.ind(i, j))
    }
}


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
                //data[_idx(self.r, i, j)] =
                data[self.ind(i, j)] =
                    range(0, self.c).fold(0.0, |x, k| x + self.at(i, k) * rhs.at(k, j));
            }
        }

        Mat{r: self.r, c: rhs.c, data: data}
    }
}

pub struct ColView<'a, T:MatTrait + 'a> {
    m: &'a T,
    col: uint
}

impl<'a, T:MatTrait> VecBase for ColView<'a, T> {
    fn len(&self) -> uint {
        self.m.r
    }
}

impl<'a, T:MatTrait> ColVec for ColView<'a, T> {
    fn t(&self) -> RowView<Transposed<'a, T>> {
        self.m.t().row(self.col)
    }
}

impl<'a, T:MatTrait> Index<uint, f32> for ColView<'a, T> {
    fn index(&self, index: &uint) -> &f32 {
        self.m.index(&(*index, self.col))
    }
}


pub struct RowView<'a, T:MatTrait + 'a> {
    m: &'a T,
    row: uint
}

impl<'a, T:MatTrait> VecBase for RowView<'a, T> {
    fn len(&self) -> uint {
        self.m.c
    }
}

impl<'a, T:MatTrait> RowVec for RowView<'a, T> {
    fn t(&self) -> ColView<'a, T> {
        self.m.t().col(self.row)
    }
}

impl<'a, T:MatTrait> Index<uint, f32> for RowView<'a, T> {
    fn index(&self, index: &uint) -> &f32 {
        self.m.index(&(self.row, *index))
    }
}


pub struct Permutation {
    v: Vec<uint>,
}

impl Permutation {
    pub fn ident(n: uint) -> Permutation {
        let v = Vec::from_fn(n, |i| i);
        Permutation{v: v}
    }

    pub fn swap(n: uint, i: uint, j: uint) -> Permutation {
        let v = Vec::from_fn(
            n,
            |k|
            if k == i { j }
            else if k == j { i }
            else { k });
        Permutation{v: v}
    }

    pub fn len(&self) -> uint {
        self.v.len()
    }

    // Equavalent to Permutation::swap(i, j) * self
    pub fn swap_left(&mut self, i: uint, j: uint) {
        let val = self.v[i];
        self.v[i] = self.v[j];
        self.v[j] = val;
    }

    pub fn to_mat(&self) -> Mat {
        let mut m = Mat::zero(self.v.len(), self.v.len());
        for (i, j) in self.v.iter().enumerate() {
            m.set(i, *j, 1.0);
        }
        m
    }

    pub fn inv(&self) -> Permutation {
        let mut v = Vec::from_elem(self.v.len(), 0);
        for (a, b) in self.v.iter().enumerate() {
            v[*b] = a;
        }
        Permutation{v: v}
    }
}

impl fmt::Show for Permutation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.v.fmt(f)
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
                *m.at_mut(i, j) = rhs.at(self[i], j);
            }
        }
        m
    }
}

impl<T:Clone> Mul<Vec<T>, Vec<T>> for Permutation {
    fn mul(&self, rhs: &Vec<T>) -> Vec<T> {
        if self.len() != rhs.len() {
            panic!("Permutation len doesn't match Vec len");
        }
        Vec::from_fn(rhs.len(), |i| rhs[self[i]].clone())
    }
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
fn test_perm_vec() {
    let mut p = Permutation::ident(3);
    p.swap_left(1, 2);
    p.swap_left(0, 1);

    let a : Vec<f32> = vec!{1.0, 2.0, 3.0};
    let b : Vec<f32> = vec!{3.0, 1.0, 2.0};
    assert_vec_near!(p * a, b, 0.00001);
}

#[test]
fn test_perm_mat() {
    let A = Mat::from_slice(
        3, 3,
        &[1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
          7.0, 8.0, 9.0]);

    let mut p = Permutation::ident(3);
    p.swap_left(0, 1);
    p.swap_left(1, 2);

    let B = p * A;
    let expect = Mat::from_slice(
        3, 3,
        &[4.0, 5.0, 6.0,
          7.0, 8.0, 9.0,
          1.0, 2.0, 3.0]);

    assert_mat_near!(B, expect, 0.00001);
}

#[test]
fn test_col_view() {
    let a = Mat::from_slice(
        3, 4, &[2.0, 4.0, 6.0, 8.0,
                10.0, 12.0, 14.0, 16.0,
                18.0, 20.0, 22.0, 24.0]);

    let c0 = a.col(0);
    assert_eq!(c0[0], 2.0);
    assert_eq!(c0[1], 10.0);

    let c2 = a.col(2);
    assert_eq!(c2[0], 6.0);
    assert_eq!(c2[2], 22.0);
}

#[test]
fn test_transpose_vecs() {
    let a = Mat::from_slice(3, 1, &[2.0, 4.0, 6.0]);
    let v = a.col(0);
    let vt = v.t();

    assert_eq!(vt * v, 56.0);
    assert_mat_near!(v * vt,
                     Mat::from_slice(3, 3,
                                     &[4.0, 8.0, 12.0,
                                       8.0, 16.0, 24.0,
                                       12.0, 24.0, 36.0]),
                     0.00001);
}
