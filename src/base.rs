use std::fmt;
use std::iter;
use std::iter::AdditiveIterator;
use std::marker;
use std::num;
use std::ops::{Index, IndexMut, Add, Sub, Mul};
use std::ptr;

use std::slice;

use lu::LU;


pub trait AddAssign<RHS> {
    fn add_assign(&mut self, other: RHS);
}

pub trait SubAssign<RHS> {
    fn sub_assign(&mut self, other: RHS);
}

struct CoorIterator {
    i: usize,
    j: usize,
    rows: usize,
    cols: usize
}

impl CoorIterator {
    fn new(rows: usize, cols: usize) -> CoorIterator {
        CoorIterator{i: 0, j: 0, rows: rows, cols: cols}
    }
}

impl Iterator for CoorIterator {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<(usize, usize)> {
        if self.j == self.cols { // Fell off the end
            None
        }
        else {
            let val = (self.i, self.j);
            self.i += 1;
            if self.i == self.rows {
                self.i = 0;
                self.j += 1;
            }
            Some(val)
        }
    }
}

fn sum_sq<I: Iterator>(iter: I) -> f32 {
    iter.map(|x| x * x).sum()
}

pub trait MatBase : Index<(usize, usize)> + fmt::Show + Sized {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn len(&self) -> usize;
    fn t<'a>(&'a self) -> Transposed<'a, &Self>;
    fn norm(&self) -> f32;

    fn resolved(&self) -> Mat {
        Mat::from_fn(self.rows(), self.cols(),
                     |i, j| self[(i, j)])
    }

    fn coor_iter(&self) -> CoorIterator {
        CoorIterator::new(self.rows(), self.cols())
    }

    fn same_size<T:MatBase>(&self, other: &T) -> bool{
        self.rows() == other.rows() && self.cols() == other.cols()
    }

    fn is_column(&self) -> bool {
        self.cols() == 1
    }

    fn is_row(&self) -> bool {
        self.rows() == 1
    }

    fn is_zero(&self) -> bool {
        for coor in self.coor_iter() {
            if self[coor] != 0.0 {
                return false;
            }
        }
        true
    }

    fn col(&self, j: usize) -> Block<&Self> {
        if j >= self.cols() {
            panic!("Column index out of bounds");
        }
        make_block(self, 0, j, self.rows(), j + 1)
    }

    fn block<'a>(&self, i0: usize, j0: usize, i1: usize, j1: usize) -> Block<'a, &Self> {
        make_block(self, i0, j0, i1, j1)
    }

    fn block_right<'a>(&'a mut self, j: usize) -> Block<'a, &Self> {
        make_block(self, 0, j, self.rows(), self.cols())
    }
}

pub trait MatBaseMut : MatBase + IndexMut<(usize, usize)> + Sized {
    //fn t_mut<'a>(&'a mut self) -> Transposed<'a, &mut Self>;

    fn block_mut<'a>(&mut self, i0: usize, j0: usize, i1: usize, j1: usize) -> Block<'a, &mut Self> {
        make_block_mut(self, i0, j0, i1, j1)
    }

    fn block_right_mut<'a>(&'a mut self, j: usize) -> Block<'a, &mut Self> {
        let rows = self.rows();
        let cols = self.cols();
        make_block_mut(self, 0, j, rows, cols)
    }

    fn block_bottom_mut<'a>(&'a mut self, i: usize) -> Block<'a, &mut Self> {
        let rows = self.rows();
        let cols = self.cols();
        make_block_mut(self, i, 0, rows, cols)
    }
}

fn check_same_size<T:MatBase, U:MatBase, M:fmt::Show>(a: &T, b: &U, text: &M) {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        panic!("Different sizes in {}: ({}, {}) vs ({}, {})",
               text, a.rows(), a.cols(), b.rows(), b.cols());
    }
}

#[derive(Clone)]
pub struct Mat {
    pub r: usize,
    pub c: usize,
    pub data: Vec<f32>  // Column major
}

impl Mat {
    pub fn new() -> Mat {
        Mat{r: 0, c: 0, data: Vec::new()}
    }

    pub fn zero(r: usize, c: usize) -> Mat {
        Mat{r: r, c: c,
            data: iter::repeat(0.0).take(r * c).collect()}
    }

    pub fn ident(n: usize) -> Mat {
        Mat{r: n, c: n,
            data: range(0, n*n).map(
                |i| if i % n == i / n { 1.0 } else { 0.0 }).collect()}
    }

    pub fn from_slice(r: usize, c: usize, rowmajor: &[f32]) -> Mat {
        // Data must be converted from row-major
        let mut data = Vec::with_capacity(r * c);
        for j in range(0, c) {
            for i in range(0, r) {
                data.push(rowmajor[i * c + j]);
            }
        }
        Mat{r: r, c: c, data: data}
    }

    pub fn from_fn<F: FnMut<(usize, usize), f32>>(r: usize, c: usize, mut op: F) -> Mat {
        let mut data = Vec::with_capacity(r * c);
        for j in range(0, c) {
            for i in range(0, r) {
                data.push(op(i, j));
            }
        }
        Mat{r: r, c: c, data: data}
    }

    pub fn ind(&self, i: usize, j: usize) -> usize {
        j * self.r + i
    }

    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.data[self.ind(i, j)]
    }

    pub fn at_mut<'a>(&'a mut self, i: usize, j: usize) -> &'a mut f32 {
        let i = self.ind(i, j);
        self.data.index_mut(&i)
    }

    pub fn set(&mut self, i: usize, j: usize, x: f32) {
        let i = self.ind(i, j);
        self.data[i] = x;
    }

    pub fn is_square(&self) -> bool {
        self.r == self.c
    }

    pub fn iter(&self) -> slice::Iter<f32> {
        self.data.iter()
    }

    pub fn lu(&self) -> LU {
        LU::of(self)
    }

    pub fn row_add(&mut self, src: usize, dst: usize, c: f32) {
        for j in range(0, self.c) {
            *self.at_mut(dst, j) += c * self.at(src, j);
        }
    }

    pub fn col_add(&mut self, src: usize, dst: usize, c: f32) {
        for i in range(0, self.r) {
            *self.at_mut(i, dst) += c * self.at(i, src);
        }
    }

    pub fn swap_row(&mut self, a: usize, b: usize) {
        for j in range(0, self.c) {
            unsafe {
                ptr::swap(self.at_mut(a, j), self.at_mut(b, j));
            }
        }
    }

    pub fn normalize(&mut self) {
        let n = self.norm();
        if n == 0.0 {
            panic!("Cannot normalize when norm is 0");
        }
        for el in self.data.iter_mut() {
            *el /= n;
        }
    }
}

impl MatBase for Mat {
    fn rows(&self) -> usize {
        self.r
    }

    fn cols(&self) -> usize {
        self.c
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn t<'a>(&'a self) -> Transposed<'a, &Mat> {
        Transposed{m: self}
    }

    fn norm(&self) -> f32 {
        num::Float::sqrt(sum_sq(self.iter().cloned()))
    }
}

impl MatBaseMut for Mat {
}

impl Fn<(usize, usize), f32> for Mat {
    extern "rust-call" fn call(&self, args: (usize, usize)) -> f32 {
        let (i, j) = args;
        self.data[self.ind(i, j)]
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f32;
    fn index(&self, &(i, j): &(usize, usize)) -> &f32 {
        self.data.index(&self.ind(i, j))
    }
}

impl IndexMut<(usize, usize)> for Mat {
    type Output = f32;
    fn index_mut<'a>(&'a mut self, &(i, j): &(usize, usize)) -> &'a mut f32 {
        let idx = self.ind(i, j);
        self.data.index_mut(&idx)
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

// TODO: It would be nice to use `add_assign()` to perform addition in
// place on MatBaseMut values.

// lhs + rhs
impl<LHS:MatBase, RHS:MatBase> Add<RHS> for LHS {
    type Output = Mat;
    fn add(self, rhs: RHS) -> Mat {
        check_same_size(&self, &rhs, &"Add");
        let mut m = Mat::zero(self.rows(), self.cols());
        for coor in self.coor_iter() {
            m[coor] = *self.index(&coor) + rhs[coor];
        }
        m
    }
}

// lhs + &rhs
impl<'b, LHS:MatBase, RHS:MatBase> Add<&'b RHS> for LHS {
    type Output = Mat;
    fn add(self, rhs: &RHS) -> Mat {
        check_same_size(&self, rhs, &"Add");
        let mut m = Mat::zero(self.rows(), self.cols());
        for coor in self.coor_iter() {
            m[coor] = *self.index(&coor) + rhs[coor];
        }
        m
    }
}

impl<LHS:MatBaseMut, RHS:MatBase> AddAssign<RHS> for LHS {
    fn add_assign(&mut self, rhs: RHS) {
        check_same_size(self, &rhs, &"AddAssign");
        for coor in self.coor_iter() {
            self[coor] += rhs[coor];
        }
    }
}

// lhs - rhs
impl<LHS:MatBase, RHS:MatBase> Sub<RHS> for LHS {
    type Output = Mat;
    fn sub(self, rhs: RHS) -> Mat {
        check_same_size(&self, &rhs, &"Sub");

        let mut m = Mat::zero(self.rows(), self.cols());
        for coor in self.coor_iter() {
            m[coor] = *self.index(&coor) - rhs[coor];
        }
        m
    }
}

// lhs - &rhs
impl<'b, LHS:MatBase, RHS:MatBase> Sub<&'b RHS> for LHS {
    type Output = Mat;
    fn sub(self, rhs: &RHS) -> Mat {
        check_same_size(&self, rhs, &"Sub");

        let mut m = Mat::zero(self.rows(), self.cols());
        for coor in self.coor_iter() {
            m[coor] = *self.index(&coor) - rhs[coor];
        }
        m
    }
}

impl<LHS:MatBaseMut, RHS:MatBase> SubAssign<RHS> for LHS {
    fn sub_assign(&mut self, rhs: RHS) {
        check_same_size(self, &rhs, &"SubAssign");
        for coor in self.coor_iter() {
            self[coor] -= rhs[coor];
        }
    }
}

// lhs * rhs
impl<LHS:MatBase, RHS:MatBase> Mul<RHS> for LHS {
    type Output = Mat;
    fn mul(self, rhs:RHS) -> Mat {
        (&self).mul(&rhs)
    }
}

// lhs * &rhs
impl<'b, LHS:MatBase, RHS:MatBase> Mul<&'b RHS> for LHS {
    type Output = Mat;
    fn mul(self, rhs:&RHS) -> Mat {
        (&self).mul(rhs)
    }
}

// &lhs * rhs
impl<'a, LHS:MatBase + 'a, RHS:MatBase> Mul<RHS> for &'a LHS {
    type Output = Mat;
    fn mul(self, rhs:RHS) -> Mat {
        self.mul(&rhs)
    }
}

// &lhs * &rhs
impl<'a, 'b, LHS:MatBase, RHS:MatBase> Mul<&'b RHS> for &'a LHS {
    type Output = Mat;
    fn mul(self, rhs: &RHS) -> Mat {
        Mat::from_fn(
            self.rows(), rhs.cols(),
            |i, j|
            range(0, self.cols()).fold(0.0, |x, k| x + self[(i, k)] * rhs[(k, j)]))
    }
}

// &lhs * &mut rhs
impl<'a, 'b, LHS:MatBase, RHS:MatBase> Mul<&'b mut RHS> for &'a LHS {
    type Output = Mat;
    fn mul(self, rhs: &mut RHS) -> Mat {
        self.mul(rhs)
    }
}

// lhs * scalar
impl<LHS:MatBase> Mul<f32> for LHS {
    type Output = Mat;
    fn mul(self, scalar: f32) -> Mat {
        Mat::from_fn(self.rows(), self.cols(),
                     |i, j| self[(i, j)] * scalar)
    }
}


// TODO: Would be nice to specify that T is `&MatBase` or `&mut MatBase`
pub struct Transposed<'a, T> {
    m: T
}

impl<'a, T:MatBase + 'a> Transposed<'a, &'a T> {}
impl<'a, T:MatBaseMut + 'a> Transposed<'a, &'a mut T> {}

macro_rules! transposed_matbase_impl {
    ($Base: ident, $Transposed:ty) => {
        impl<'a, T:$Base + 'a> MatBase for $Transposed {
            fn rows(&self) -> usize {
                self.m.cols()
            }

            fn cols(&self) -> usize {
                self.m.rows()
            }

            fn len(&self) -> usize {
                self.m.len()
            }

            fn norm(&self) -> f32 {
                self.m.norm()
            }

            fn t(&self) -> Transposed<&Self> {
                Transposed{m: self}
            }
        }
    }
}
transposed_matbase_impl!(MatBase, Transposed<'a, &'a T>);
transposed_matbase_impl!(MatBaseMut, Transposed<'a, &'a mut T>);

macro_rules! transposed_index_impl {
    ($Base:ident, $Block:ty) => {
        impl<'a, T:$Base> Index<(usize, usize)> for $Block {
            type Output = f32;
            //impl<'a, T:MatBase + 'a> Index<(usize, usize), f32> for Transposed<'a, &'a T> {
            fn index<'r>(&'r self, &(i, j): &(usize, usize)) -> &f32 {
                self.m.index(&(j, i))
            }
        }
    }
}
transposed_index_impl!(MatBase, Transposed<'a, &'a T>);
transposed_index_impl!(MatBaseMut, Transposed<'a, &'a mut T>);

impl<'a, T:MatBaseMut + 'a> IndexMut<(usize, usize)> for Transposed<'a, &'a mut T> {
    type Output = f32;
    fn index_mut<'r>(&'r mut self, &(i, j): &(usize, usize)) -> &'r mut f32 {
        self.m.index_mut(&(j, i))
    }
}



pub struct Block<'a, T> {
    m: T,
    i0: usize,
    j0: usize,
    i1: usize,
    j1: usize
}

fn make_block<'b, U: MatBase>(m: &'b U, i0: usize, j0: usize, i1: usize, j1: usize) -> Block<'b, &'b U> {
    if i0 >= i1 || i1 > m.rows() || j0 >= j1 || j1 > m.cols() {
        panic!("Invalid block ({}, {} to {}, {}) for {} x {}",
               i0, j0, i1, j1, m.rows(), m.cols());
    }
    Block{m: m, i0: i0, j0: j0, i1: i1, j1: j1}
}

fn make_block_mut<'b, U: MatBaseMut>(m: &'b mut U, i0: usize, j0: usize, i1: usize, j1: usize) -> Block<'b, &'b mut U> {
    if i0 >= i1 || i1 > m.rows() || j0 >= j1 || j1 > m.cols() {
        panic!("Invalid block ({}, {} to {}, {}) for {} x {}",
               i0, j0, i1, j1, m.rows(), m.cols());
    }
    Block{m: m, i0: i0, j0: j0, i1: i1, j1: j1}
}


// Block needs be implemented for `&MatBase` and for `&mut MatBase`.
// The impl can only implement one, so these shared methods need to be
// gathered into a trait.  `BlockTrait` contains all methods that
// would have been implemented in the impl.
pub trait BlockTrait : MatBase + Sized {
    fn iter<'b>(&'b self) -> BlockIterator<'b, &'b Self> {
        BlockIterator::new(self)
    }
}

// impl MatBase[Mut] for Block
macro_rules! block_impl {
    ($Base:ident, $Block:ty) => {
        impl<'a, T:$Base + 'a> BlockTrait for $Block {}
    }
}
block_impl!(MatBase, Block<'a, &'a T>);
block_impl!(MatBaseMut, Block<'a, &'a mut T>);

impl<'a, T:MatBaseMut + 'a> Block<'a, &'a mut T> {
}


impl<'a, T:MatBase> Index<usize> for Block<'a, &'a T> {
    type Output = f32;
    fn index(&self, &idx: &usize) -> &f32 {
        if self.rows() == 1 {
            self.index(&(0, idx))
        }
        else if self.cols() == 1 {
            self.index(&(idx, 0))
        }
        else {
            panic!("Single coord indexing only works for vector blocks");
        }
    }
}

//impl<'a, T:MatBase> Index<(usize, usize), f32> for Block<'a, &'a T> {
macro_rules! block_index_impl (
    ($Base:ident, $Block:ty) => (
        impl<'a, T:$Base> Index<(usize, usize)> for $Block {
            type Output = f32;
            fn index(&self, &(i, j): &(usize, usize)) -> &f32 {
                if i >= self.rows() || j >= self.cols() {
                    panic!("Index is outside the range of the block");
                }
                self.m.index(&(i + self.i0, j + self.j0))
            }
        }
    )
);
block_index_impl!(MatBase, Block<'a, &'a T>);
block_index_impl!(MatBaseMut, Block<'a, &'a mut T>);

impl<'a, T:MatBaseMut> IndexMut<(usize, usize)> for Block<'a, &'a mut T> {
    type Output = f32;
    fn index_mut(&mut self, &(i, j): &(usize, usize)) -> &mut f32 {
        if i >= self.rows() || j >= self.cols() {
            panic!("Index is outside the range of the block");
        }
        self.m.index_mut(&(i + self.i0, j + self.j0))
    }
}


// No longer just for blocks.  Now for all MatBase things
pub struct BlockIterator<'a, T: 'a> {
    m: T,
    idx: usize,
    marker: marker::ContravariantLifetime<'a>
}

impl<'a, T:MatBase> BlockIterator<'a, &'a T> {
    pub fn new<'r>(m: &'r T) -> BlockIterator<'r, &'r T> {
        BlockIterator{m: m, idx: 0, marker: marker::ContravariantLifetime::<'r>}
    }
}

impl<'a, T:MatBase> Iterator for BlockIterator<'a, &'a T> {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.idx == self.m.rows() * self.m.cols() {
            None
        }
        else {
            let i = self.idx % self.m.rows();
            let j = self.idx / self.m.rows();
            self.idx += 1;
            Some(*self.m.index(&(i, j)))
        }
    }
}

macro_rules! block_matbase_impl (
    ($Base: ident, $Block:ty) => (
        impl<'a, T:$Base + 'a> MatBase for $Block {
            fn rows(&self) -> usize {
                self.i1 - self.i0
            }

            fn cols(&self) -> usize {
                self.j1 - self.j0
            }

            fn len(&self) -> usize {
                self.rows() * self.cols()
            }

            fn norm<'b>(&'b self) -> f32 {
                num::Float::sqrt(sum_sq(self.iter()))
            }

            fn t(&self) -> Transposed<&Self> {
                Transposed{m: self}
            }
        }
    )
);
block_matbase_impl!(MatBase, Block<'a, &'a T>);
block_matbase_impl!(MatBaseMut, Block<'a, &'a mut T>);

impl<'a, T:MatBaseMut + 'a> MatBaseMut for Block<'a, &'a mut T> {}


fn show_matbase<T:MatBase>(mat: &T, f: &mut fmt::Formatter) -> fmt::Result {
    for i in range(0, mat.rows()) {
        for j in range(0, mat.cols()) {
            try!(write!(f, "{}  ", mat[(i, j)]))
        }
        try!(write!(f, "\n"))
    }
    write!(f, "")
}

impl fmt::Show for Mat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        show_matbase(self, f)
    }
}

macro_rules! impl_show_for_matbase {
    ($Base:ident, $Mat:ty) => {
        impl<'a, T:$Base + 'a> fmt::Show for $Mat {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                show_matbase(self, f)
            }
        }
    }
}
impl_show_for_matbase!(MatBase, Transposed<'a, &'a T>);
impl_show_for_matbase!(MatBaseMut, Transposed<'a, &'a mut T>);
impl_show_for_matbase!(MatBase, Block<'a, &'a T>);
impl_show_for_matbase!(MatBaseMut, Block<'a, &'a mut T>);


pub struct Permutation {
    v: Vec<usize>,
}

impl Permutation {
    pub fn ident(n: usize) -> Permutation {
        Permutation{v: range(0, n).collect()}
    }

    pub fn swap(n: usize, i: usize, j: usize) -> Permutation {
        Permutation{v: range(0, n).map(
            |k|
            if k == i { j }
            else if k == j { i }
            else { k }).collect()}
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    // Equavalent to Permutation::swap(i, j) * self
    pub fn swap_left(&mut self, i: usize, j: usize) {
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
        let mut v : Vec<usize> = iter::repeat(0).take(self.v.len()).collect();
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

impl Index<usize> for Permutation {
    type Output = usize;
    fn index<'a>(&'a self, _index: &usize) -> &'a usize {
        self.v.index(_index)
    }
}

impl Mul<Permutation> for Permutation {
    type Output = Permutation;
    fn mul(self, rhs: Permutation) -> Permutation {
        Permutation{v: range(0, self.len()).map(|i| rhs[self[i]]).collect()}
    }
}

impl Mul<Mat> for Permutation {
    type Output = Mat;
    fn mul(self, rhs: Mat) -> Mat {
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

impl<T:Clone> Mul<Vec<T>> for Permutation {
    type Output = Vec<T>;
    fn mul(self, rhs: Vec<T>) -> Vec<T> {
        if self.len() != rhs.len() {
            panic!("Permutation len doesn't match Vec len");
        }
        range(0, rhs.len()).map(|i| rhs[self[i]].clone()).collect()
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
fn test_transpose() {
    let a = Mat::from_slice(
        3, 4, &[2.0, 4.0, 6.0, 8.0,
                10.0, 12.0, 14.0, 16.0,
                18.0, 20.0, 22.0, 24.0]);

    let b = a.t();
    assert_eq!(b.rows(), a.cols());
    assert_eq!(b.cols(), a.rows());
    assert_mat_near!(b, Mat::from_slice(4, 3,
                                        &[2.0, 10.0, 18.0,
                                          4.0, 12.0, 20.0,
                                          6.0, 14.0, 22.0,
                                          8.0, 16.0, 24.0]),
                     0.00001);
}

#[test]
fn test_block() {
    let a = Mat::from_slice(
        3, 4, &[2.0, 4.0, 6.0, 8.0,
                10.0, 12.0, 14.0, 16.0,
                18.0, 20.0, 22.0, 24.0]);

    let b1 = a.block(0, 1, 2, 4);
    assert_mat_near!(b1, Mat::from_slice(2, 3,
                                         &[4.0, 6.0, 8.0,
                                           12.0, 14.0, 16.0]), 0.00001);
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
