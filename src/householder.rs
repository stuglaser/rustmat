use base::{MatBase, MatBaseMut, Mat, SubAssign};

pub struct Householder {
    v: Mat,  // Reflection vector.  H = I - 2*v*v'
}

impl Householder {
    pub fn new<T:MatBase>(v: &T) -> Householder {
        Householder::new_move(v.resolved())
    }

    pub fn new_move(v: Mat) -> Householder {
        // Moves the vector, using no extra storage
        if v.cols() != 1 {
            panic!("Householder must be created from a column vector");
        }
        Householder{v: v}
    }

    // Returns the actual Householder matrix
    pub fn resolve(&self) -> Mat {
        let mut m = Mat::ident(self.v.len());
        self.lapply(&mut m);
        m
    }

    // Performs m := H * m
    pub fn lapply<T:MatBaseMut>(&self, m: &mut T) {
        if self.v.rows() != m.rows() {
            panic!("Wrong dims {}x{} for lapply Householder of {}",
                   m.rows(), m.cols(), self.v.rows());
        }

        // (I - 2*v*v') * m = m - 2 * v * (v' * m)
        let tmp = &self.v.t() * &(*m);  // Force to an immutable borrow
        m.sub_assign(&self.v * tmp * 2.0);
    }

    // Performs m := m * H
    pub fn rapply<T:MatBaseMut>(&self, m: &mut T) {
        if m.cols() != self.v.rows() {
            panic!("Wrong dims {}x{} for rapply Householder of {}",
                   m.rows(), m.cols(), self.v.rows());
        }

        // m * (I - 2*v*v') = m - 2 * (m * v) * v'
        let tmp = &(*m) * &self.v;
        m.sub_assign(tmp * &self.v.t() * 2.0);
    }
}

// Creates a Householder reflection that reflects the given (column)
// vector, x, to [c, 0, ..., 0]
pub fn reflector_to_e1<T:MatBase>(x: &T) -> Householder {
    if !x.is_column() {
        panic!("Can only reflect along (column) vectors:\n{}", x);
    }
    
    let norm = x.norm();
    let mut v = x.resolved();
    v[(0, 0)] -= norm;  // v = x - |x| * e1

    if !v.is_zero() {
        v.normalize();
    }
    
    Householder::new_move(v)
}
