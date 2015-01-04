//#![feature(overloaded_calls)]
#![feature(unboxed_closures)]
#![allow(non_snake_case)]
#![allow(dead_code)]  // TODO: Remove eventually
#![feature(macro_rules)]

// They broke flexible ways of implementing operator traits.
//
// Broken by: https://github.com/rust-lang/rust/pull/20416
#![feature(old_orphan_check)]

#![crate_name = "rustmat"]

pub use base::{Mat, Transposed};
pub use lu::LU;

mod assert_macros;
mod base;
mod householder;
mod lu;
mod qr;
mod svd;
