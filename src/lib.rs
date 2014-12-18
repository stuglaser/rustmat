//#![feature(overloaded_calls)]
#![feature(unboxed_closures)]
#![allow(non_snake_case)]
#![allow(dead_code)]  // TODO: Remove eventually
#![feature(macro_rules)]

#![crate_name = "rustmat"]

pub use base::{Mat, Transposed};
pub use lu::LU;

mod assert_macros;
mod base;
mod lu;
mod qr;
