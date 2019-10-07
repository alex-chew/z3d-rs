#![feature(box_patterns)]
#![feature(proc_macro_hygiene)]

extern crate proc_macro;

mod common;
mod dec;
mod exp;

#[proc_macro]
pub fn dec(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    dec::macro_main(input)
}

#[proc_macro]
pub fn exp(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    exp::macro_main(input)
}
