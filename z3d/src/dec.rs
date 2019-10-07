use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::parse::Parse;

use crate::common::Sort;

type FormatArgs = syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>;

#[derive(PartialEq, Debug)]
struct FormattedName {
    format_string: syn::LitStr,
    format_args: FormatArgs,
}

impl Parse for FormattedName {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _dollar: syn::Token![$] = input.parse()?;
        let content;
        let _paren = syn::parenthesized!(content in input);

        let format_string: syn::LitStr = content.parse()?;
        let lookahead = content.lookahead1();
        let format_args = if lookahead.peek(syn::Token![,]) {
            content.parse::<syn::Token![,]>()?;
            content.parse_terminated(syn::Expr::parse)?
        } else {
            syn::punctuated::Punctuated::new()
        };

        Ok(FormattedName {
            format_string,
            format_args,
        })
    }
}

#[derive(PartialEq, Debug)]
enum ConstName {
    Plain(syn::Ident),
    Formatted(FormattedName),
}

impl ConstName {
    fn into_string_expr_tokens(&self) -> TokenStream {
        match self {
            ConstName::Plain(ident) => {
                syn::LitStr::new(&ident.to_string(), ident.span()).into_token_stream()
            }
            ConstName::Formatted(FormattedName {
                format_string,
                format_args,
            }) => {
                quote! {
                    ::std::format!(#format_string, #format_args)
                }
            }
        }
    }
}

impl Parse for ConstName {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(syn::Ident) {
            Ok(ConstName::Plain(input.parse()?))
        } else {
            Ok(ConstName::Formatted(input.parse()?))
        }
    }
}

#[derive(PartialEq, Debug)]
struct ConstDecl {
    sort: Sort,
    name: ConstName,
}

impl Parse for ConstDecl {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<syn::Token![:]>()?;
        let sort = input.parse()?;
        Ok(ConstDecl { sort, name })
    }
}

struct ConstDeclInput {
    ctx: syn::Ident,
    decl: ConstDecl,
}

impl Parse for ConstDeclInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let decl = input.parse()?;
        input.parse::<syn::Token![in]>()?;
        let ctx = input.parse()?;
        Ok(Self { ctx, decl })
    }
}

/// Constant declarations of the following forms:
///
/// ```ignore
/// let ctx = &z3::Context::new(z3::Config::default());
/// let x = dec!(x: int in ctx);            // integer constant
/// let b = dec!(b: bool in ctx);           // boolean constant
/// let v = dec!(v: bitvec<16> in ctx);     // 16-bit bitvector constant
/// let ys = (1..=3).map(|i| dec!($("c_{}_{}", 0, i): int in ctx));   // formatted names
/// ```
pub fn macro_main(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ConstDeclInput {
        ctx,
        decl: ConstDecl { sort, name },
    } = syn::parse_macro_input!(input as ConstDeclInput);
    let name_tokens = name.into_string_expr_tokens();
    (match sort {
        Sort::Int => quote! { ::z3::ast::Int::new_const(#ctx, #name_tokens) },
        Sort::Bool => quote! { ::z3::ast::Bool::new_const(#ctx, #name_tokens) },
        Sort::Bitvec { length } => quote! { ::z3::ast::BV::new_const(#ctx, #name_tokens, #length) },
    })
    .into()
}

#[cfg(test)]
mod const_decl_parse_tests {
    use super::*;
    use proc_macro2::Span;
    use syn::parse::Parser;

    #[test]
    fn simple_int() {
        let decl: ConstDecl = syn::parse_str("something: int").unwrap();
        let expected_name = ConstName::Plain(syn::Ident::new("something", Span::call_site()));
        assert_eq!(
            decl,
            ConstDecl {
                sort: Sort::Int,
                name: expected_name
            }
        );
    }

    #[test]
    fn simple_bool() {
        let decl: ConstDecl = syn::parse_str("flag: bool").unwrap();
        let expected_name = ConstName::Plain(syn::Ident::new("flag", Span::call_site()));
        assert_eq!(
            decl,
            ConstDecl {
                sort: Sort::Bool,
                name: expected_name
            }
        );
    }

    #[test]
    fn simple_bitvec() {
        let decl: ConstDecl = syn::parse_str("word: bitvec<32>").unwrap();
        let expected_sort = Sort::Bitvec {
            length: syn::LitInt::new("32", Span::call_site()),
        };
        let expected_name = ConstName::Plain(syn::Ident::new("word", Span::call_site()));
        assert_eq!(
            decl,
            ConstDecl {
                sort: expected_sort,
                name: expected_name
            }
        );
    }

    #[test]
    fn lit_formatted_name() {
        let decl: ConstDecl = syn::parse_str(r#"$("x{}", 1): int"#).unwrap();
        let expected_name = ConstName::Formatted(FormattedName {
            format_string: syn::parse_str(r#""x{}""#).unwrap(),
            format_args: FormatArgs::parse_terminated.parse_str(r#"1"#).unwrap(),
        });
        assert_eq!(decl.name, expected_name);
    }

    #[test]
    fn misc_formatted_name() {
        let decl: ConstDecl = syn::parse_str(r#"$("x_{}_{}_{}", 1, i, (5 + 6)): int"#).unwrap();
        let expected_name = ConstName::Formatted(FormattedName {
            format_string: syn::parse_str(r#""x_{}_{}_{}""#).unwrap(),
            format_args: FormatArgs::parse_terminated
                .parse_str(r#"1, i, (5 + 6)"#)
                .unwrap(),
        });
        assert_eq!(decl.name, expected_name);
    }

    #[test]
    fn bitvec_with_formatted_name() {
        let decl: ConstDecl = syn::parse_str(r#"$("cell{}{}", 1, (3 - 1)): bitvec<8>"#).unwrap();
        let expected_sort = Sort::Bitvec {
            length: syn::LitInt::new("8", Span::call_site()),
        };
        let expected_name = ConstName::Formatted(FormattedName {
            format_string: syn::parse_str(r#""cell{}{}""#).unwrap(),
            format_args: FormatArgs::parse_terminated
                .parse_str(r#"1, (3 - 1)"#)
                .unwrap(),
        });
        assert_eq!(
            decl,
            ConstDecl {
                sort: expected_sort,
                name: expected_name
            }
        );
    }
}
