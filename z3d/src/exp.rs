use proc_macro2::TokenStream;
use quote::{quote, quote_spanned, ToTokens};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;

use crate::common::{kw, punc, Sort};

trait ToExprTokens {
    fn to_expr_tokens(&self, ctx_ident: &syn::Ident) -> TokenStream;
}

/// Binary operators.
#[derive(Debug)]
pub enum BinOp {
    // Generic AST ops
    Equals(syn::Token![==]),

    // Bool ops
    And(syn::Token![&]),
    Or(syn::Token![|]),
    Xor(syn::Token![^]),
    Impl(syn::Token![->]),
    Iff(punc::Iff),

    // Int ops
    Add(syn::Token![+]),
    Sub(syn::Token![-]),
    Div(syn::Token![/]),
    Pow(punc::Pow),
    Mod(syn::Token![%]),
    Le(syn::Token![<=]),
    Ge(syn::Token![>=]),

    // Bitvec ops
    BvAnd(punc::BvAnd),
    BvOr(punc::BvOr),
    BvXor(punc::BvXor),
    BvShl(syn::Token![<<]),
    BvLogicShr(syn::Token![>>]),
    BvArithShr(punc::BvArithShr),

    // Int ops, but must be parsed later, since they are prefixes of others
    Mul(syn::Token![*]),
    Lt(syn::Token![<]),
    Gt(syn::Token![>]),
}

impl BinOp {
    fn ast_method(&self) -> TokenStream {
        let method_name = match self {
            BinOp::Equals(e) => quote_spanned! { e.spans[0] => ::z3::ast::Ast::_eq },

            BinOp::And(e) => quote_spanned! { e.spans[0] => ::z3::ast::Bool::and },
            BinOp::Or(e) => quote_spanned! { e.spans[0] => ::z3::ast::Bool::or },
            BinOp::Xor(e) => quote_spanned! { e.spans[0] => ::z3::ast::Bool::xor },
            BinOp::Impl(e) => quote_spanned! { e.spans[0] => ::z3::ast::Bool::implies },
            BinOp::Iff(e) => quote_spanned! { e.spans[0] => ::z3::ast::Bool::iff },

            BinOp::Add(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::add },
            BinOp::Sub(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::sub },
            BinOp::Div(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::div },
            BinOp::Pow(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::power },
            BinOp::Mod(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::modulo },
            BinOp::Le(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::le },
            BinOp::Ge(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::ge },

            BinOp::BvAnd(e) => quote_spanned! { e.spans[0] => ::z3::ast::BV::bvand },
            BinOp::BvOr(e) => quote_spanned! { e.spans[0] => ::z3::ast::BV::bvor },
            BinOp::BvXor(e) => quote_spanned! { e.spans[0] => ::z3::ast::BV::bvxor },
            BinOp::BvShl(e) => quote_spanned! { e.spans[0] => ::z3::ast::BV::bvshl },
            BinOp::BvLogicShr(e) => quote_spanned! { e.spans[0] => ::z3::ast::BV::bvlshr },
            BinOp::BvArithShr(e) => quote_spanned! { e.spans[0] => ::z3::ast::BV::bvashr },

            BinOp::Mul(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::mul },
            BinOp::Lt(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::lt },
            BinOp::Gt(e) => quote_spanned! { e.spans[0] => ::z3::ast::Int::gt },
        };
        method_name.into_token_stream()
    }
}

impl Parse for BinOp {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(syn::Token![&]) {
            Ok(BinOp::And(input.parse()?))
        } else if lookahead.peek(syn::Token![|]) {
            Ok(BinOp::Or(input.parse()?))
        } else if lookahead.peek(syn::Token![^]) {
            Ok(BinOp::Xor(input.parse()?))
        } else if lookahead.peek(syn::Token![->]) {
            Ok(BinOp::Impl(input.parse()?))
        } else if lookahead.peek(punc::Iff) {
            Ok(BinOp::Iff(input.parse()?))
        } else if lookahead.peek(syn::Token![+]) {
            Ok(BinOp::Add(input.parse()?))
        } else if lookahead.peek(syn::Token![-]) {
            Ok(BinOp::Sub(input.parse()?))
        } else if lookahead.peek(syn::Token![/]) {
            Ok(BinOp::Div(input.parse()?))
        } else if lookahead.peek(punc::Pow) {
            Ok(BinOp::Pow(input.parse()?))
        } else if lookahead.peek(syn::Token![%]) {
            Ok(BinOp::Mod(input.parse()?))
        } else if lookahead.peek(syn::Token![<=]) {
            Ok(BinOp::Le(input.parse()?))
        } else if lookahead.peek(syn::Token![>=]) {
            Ok(BinOp::Ge(input.parse()?))
        } else if lookahead.peek(syn::Token![==]) {
            Ok(BinOp::Equals(input.parse()?))
        } else if lookahead.peek(punc::BvAnd) {
            Ok(BinOp::BvAnd(input.parse()?))
        } else if lookahead.peek(punc::BvOr) {
            Ok(BinOp::BvOr(input.parse()?))
        } else if lookahead.peek(punc::BvXor) {
            Ok(BinOp::BvXor(input.parse()?))
        } else if lookahead.peek(syn::Token![<<]) {
            Ok(BinOp::BvShl(input.parse()?))
        } else if lookahead.peek(syn::Token![>>]) {
            Ok(BinOp::BvLogicShr(input.parse()?))
        } else if lookahead.peek(punc::BvArithShr) {
            Ok(BinOp::BvArithShr(input.parse()?))
        } else if lookahead.peek(syn::Token![*]) {
            Ok(BinOp::Mul(input.parse()?))
        } else if lookahead.peek(syn::Token![<]) {
            Ok(BinOp::Lt(input.parse()?))
        } else if lookahead.peek(syn::Token![>]) {
            Ok(BinOp::Gt(input.parse()?))
        } else {
            Err(lookahead.error())
        }
    }
}

/// Unary operators.
#[derive(Debug)]
pub enum UnOp {
    Neg(syn::Token![-]),
    Not(syn::Token![!]),
    BvNeg(punc::BvNeg),
    BvNot(punc::BvNot),
}

impl UnOp {
    fn ast_method(&self) -> TokenStream {
        let method_name = match self {
            UnOp::Neg(e) => syn::Ident::new("minus", e.span),
            UnOp::Not(e) => syn::Ident::new("not", e.span),
            UnOp::BvNeg(e) => syn::Ident::new("bvneg", e.spans[0]),
            UnOp::BvNot(e) => syn::Ident::new("bvnot", e.spans[0]),
        };
        method_name.into_token_stream()
    }
}

impl Parse for UnOp {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(syn::Token![-]) {
            Ok(UnOp::Neg(input.parse()?))
        } else if lookahead.peek(syn::Token![!]) {
            Ok(UnOp::Not(input.parse()?))
        } else if lookahead.peek(punc::BvNeg) {
            Ok(UnOp::BvNeg(input.parse()?))
        } else if lookahead.peek(punc::BvNot) {
            Ok(UnOp::BvNot(input.parse()?))
        } else {
            Err(lookahead.error())
        }
    }
}

/// Variadic operators.
#[derive(Debug)]
pub enum VarOp {
    And(kw::and),
    Or(kw::or),
    Add(kw::add),
    BvAnd(kw::bvand),
    BvOr(kw::bvor),
    Distinct(kw::distinct),
}

impl VarOp {
    fn ast_method(&self) -> TokenStream {
        match self {
            VarOp::And(e) => quote_spanned!(e.span => ::z3::ast::Bool::and),
            VarOp::Or(e) => quote_spanned!(e.span => ::z3::ast::Bool::or),
            VarOp::Add(e) => quote_spanned!(e.span => ::z3::ast::Int::add),

            // These are manually expanded using method call syntax, so we
            // can't use absolute paths to refer to the methods
            VarOp::BvAnd(e) => quote_spanned!(e.span => bvand),
            VarOp::BvOr(e) => quote_spanned!(e.span => bvor),

            VarOp::Distinct(e) => quote_spanned!(e.span => ::z3::ast::Ast::distinct),
        }
    }
}

impl Parse for VarOp {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(kw::and) {
            Ok(VarOp::And(input.parse()?))
        } else if lookahead.peek(kw::or) {
            Ok(VarOp::Or(input.parse()?))
        } else if lookahead.peek(kw::add) {
            Ok(VarOp::Add(input.parse()?))
        } else if lookahead.peek(kw::bvand) {
            Ok(VarOp::BvAnd(input.parse()?))
        } else if lookahead.peek(kw::bvor) {
            Ok(VarOp::BvOr(input.parse()?))
        } else if lookahead.peek(kw::distinct) {
            Ok(VarOp::Distinct(input.parse()?))
        } else {
            Err(lookahead.error())
        }
    }
}

#[derive(Debug)]
pub enum Atom {
    Bool(syn::LitBool),
    Int(syn::LitInt),
    Ident(syn::Ident),
}

impl Parse for Atom {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(syn::LitBool) {
            Ok(Atom::Bool(input.parse()?))
        } else if lookahead.peek(syn::LitInt) {
            Ok(Atom::Int(input.parse()?))
        } else if lookahead.peek(syn::Ident) {
            Ok(Atom::Ident(input.parse()?))
        } else {
            Err(lookahead.error())
        }
    }
}

impl ToExprTokens for Atom {
    fn to_expr_tokens(&self, ctx: &syn::Ident) -> TokenStream {
        match self {
            Atom::Bool(lit) => quote! { ::z3::ast::Bool::from_bool(#ctx, #lit) },
            Atom::Int(lit) => quote! { ::z3::ast::Int::from_i64(#ctx, #lit as i64) },
            Atom::Ident(ident) => quote! { #ident },
        }
    }
}

#[derive(Debug)]
pub enum AtomCastableToBitvec {
    Int(syn::LitInt),
    Ident(syn::Ident),
}

impl Parse for AtomCastableToBitvec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(syn::Ident) {
            Ok(AtomCastableToBitvec::Ident(input.parse().unwrap()))
        } else if lookahead.peek(syn::LitInt) {
            Ok(AtomCastableToBitvec::Int(input.parse().unwrap()))
        } else {
            Err(input.error("only integer literals and identifiers may be cast to bitvec"))
        }
    }
}

/// A parenthesized DSL expression.
#[derive(Debug)]
pub struct ParenExpr {
    paren: syn::token::Paren,
    inner: Box<Expr>,
}

impl Parse for ParenExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        let paren = syn::parenthesized!(content in input);
        let inner = content.parse()?;
        Ok(ParenExpr {
            paren,
            inner: Box::new(inner),
        })
    }
}

impl ToExprTokens for ParenExpr {
    fn to_expr_tokens(&self, ctx_ident: &syn::Ident) -> TokenStream {
        let inner_tokens = self.inner.to_expr_tokens(ctx_ident);
        quote! { ( #inner_tokens ) }
    }
}

/// A braced Rust expression of type z3::Ast.
#[derive(Debug)]
pub struct RustExpr {
    brace: syn::token::Brace,
    inner: Box<syn::Expr>,
}

impl Parse for RustExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        let brace = syn::braced!(content in input);
        let inner = content.parse()?;
        Ok(RustExpr {
            brace,
            inner: Box::new(inner),
        })
    }
}

impl ToExprTokens for RustExpr {
    fn to_expr_tokens(&self, _ctx_ident: &syn::Ident) -> TokenStream {
        let RustExpr { box inner, .. } = self;
        quote_spanned! { inner.span() => ( #inner ) }
    }
}

/// An expression operand which is visually unambiguous (in terms of precendence).
#[derive(Debug)]
pub enum SimpleOperand {
    Atom(Atom),
    Paren(ParenExpr),
    Rust(RustExpr),
    Unary(UnaryExpr),
    Variadic(VariadicExpr),
}

impl Parse for SimpleOperand {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.fork().parse::<ParenExpr>().is_ok() {
            Ok(Self::Paren(input.parse()?))
        } else if input.fork().parse::<RustExpr>().is_ok() {
            Ok(Self::Rust(input.parse()?))
        } else if input.fork().parse::<UnaryExpr>().is_ok() {
            Ok(Self::Unary(input.parse()?))
        } else if input.fork().parse::<VariadicExpr>().is_ok() {
            Ok(Self::Variadic(input.parse()?))
        } else if input.fork().parse::<Atom>().is_ok() {
            Ok(Self::Atom(input.parse()?))
        } else {
            Err(input.error("invalid simple operand"))
        }
    }
}

impl ToExprTokens for SimpleOperand {
    fn to_expr_tokens(&self, ctx_ident: &syn::Ident) -> TokenStream {
        match self {
            Self::Atom(e) => e.to_expr_tokens(ctx_ident),
            Self::Paren(e) => e.to_expr_tokens(ctx_ident),
            Self::Rust(e) => e.to_expr_tokens(ctx_ident),
            Self::Unary(e) => e.to_expr_tokens(ctx_ident),
            Self::Variadic(e) => e.to_expr_tokens(ctx_ident),
        }
    }
}

#[derive(Debug)]
pub struct BinaryExpr {
    left: Box<SimpleOperand>,
    op: BinOp,
    right: Box<SimpleOperand>,
}

impl Parse for BinaryExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let left = input.parse()?;
        let op = input.parse()?;
        let right = input.parse()?;
        Ok(BinaryExpr {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }
}

impl ToExprTokens for BinaryExpr {
    fn to_expr_tokens(&self, ctx_ident: &syn::Ident) -> TokenStream {
        let left_tokens = self.left.to_expr_tokens(ctx_ident);
        let op_method = self.op.ast_method();
        let right_tokens = self.right.to_expr_tokens(ctx_ident);
        let is_variadic_op = match self.op {
            BinOp::And(_) | BinOp::Or(_) | BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) => true,
            _ => false,
        };
        // Some of these methods take slices, representing variadic operators
        let wrapped_right_tokens = if is_variadic_op {
            quote! { [&#right_tokens] }
        } else {
            right_tokens
        };

        quote! {
            #op_method(&#left_tokens, &#wrapped_right_tokens)
        }
    }
}

#[derive(Debug)]
pub struct UnaryExpr {
    op: UnOp,
    right: Box<SimpleOperand>,
}

impl Parse for UnaryExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let op = input.parse()?;
        let right = input.parse()?;
        Ok(UnaryExpr {
            op,
            right: Box::new(right),
        })
    }
}

impl ToExprTokens for UnaryExpr {
    fn to_expr_tokens(&self, ctx_ident: &syn::Ident) -> TokenStream {
        let op_method = self.op.ast_method();
        let right_tokens = self.right.to_expr_tokens(ctx_ident);
        quote! { #right_tokens.#op_method() }
    }
}

#[derive(Debug)]
pub struct VariadicExpr {
    op: VarOp,
    first: Box<Expr>,
    second: Box<Expr>,
    rest: Vec<Expr>,
}

impl Parse for VariadicExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let op = input.parse()?;
        let content;
        let _paren = syn::parenthesized!(content in input);
        let args = syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_separated_nonempty(
            &content,
        )?;
        if args.len() < 2 {
            return Err(input.error("variadic operator requires at least 2 arguments"));
        }

        let mut args = args.into_iter();
        let first = args.next().unwrap();
        let second = args.next().unwrap();
        let rest = args.collect();
        return Ok(VariadicExpr {
            op,
            first: Box::new(first),
            second: Box::new(second),
            rest,
        });
    }
}

impl ToExprTokens for VariadicExpr {
    fn to_expr_tokens(&self, ctx_ident: &syn::Ident) -> TokenStream {
        let first_tokens = self.first.to_expr_tokens(ctx_ident);
        let second_tokens = self.second.to_expr_tokens(ctx_ident);
        let rest_tokens: Vec<TokenStream> = self
            .rest
            .iter()
            .map(|operand| operand.to_expr_tokens(ctx_ident))
            .collect();
        let method = self.op.ast_method();

        match self.op {
            VarOp::And(_) | VarOp::Or(_) | VarOp::Add(_) | VarOp::Distinct(_) => {
                quote! {
                    #method(&#first_tokens, &[&#second_tokens #(, &#rest_tokens)*])
                }
            }
            VarOp::BvAnd(_) | VarOp::BvOr(_) => {
                let method_reps = std::iter::repeat(&method);
                quote! {
                    #first_tokens.#method(&#second_tokens)#(.#method_reps(&#rest_tokens))*
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct AsBitvecExpr {
    val: AtomCastableToBitvec,
    length: syn::LitInt,
}

impl Parse for AsBitvecExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let val = input.parse()?;
        input.parse::<syn::Token![as]>()?;

        let length = match input.parse()? {
            Sort::Bitvec { length } => Ok(length),
            _ => Err(input.error("expected bitvec sort")),
        }?;
        Ok(AsBitvecExpr { val, length })
    }
}

impl ToExprTokens for AsBitvecExpr {
    fn to_expr_tokens(&self, ctx_ident: &syn::Ident) -> TokenStream {
        let val_tokens = match &self.val {
            AtomCastableToBitvec::Int(e) => e.into_token_stream(),
            AtomCastableToBitvec::Ident(e) => e.into_token_stream(),
        };
        let length_tokens = &self.length;
        quote! { ::z3::ast::BV::from_i64(#ctx_ident, #val_tokens, #length_tokens) }
    }
}

#[derive(Debug)]
pub enum Expr {
    Atom(Atom),
    Paren(ParenExpr),
    Rust(RustExpr),
    Unary(UnaryExpr),
    Variadic(VariadicExpr),
    Binary(BinaryExpr),
    AsBitvec(AsBitvecExpr),
}

impl Parse for Expr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.fork().parse::<BinaryExpr>().is_ok() {
            Ok(Expr::Binary(input.parse()?))
        } else if input.fork().parse::<AsBitvecExpr>().is_ok() {
            Ok(Expr::AsBitvec(input.parse()?))
        } else if input.fork().parse::<ParenExpr>().is_ok() {
            Ok(Expr::Paren(input.parse()?))
        } else if input.fork().parse::<RustExpr>().is_ok() {
            Ok(Expr::Rust(input.parse()?))
        } else if input.fork().parse::<UnaryExpr>().is_ok() {
            Ok(Expr::Unary(input.parse()?))
        } else if input.fork().parse::<VariadicExpr>().is_ok() {
            Ok(Expr::Variadic(input.parse()?))
        } else if input.fork().parse::<Atom>().is_ok() {
            Ok(Expr::Atom(input.parse()?))
        } else {
            Err(input.error("invalid expression"))
        }
    }
}

impl ToExprTokens for Expr {
    fn to_expr_tokens(&self, ctx_ident: &syn::Ident) -> TokenStream {
        match self {
            Self::Atom(e) => e.to_expr_tokens(ctx_ident),
            Self::Paren(e) => e.to_expr_tokens(ctx_ident),
            Self::Rust(e) => e.to_expr_tokens(ctx_ident),
            Self::Unary(e) => e.to_expr_tokens(ctx_ident),
            Self::Variadic(e) => e.to_expr_tokens(ctx_ident),
            Self::Binary(e) => e.to_expr_tokens(ctx_ident),
            Self::AsBitvec(e) => e.to_expr_tokens(ctx_ident),
        }
    }
}

struct Input {
    expr: Expr,
    ctx_ident: syn::Ident,
}

impl Parse for Input {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let expr = input.parse()?;
        input.parse::<syn::Token![in]>()?;
        let ctx_ident = input.parse()?;
        Ok(Self { expr, ctx_ident })
    }
}

/// DSL expressions like the following:
///
/// ```ignore
/// // ctx is a z3::Context, and solver is a z3::Solver
/// let x = dec!(x: int in ctx);
/// let b1 = dec!(b1: bool in ctx);
/// let b2 = dec!(b2: bool in ctx);
/// let v = dec!(v: bitvec<16> in ctx);
/// let constraint = exp!((b1 <-> b2) ^ (-x > (x / 4)) in ctx);
/// solver.assert(constraint);
/// ```
pub fn macro_main(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let Input { ctx_ident, expr } = syn::parse_macro_input!(input as Input);
    expr.to_expr_tokens(&ctx_ident).into()
}

#[cfg(test)]
mod expr_parse_tests {
    use super::*;
    use assert_matches::assert_matches;

    #[test]
    fn int_atom() {
        let parsed: Atom = syn::parse_str("3").unwrap();
        match parsed {
            Atom::Int(lit) => assert_eq!(lit.base10_digits(), "3"),
            _ => panic!(),
        }
    }

    #[test]
    fn bool_atom() {
        let parsed: Atom = syn::parse_str("false").unwrap();
        match parsed {
            Atom::Bool(lit) => assert_eq!(lit.value, false),
            _ => panic!(),
        }
    }

    #[test]
    fn ident_atom() {
        let parsed: Atom = syn::parse_str("something").unwrap();
        match parsed {
            Atom::Ident(ident) => assert_eq!(ident.to_string(), "something"),
            _ => panic!(),
        }
    }

    #[test]
    fn atom_operand() {
        let parsed: SimpleOperand = syn::parse_str("something").unwrap();
        match parsed {
            SimpleOperand::Atom(Atom::Ident(ident)) => assert_eq!(ident.to_string(), "something"),
            _ => panic!(),
        }
    }

    #[test]
    fn paren_atom_operand() {
        let parsed: SimpleOperand = syn::parse_str("(88)").unwrap();
        assert_matches!(parsed, SimpleOperand::Paren(ParenExpr { inner, .. }) => {
            assert_matches!(*inner,
                            Expr::Atom(Atom::Int(ref lit))
                            if lit.base10_digits() == "88");
        });
    }

    #[test]
    fn misc_binary_and_unary() {
        let parsed: Expr = syn::parse_str("!(false == p) <-> (4 / -i)").unwrap();
        let (left, op, right) = assert_matches!(parsed, Expr::Binary(BinaryExpr { left, op, right }) => (left, op, right));
        assert_matches!(op, BinOp::Iff(_));

        // Check left expression
        {
            let (op, right) = assert_matches!(
                left,
                box SimpleOperand::Unary(UnaryExpr { op, right }) => (op, right));
            assert_matches!(op, UnOp::Not(_));

            let (left, op, right) = assert_matches!(
                right,
                box SimpleOperand::Paren(ParenExpr {
                    inner: box Expr::Binary(BinaryExpr { left, op, right }),
                    ..
                }) => (left, op, right));
            assert_matches!(op, BinOp::Equals(_));
            assert_matches!(
                left,
                box SimpleOperand::Atom(Atom::Bool(ref lit)) if lit.value == false);
            assert_matches!(
                right,
                box SimpleOperand::Atom(Atom::Ident(ref ident)) if ident.to_string() == "p");
        }

        // Check right expression
        {
            let (left, op, right) = assert_matches!(
                right,
                box SimpleOperand::Paren(ParenExpr {
                    inner: box Expr::Binary(BinaryExpr { left, op, right }),
                    ..
                }) => (left, op, right));
            assert_matches!(op, BinOp::Div(_));

            assert_matches!(
                left,
                box SimpleOperand::Atom(Atom::Int(ref lit)) if lit.base10_digits() == "4");

            let (op, right) = assert_matches!(
                right,
                box SimpleOperand::Unary(UnaryExpr { op, right }) => (op, right));
            assert_matches!(op, UnOp::Neg(_));
            assert_matches!(
                right,
                box SimpleOperand::Atom(Atom::Ident(ref ident)) if ident.to_string() == "i");
        }
    }

    #[test]
    fn varop_expr() {
        let parsed: Expr =
            syn::parse_str("bvor(arg1, (0b101010 as bitvec<16>), arg3, {etc[2]})").unwrap();

        let (op, first, second, rest) = assert_matches!(
            parsed,
            Expr::Variadic(VariadicExpr { op, first, second, rest }) => (op, first, second, rest));
        assert_matches!(op, VarOp::BvOr(_));
        assert_eq!(rest.len(), 2);

        assert_matches!(
            &first,
            box Expr::Atom(Atom::Ident(ref ident)) if ident.to_string() == "arg1");

        let (val, length) = assert_matches!(
            &second,
            box Expr::Paren(ParenExpr { inner: box Expr::AsBitvec(AsBitvecExpr { val, length }), .. }) => (val, length));
        assert_matches!(
            val,
            AtomCastableToBitvec::Int(lit) if lit.base10_digits() == format!("{}", 0b101010));
        assert_eq!(length.base10_digits(), "16");

        assert_matches!(
            &rest[0],
            Expr::Atom(Atom::Ident(ref ident)) if ident.to_string() == "arg3");

        assert_matches!(
            &rest[1],
            Expr::Rust(RustExpr { inner: box e, .. })
            if *e == syn::parse_str("etc[2]").unwrap());
    }

    #[test]
    fn add_series() {
        let parsed: Expr = syn::parse_str("add(1 * 100, 2 * 10, 3) == 123").unwrap();

        let (add_expr, equals, total) = assert_matches!(
            parsed,
            Expr::Binary(BinaryExpr { left, op, right }) => (left, op, right)
        );
        assert_matches!(equals, BinOp::Equals(_));

        let var_expr = assert_matches!(add_expr, box SimpleOperand::Variadic(e) => e);
        assert_matches!(var_expr.op, VarOp::Add(_));
        assert_matches!(
            var_expr.first,
            box Expr::Binary(BinaryExpr {
                left: box SimpleOperand::Atom(Atom::Int(_)),
                op: BinOp::Mul(_),
                right: box SimpleOperand::Atom(Atom::Int(_)),
            })
        );
        assert_matches!(
            var_expr.second,
            box Expr::Binary(BinaryExpr {
                left: box SimpleOperand::Atom(Atom::Int(_)),
                op: BinOp::Mul(_),
                right: box SimpleOperand::Atom(Atom::Int(_)),
            })
        );
        assert_eq!(var_expr.rest.len(), 1);
        assert_matches!(var_expr.rest[0], Expr::Atom(Atom::Int(_)));

        assert_matches!(
            total,
            box SimpleOperand::Atom(Atom::Int(i)) if i.base10_digits() == "123"
        );
    }

    #[test]
    fn many_omitted_parens() {
        let parsed: Expr = syn::parse_str("--(.~. .!. some_bitvec + -!-{x})").unwrap();

        let in_parens = assert_matches!(
            parsed,
            Expr::Unary(UnaryExpr {
                op: UnOp::Neg(_),
                right: box SimpleOperand::Unary(UnaryExpr {
                    op: UnOp::Neg(_),
                    right: box SimpleOperand::Paren(ParenExpr { inner, .. }),
                })
            }) => inner
        );

        let (bitvec_expr, x_expr) = assert_matches!(
            in_parens,
            box Expr::Binary(BinaryExpr { left, op: BinOp::Add(_), right })
            => (left, right)
        );

        assert_matches!(
            bitvec_expr,
            box SimpleOperand::Unary(UnaryExpr {
                op: UnOp::BvNeg(_),
                right: box SimpleOperand::Unary(UnaryExpr {
                    op: UnOp::BvNot(_),
                    right: box SimpleOperand::Atom(Atom::Ident(ident)),
                })
            })
            if ident.to_string() == "some_bitvec"
        );

        assert_matches!(
            x_expr,
            box SimpleOperand::Unary(UnaryExpr {
                op: UnOp::Neg(_),
                right: box SimpleOperand::Unary(UnaryExpr {
                    op: UnOp::Not(_),
                    right: box SimpleOperand::Unary(UnaryExpr {
                        op: UnOp::Neg(_),
                        right: box SimpleOperand::Rust(RustExpr { box inner, ..  }),
                    })
                })
            })
            if inner.to_token_stream().to_string() == "x"
        );
    }
}

#[cfg(test)]
mod to_expr_tokens_tests {
    use std::str::FromStr;

    use super::*;

    fn assert_tokens_match(tokens: &TokenStream, target: &str) {
        let tokens_str = tokens.to_string();
        let target_str = TokenStream::from_str(target).unwrap().to_string();
        assert_eq!(tokens_str, target_str);
    }

    fn default_ctx_ident() -> syn::Ident {
        syn::Ident::new("ctx", proc_macro2::Span::call_site())
    }

    #[test]
    fn cast_int_to_bitvec() {
        let val = AtomCastableToBitvec::Int(syn::parse_str("0b11_1110").unwrap());
        let length = syn::parse_str("7").unwrap();
        let tokens = AsBitvecExpr { val, length }.to_expr_tokens(&default_ctx_ident());
        assert_tokens_match(&tokens, "::z3::ast::BV::from_i64(ctx, 0b11_1110, 7)");
    }

    #[test]
    fn cast_ident_to_bitvec() {
        let val = AtomCastableToBitvec::Ident(syn::parse_str("num").unwrap());
        let length = syn::parse_str("16").unwrap();
        let tokens = AsBitvecExpr { val, length }.to_expr_tokens(&default_ctx_ident());
        assert_tokens_match(&tokens, "::z3::ast::BV::from_i64(ctx, num, 16)");
    }

    #[test]
    fn distinct() {
        let op = syn::parse_str("distinct").unwrap();
        let first = syn::parse_str("arg1").unwrap();
        let second = syn::parse_str("arg2").unwrap();
        let rest: Vec<Expr> = ["arg3", "arg4", "arg5"]
            .into_iter()
            .map(|s| syn::parse_str(s).unwrap())
            .collect();
        let tokens = VariadicExpr {
            op,
            first,
            second,
            rest,
        }
        .to_expr_tokens(&default_ctx_ident());
        assert_tokens_match(
            &tokens,
            "::z3::ast::Ast::distinct(&arg1, &[&arg2, &arg3, &arg4, &arg5])",
        );
    }

    #[test]
    fn range() {
        let left = syn::parse_str("(num >= 0)").unwrap();
        let op = syn::parse_str("&").unwrap();
        let right = syn::parse_str("(num <= 9)").unwrap();
        let tokens = BinaryExpr { left, op, right }.to_expr_tokens(&default_ctx_ident());
        assert_tokens_match(
            &tokens,
            "::z3::ast::Bool::and(
                &(::z3::ast::Int::ge(&num, & ::z3::ast::Int::from_i64(ctx, 0 as i64))),
                &[
                    &(::z3::ast::Int::le(&num, & ::z3::ast::Int::from_i64(ctx, 9 as i64)))
                ]
            )",
        );
    }

    #[test]
    fn variadic_bvor() {
        let op = syn::parse_str("bvor").unwrap();
        let first = syn::parse_str("arg1").unwrap();
        let second = syn::parse_str("arg2").unwrap();
        let rest: Vec<Expr> = ["arg3", "arg4", "arg5"]
            .into_iter()
            .map(|s| syn::parse_str(s).unwrap())
            .collect();
        let tokens = VariadicExpr {
            op,
            first,
            second,
            rest,
        }
        .to_expr_tokens(&default_ctx_ident());
        assert_tokens_match(
            &tokens,
            "arg1.bvor(&arg2).bvor(&arg3).bvor(&arg4).bvor(&arg5)",
        );
    }
}
