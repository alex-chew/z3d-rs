use syn::parse::Parse;

pub mod kw {
    syn::custom_keyword!(int);
    syn::custom_keyword!(bool);
    syn::custom_keyword!(bitvec);
    syn::custom_keyword!(and);
    syn::custom_keyword!(or);
    syn::custom_keyword!(add);
    syn::custom_keyword!(bvand);
    syn::custom_keyword!(bvor);
    syn::custom_keyword!(distinct);
}

pub mod punc {
    syn::custom_punctuation!(Iff, <->);
    syn::custom_punctuation!(Pow, **);
    syn::custom_punctuation!(BvAnd, .&.);
    syn::custom_punctuation!(BvOr, .|.);
    syn::custom_punctuation!(BvXor, .^.);
    syn::custom_punctuation!(BvNeg, .~.);
    syn::custom_punctuation!(BvNot, .!.);
    syn::custom_punctuation!(BvArithShr, #>>);
}

#[derive(PartialEq, Debug)]
pub enum Sort {
    Int,
    Bool,
    Bitvec { length: syn::LitInt },
}

impl Parse for Sort {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(kw::int) {
            input.parse::<kw::int>()?;
            Ok(Sort::Int)
        } else if lookahead.peek(kw::bool) {
            input.parse::<kw::bool>()?;
            Ok(Sort::Bool)
        } else if lookahead.peek(kw::bitvec) {
            input.parse::<kw::bitvec>()?;
            input.parse::<syn::Token![<]>()?;
            let length = input.parse()?;
            input.parse::<syn::Token![>]>()?;
            Ok(Sort::Bitvec { length })
        } else {
            Err(lookahead.error())
        }
    }
}
