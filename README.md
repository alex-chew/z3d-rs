# Z3D — the Z3 DSL interface for Rust

**Z3D** (short for "Z3 DSL") is a Rust interface to the [Z3 theorem
prover](https://github.com/Z3Prover/z3). It provides two macros — `dec!` and
`exp!` — which allow the user to write Z3 constant (variable) **declarations** and
assertion **expressions** in a natural DSL, respectively.

Z3D has *zero run-time overhead*, since Rust macros expand source code (in DSL)
to source code (in plain Z3 code), and the macros expand to the same plain Z3
code one would write without Z3D.

*Disclaimer:*
Z3D is an experimental project, and has not been thoroughly tested. It should
be considered alpha-quality software.

## Examples

Here we'll demonstrate usage of Z3D, comparing its API to that of the [`z3`
package](https://github.com/prove-rs/z3.rs) (which provides the underlying
high-level Z3 bindings). For each example below, we'll assume the following
setup:

```rust
use z3::{Config, Context, Solver};  // basic Z3 interfaces from the z3 package
use z3d::{dec, exp};                // Z3D declaration and expression macros, respectively

let ctx = &Context::new(&Config::default());   // we declare constants in a Context
let solver = Solver::new(ctx);                 // we make assertions in a Solver
```

(These examples and their implementations are adapted from Dennis Yurichev's
excellent document on practical SMT solver usage, entitled "SAT/SMT by
example". Its latest version can be found
[here](https://yurichev.com/writings/SAT_SMT_by_example.pdf)).

### Sudoku

Let's solve a Sudoku puzzle. First we'll declare constants to represent the
`cells` of the board, and constrain any `known_values`.

```rust
let cells: [[z3::ast::BV; 9]; 9] = ...;
for rr in 0..9 {
    for cc in 0..9 {
        // using z3d
        cells[rr][cc] = dec!($("cell_{}_{}", rr, cc): bitvec<16> in ctx);
        //                   ^^^^^^^^^^^^^^^^^^^^^^^
        //                   ^ format constant names using $(...)

        // using z3
        cells[rr][cc] = z3::ast::BV::new_const(ctx, format!("cell_{}_{}", rr, cc), 16));

        if let Some(val) = known_values[rr][cc] {
            // using z3d
            solver.assert(&exp!({cells[rr][cc]} == (val as bitvec<16>) in ctx));
            //                   ^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^
            //     arbitrary Rust expression ^      ^ cast to bitvector

            // using z3
            solver.assert(&cells[rr][cc]._eq(&ctx.bitvector_sort(16).from_i64(val)));
        }
    }
}
```

Next we enforce uniqueness of cell values within each row, column, and 3x3
square (here we only demonstrate uniqueness in row `i`, for brevity). Note the
usage of the variadic `bvor` operator, which otherwise requires repeated
`.bvor(...)` calls.

```rust
// using z3d
let one = exp!(1 as bitvec<16> in ctx);
let mask = exp!(0b11_1111_1110 as bitvec<16> in ctx);
solver.assert(&exp!(bvor(    // variadic bvor
    one << {cells[i][0]},
    one << {cells[i][1]},
    one << {cells[i][2]},
    one << {cells[i][3]},
    one << {cells[i][4]},
    one << {cells[i][5]},
    one << {cells[i][6]},
    one << {cells[i][7]},
    one << {cells[i][8]}) == mask
in ctx));

// using z3
let one = z3::ast::BV::from_i64(ctx, 1, 16);
let mask = z3::ast::BV::from_i64(ctx, 0b11_1111_1110, 16);
solver.assert(
    &one.bvshl(&cells[i][0])
        .bvor(&one.bvshl(&cells[i][1]))     // repeated .bvor(...)
        .bvor(&one.bvshl(&cells[i][2]))
        .bvor(&one.bvshl(&cells[i][3]))
        .bvor(&one.bvshl(&cells[i][4]))
        .bvor(&one.bvshl(&cells[i][5]))
        .bvor(&one.bvshl(&cells[i][6]))
        .bvor(&one.bvshl(&cells[i][7]))
        .bvor(&one.bvshl(&cells[i][8]))
        ._eq(&mask),
);
```

Then a solution is just a `solver.get_model()` away. The full source code for
the Sudoku example is available as `z3d-examples/src/bin/sudoku.rs`.

### Verbal arithmetic

Let's solve the following [verbal
arithmetic](https://en.wikipedia.org/wiki/Verbal_arithmetic) problem:

```
  V I O L I N
  V I O L I N
+   V I O L A
-------------
  S O N A T A
+     T R I O
```

where each letter corresponds to a distinct digit 0-9, and the leading digit
of each word is nonzero. We first declare constants for each letter and word:

```rust
// using z3d
let num_a = dec!(a: int in ctx);
let num_i = dec!(i: int in ctx);
...
let num_v = dec!(v: int in ctx);
let violin = dec!(violin: int in ctx);
let viola = dec!(viola: int in ctx);
let sonata = dec!(sonata: int in ctx);
let trio = dec!(trio: int in ctx);

// using z3
let num_a = z3::ast::Int::new_const(ctx, "a");
let num_i = z3::ast::Int::new_const(ctx, "i");
...
let num_v = z3::ast::Int::new_const(ctx, "v");
let violin = z3::ast::Int::new_const(ctx, "violin");
let viola = z3::ast::Int::new_const(ctx, "viola");
let sonata = z3::ast::Int::new_const(ctx, "sonata");
let trio = z3::ast::Int::new_const(ctx, "trio");
```

Next we assert that the letters encode distinct 0-9 digits:

```rust
// using z3d
solver.assert(&exp!(
    distinct(num_a, num_i, num_l, num_n, num_o, num_r, num_s, num_t, num_v) in ctx));
for num in &[&num_a, &num_i, &num_l, &num_n, &num_o, &num_r, &num_s, &num_t, &num_v] {
    solver.assert(&exp!((num >= 0) & (num <= 9) in ctx));
}

// using z3
solver.assert(&num_a.distinct(&[
    &num_i, &num_l, &num_n, &num_o, &num_r, &num_s, &num_t, &num_v,
]));
for num in &[&num_a, &num_i, &num_l, &num_n, &num_o, &num_r, &num_s, &num_t, &num_v] {
    solver.assert(
        &num.ge(&ast::Int::from_i64(ctx, 0))
            .and(&[&num.le(&ast::Int::from_i64(ctx, 9))]),
    );
}
```

Then we assert that the letters form words (in the arithmetic sense):

```rust
// using z3d
solver.assert(&exp!(
    add(
        num_v * 100000,
        num_i * 10000,
        num_o * 1000,
        num_l * 100,
        num_i * 10,
        num_n
    ) == violin
in ctx));
...

// using z3
solver.assert(
    &(num_v.mul(&[&ast::Int::from_i64(ctx, 100000)]))
        .add(&[
            &num_i.mul(&[&ast::Int::from_i64(ctx, 10000)]),
            &num_o.mul(&[&ast::Int::from_i64(ctx, 1000)]),
            &num_l.mul(&[&ast::Int::from_i64(ctx, 100)]),
            &num_i.mul(&[&ast::Int::from_i64(ctx, 10)]),
            &num_n,
        ])
        ._eq(&violin),
);
...
```

And finally, we assert that the word equation holds:

```rust
// using z3d
solver.assert(&exp!(add(violin, violin, viola) == (trio + sonata) in ctx));

// using z3
solver.assert(&violin.add(&[&violin, &viola])._eq(&trio.add(&[&sonata])));
```

The full source code for the verbal arithmetic example is available as
`z3d-examples/src/bin/trio.rs`.

## Usage

Z3D has been tested using the `nightly-x86_64-unknown-linux-gnu` toolchain, in
particular `rustc 1.40.0-nightly (032a53a06 2019-10-03)`.

The only external dependency of this project is Z3; this project has been
tested with `libz3-dev` version `4.4.0-5` in the Ubuntu repositories. All Rust
dependencies are specified in `Cargo.toml` and will be installed automatically
using the appropriate commands as listed below.

```bash
$ cargo build                               # install dependencies and build
$ cargo run --bin sudoku [ z3d | plain ]    # run the Sudoku example
$ cargo run --bin trio [ z3d | plain ]      # run the Trio example
```

Each example requires an argument either `z3d` or `plain`, indicating which
implementation to use (either using the Z3D API, or the plain Z3 bindings). It
is thus easy to verify that the two implementations are equivalent in function
and performance.

## Roadmap

### `dec!`

- [x] Boolean, integer, and bitvector sorts
- [x] Named constants, with name formatting
- [ ] Support for more sorts: reals, relations, functions, quantifiers, ADTs
- [ ] Unnamed constants

### `exp!`

- [x] Atoms: integral/boolean literals, Rust identifiers
- [x] Common unary, binary, and variadic expressions
- [ ] Variadic expressions taking AST iterators
- [ ] Operator precedence and/or associativity

### Misc

- [x] Basic test suite
- [ ] Implicit `ctx` in macros
