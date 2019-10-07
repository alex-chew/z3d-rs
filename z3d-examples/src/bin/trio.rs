//! # Trio puzzle
//!
//! Solves the Alphametics puzzle "VIOLIN + VIOLIN + VIOLA = TRIO + SONATA" as described in Dennis
//! Yurichev's "SAT/SMT by example" document.

#![feature(proc_macro_hygiene)]

use std::env;

use z3::{ast, Config, Context, Model, Solver};
use z3d::{dec, exp};

struct Vars<'a> {
    num_a: ast::Int<'a>,
    num_i: ast::Int<'a>,
    num_l: ast::Int<'a>,
    num_n: ast::Int<'a>,
    num_o: ast::Int<'a>,
    num_r: ast::Int<'a>,
    num_s: ast::Int<'a>,
    num_t: ast::Int<'a>,
    num_v: ast::Int<'a>,
    violin: ast::Int<'a>,
    viola: ast::Int<'a>,
    sonata: ast::Int<'a>,
    trio: ast::Int<'a>,
}

fn solve_z3d(ctx: &'_ Context) -> (Vars<'_>, Solver<'_>) {
    let solver = Solver::new(&ctx);

    let num_a = dec!(a: int in ctx);
    let num_i = dec!(i: int in ctx);
    let num_l = dec!(l: int in ctx);
    let num_n = dec!(n: int in ctx);
    let num_o = dec!(o: int in ctx);
    let num_r = dec!(r: int in ctx);
    let num_s = dec!(s: int in ctx);
    let num_t = dec!(t: int in ctx);
    let num_v = dec!(v: int in ctx);

    solver.assert(&exp!(
        distinct(num_a, num_i, num_l, num_n, num_o, num_r, num_s, num_t, num_v) in ctx));
    for num in &[
        &num_a, &num_i, &num_l, &num_n, &num_o, &num_r, &num_s, &num_t, &num_v,
    ] {
        solver.assert(&exp!((num >= 0) & (num <= 9) in ctx));
    }

    let violin = dec!(violin: int in ctx);
    let viola = dec!(viola: int in ctx);
    let sonata = dec!(sonata: int in ctx);
    let trio = dec!(trio: int in ctx);
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
    solver.assert(&exp!(
        add(
            num_v * 10000,
            (num_i * 1000),
            num_o * 100,
            num_l * 10,
            num_a
        ) == viola
    in ctx));
    solver.assert(&exp!(
        add(
            num_s * 100000,
            num_o * 10000,
            num_n * 1000,
            num_a * 100,
            num_t * 10,
            num_a
        ) == sonata
    in ctx));
    solver.assert(&exp!(
        add(
            num_t * 1000,
            num_r * 100,
            num_i * 10,
            num_o
        ) == trio
    in ctx));
    solver.assert(&exp!(add(violin, violin, viola) == (trio + sonata) in ctx));

    let vars = Vars {
        num_a,
        num_i,
        num_l,
        num_n,
        num_o,
        num_r,
        num_s,
        num_t,
        num_v,
        violin,
        viola,
        sonata,
        trio,
    };
    (vars, solver)
}

fn solve_plain(ctx: &'_ Context) -> (Vars<'_>, Solver<'_>) {
    use z3::ast::Ast;

    let solver = Solver::new(&ctx);

    let num_a = ast::Int::new_const(ctx, "a");
    let num_i = ast::Int::new_const(ctx, "i");
    let num_l = ast::Int::new_const(ctx, "l");
    let num_n = ast::Int::new_const(ctx, "n");
    let num_o = ast::Int::new_const(ctx, "o");
    let num_r = ast::Int::new_const(ctx, "r");
    let num_s = ast::Int::new_const(ctx, "s");
    let num_t = ast::Int::new_const(ctx, "t");
    let num_v = ast::Int::new_const(ctx, "v");

    solver.assert(&num_a.distinct(&[
        &num_i, &num_l, &num_n, &num_o, &num_r, &num_s, &num_t, &num_v,
    ]));
    for num in &[
        &num_a, &num_i, &num_l, &num_n, &num_o, &num_r, &num_s, &num_t, &num_v,
    ] {
        solver.assert(
            &num.ge(&ast::Int::from_i64(ctx, 0))
                .and(&[&num.le(&ast::Int::from_i64(ctx, 9))]),
        );
    }

    let violin = ast::Int::new_const(ctx, "violin");
    let viola = ast::Int::new_const(ctx, "viola");
    let sonata = ast::Int::new_const(ctx, "sonata");
    let trio = ast::Int::new_const(ctx, "trio");
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
    solver.assert(
        &(num_v.mul(&[&ast::Int::from_i64(ctx, 10000)]))
            .add(&[
                &num_i.mul(&[&ast::Int::from_i64(ctx, 1000)]),
                &num_o.mul(&[&ast::Int::from_i64(ctx, 100)]),
                &num_l.mul(&[&ast::Int::from_i64(ctx, 10)]),
                &num_a,
            ])
            ._eq(&viola),
    );
    solver.assert(
        &(num_s.mul(&[&ast::Int::from_i64(ctx, 100000)]))
            .add(&[
                &num_o.mul(&[&ast::Int::from_i64(ctx, 10000)]),
                &num_n.mul(&[&ast::Int::from_i64(ctx, 1000)]),
                &num_a.mul(&[&ast::Int::from_i64(ctx, 100)]),
                &num_t.mul(&[&ast::Int::from_i64(ctx, 10)]),
                &num_a,
            ])
            ._eq(&sonata),
    );
    solver.assert(
        &(num_t.mul(&[&ast::Int::from_i64(ctx, 1000)]))
            .add(&[
                &num_r.mul(&[&ast::Int::from_i64(ctx, 100)]),
                &num_i.mul(&[&ast::Int::from_i64(ctx, 10)]),
                &num_o,
            ])
            ._eq(&trio),
    );
    solver.assert(&violin.add(&[&violin, &viola])._eq(&trio.add(&[&sonata])));

    let vars = Vars {
        num_a,
        num_i,
        num_l,
        num_n,
        num_o,
        num_r,
        num_s,
        num_t,
        num_v,
        violin,
        viola,
        sonata,
        trio,
    };
    (vars, solver)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let usage = "usage: cargo run --bin trio [ z3d | plain ]";
    if args.len() != 2 {
        return Err(usage.into());
    }

    let solve = match args[1].as_str() {
        "z3d" => solve_z3d,
        "plain" => solve_plain,
        _ => return Err(usage.into()),
    };

    println!("Solving: VIOLIN + VIOLIN + VIOLA = TRIO + SONATA");
    let ctx = Context::new(&Config::default());
    let (vars, solver) = solve(&ctx);
    assert!(solver.check() == z3::SatResult::Sat, "No solution found!");
    print_model(&solver.get_model(), &vars);
    Ok(())
}

fn print_model(model: &Model, vars: &Vars) {
    println!("A = {}", get_int_from_model(model, &vars.num_a));
    println!("I = {}", get_int_from_model(model, &vars.num_i));
    println!("L = {}", get_int_from_model(model, &vars.num_l));
    println!("N = {}", get_int_from_model(model, &vars.num_n));
    println!("O = {}", get_int_from_model(model, &vars.num_o));
    println!("R = {}", get_int_from_model(model, &vars.num_r));
    println!("S = {}", get_int_from_model(model, &vars.num_s));
    println!("T = {}", get_int_from_model(model, &vars.num_t));
    println!("V = {}", get_int_from_model(model, &vars.num_v));

    println!("VIOLIN = {}", get_int_from_model(model, &vars.violin));
    println!("VIOLA = {}", get_int_from_model(model, &vars.viola));
    println!("SONATA = {}", get_int_from_model(model, &vars.sonata));
    println!("TRIO = {}", get_int_from_model(model, &vars.trio));
}

fn get_int_from_model<'v>(model: &Model, var: &ast::Int<'v>) -> i64 {
    model.eval(var).unwrap().as_i64().unwrap()
}
