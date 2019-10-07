#![feature(proc_macro_hygiene)]

use z3;
use z3d::dec;

fn default_ctx() -> z3::Context {
    z3::Context::new(&z3::Config::default())
}

#[test]
fn modus_tollens() {
    let ctx = &default_ctx();
    let solver = z3::Solver::new(ctx);
    let p = dec!(p: bool in ctx);
    let q = dec!(q: bool in ctx);

    solver.assert(&p.implies(&q));
    solver.assert(&q.not());
    assert_eq!(solver.check(), z3::SatResult::Sat);

    solver.push();
    solver.assert(&p);
    assert_eq!(solver.check(), z3::SatResult::Unsat);

    solver.pop(1);
    assert_eq!(solver.check(), z3::SatResult::Sat);
}
