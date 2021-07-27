#![feature(proc_macro_hygiene)]

use std::convert::TryInto;
use std::env;
use std::fmt;

use itertools::iproduct;
use z3::{ast, Config, Context, Solver};
use z3d::{dec, exp};

#[derive(PartialEq)]
struct Puzzle {
    values: [[i64; 9]; 9],
}

impl fmt::Display for Puzzle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (row_ind, row) in self.values.iter().enumerate() {
            if row_ind > 0 {
                writeln!(f)?;
                if row_ind % 3 == 0 {
                    writeln!(f, "---+---+---")?;
                }
            }
            for (col_ind, val) in row.iter().enumerate() {
                if col_ind % 3 == 0 && col_ind > 0 {
                    write!(f, "|")?;
                }
                write!(f, "{}", val)?;
            }
        }
        Ok(())
    }
}

#[allow(clippy::identity_op)]
fn solve_z3d(puzzle: &Puzzle) -> Option<Puzzle> {
    let ctx = &Context::new(&Config::default());
    let solver = Solver::new(ctx);

    // Declare variables for cells
    let cells: Vec<Vec<ast::BV>> = (0..9)
        .map(|rr| {
            (0..9)
                .map(|cc| dec!($("cell_{}_{}", rr, cc): bitvec<16> in ctx))
                .collect()
        })
        .collect();

    // Constrain known values
    for (rr, cc) in iproduct!(0..9, 0..9) {
        let val = puzzle.values[rr][cc];
        if val != 0 {
            solver.assert(&exp!({cells[rr][cc]} == (val as bitvec<16>) in ctx));
        }
    }

    let one = exp!(1 as bitvec<16> in ctx);
    let mask = exp!(0b11_1111_1110 as bitvec<16> in ctx);

    for i in 0..9 {
        // Values in row i must be distinct
        solver.assert(&exp!(bvor(
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

        // Values in column i must be distinct
        solver.assert(&exp!(bvor(
            one << {cells[0][i]},
            one << {cells[1][i]},
            one << {cells[2][i]},
            one << {cells[3][i]},
            one << {cells[4][i]},
            one << {cells[5][i]},
            one << {cells[6][i]},
            one << {cells[7][i]},
            one << {cells[8][i]}) == mask
        in ctx));

        // Values in square i must be distinct
        let square_row = (i / 3) * 3;
        let square_col = (i % 3) * 3;
        solver.assert(&exp!(bvor(
            one << {cells[square_row + 0][square_col + 0]},
            one << {cells[square_row + 0][square_col + 1]},
            one << {cells[square_row + 0][square_col + 2]},
            one << {cells[square_row + 1][square_col + 0]},
            one << {cells[square_row + 1][square_col + 1]},
            one << {cells[square_row + 1][square_col + 2]},
            one << {cells[square_row + 2][square_col + 0]},
            one << {cells[square_row + 2][square_col + 1]},
            one << {cells[square_row + 2][square_col + 2]}) == mask
        in ctx));
    }

    extract_solution(&solver, &cells)
}

#[allow(clippy::identity_op)]
fn solve_plain(puzzle: &Puzzle) -> Option<Puzzle> {
    use z3::ast::Ast;

    let ctx = &Context::new(&Config::default());
    let solver = Solver::new(ctx);

    // Declare variables for cells
    let cells: Vec<Vec<ast::BV>> = (0..9)
        .map(|rr| {
            (0..9)
                .map(|cc| ast::BV::new_const(ctx, format!("cell_{}_{}", rr, cc), 16))
                .collect()
        })
        .collect();

    // Constrain known values
    for (rr, cc) in iproduct!(0..9, 0..9) {
        let val = puzzle.values[rr][cc];
        if val != 0 {
            solver.assert(&cells[rr][cc]._eq(&ast::BV::from_i64(ctx, val, 16)));
        }
    }

    let one = ast::BV::from_i64(ctx, 1, 16);
    let mask = ast::BV::from_i64(ctx, 0b11_1111_1110, 16);

    for i in 0..9 {
        // Values in row i must be distinct
        solver.assert(
            &one.bvshl(&cells[i][0])
                .bvor(&one.bvshl(&cells[i][1]))
                .bvor(&one.bvshl(&cells[i][2]))
                .bvor(&one.bvshl(&cells[i][3]))
                .bvor(&one.bvshl(&cells[i][4]))
                .bvor(&one.bvshl(&cells[i][5]))
                .bvor(&one.bvshl(&cells[i][6]))
                .bvor(&one.bvshl(&cells[i][7]))
                .bvor(&one.bvshl(&cells[i][8]))
                ._eq(&mask),
        );

        // Values in column i must be distinct
        solver.assert(
            &one.bvshl(&cells[0][i])
                .bvor(&one.bvshl(&cells[1][i]))
                .bvor(&one.bvshl(&cells[2][i]))
                .bvor(&one.bvshl(&cells[3][i]))
                .bvor(&one.bvshl(&cells[4][i]))
                .bvor(&one.bvshl(&cells[5][i]))
                .bvor(&one.bvshl(&cells[6][i]))
                .bvor(&one.bvshl(&cells[7][i]))
                .bvor(&one.bvshl(&cells[8][i]))
                ._eq(&mask),
        );

        // Values in square i must be distinct
        let square_row = (i / 3) * 3;
        let square_col = (i % 3) * 3;
        solver.assert(
            &one.bvshl(&cells[square_row + 0][square_col + 0])
                .bvor(&one.bvshl(&cells[square_row + 0][square_col + 1]))
                .bvor(&one.bvshl(&cells[square_row + 0][square_col + 2]))
                .bvor(&one.bvshl(&cells[square_row + 1][square_col + 0]))
                .bvor(&one.bvshl(&cells[square_row + 1][square_col + 1]))
                .bvor(&one.bvshl(&cells[square_row + 1][square_col + 2]))
                .bvor(&one.bvshl(&cells[square_row + 2][square_col + 0]))
                .bvor(&one.bvshl(&cells[square_row + 2][square_col + 1]))
                .bvor(&one.bvshl(&cells[square_row + 2][square_col + 2]))
                ._eq(&mask),
        );
    }

    extract_solution(&solver, &cells)
}

fn extract_solution(solver: &Solver, cells: &Vec<Vec<ast::BV>>) -> Option<Puzzle> {
    if solver.check() != z3::SatResult::Sat {
        return None;
    }

    if let Some(model) = solver.get_model() {
        let mut solution = [[0; 9]; 9];
        for (rr, cc) in iproduct!(0..9, 0..9) {
            let val = model.eval(&cells[rr][cc], true).unwrap().as_i64().unwrap();
            solution[rr][cc] = val.try_into().unwrap();
        }
        return Some(Puzzle { values: solution });
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let usage = "usage: cargo run --bin sudoku [ z3d | plain ]";
    if args.len() != 2 {
        return Err(usage.into());
    }

    let solve = match args[1].as_str() {
        "z3d" => solve_z3d,
        "plain" => solve_plain,
        _ => return Err(usage.into()),
    };

    let puzzle = Puzzle {
        values: [
            [0, 0, 5, 3, 0, 0, 0, 0, 0],
            [8, 0, 0, 0, 0, 0, 0, 2, 0],
            [0, 7, 0, 0, 1, 0, 5, 0, 0],
            [4, 0, 0, 0, 0, 5, 3, 0, 0],
            [0, 1, 0, 0, 7, 0, 0, 0, 6],
            [0, 0, 3, 2, 0, 0, 0, 8, 0],
            [0, 6, 0, 5, 0, 0, 0, 0, 9],
            [0, 0, 4, 0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 9, 7, 0, 0],
        ],
    };
    println!("Here is the puzzle:\n{}", puzzle);

    match solve(&puzzle) {
        Some(solution) => println!("And here is the solution:\n{}", solution),
        None => println!("No solution found!"),
    };

    Ok(())
}
