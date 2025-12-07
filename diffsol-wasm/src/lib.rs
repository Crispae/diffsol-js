//! JavaScript/WASM bindings for diffsol with callback support.
//!
//! This crate provides a way to use diffsol from JavaScript by allowing
//! users to define ODE equations as JavaScript callback functions.
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { solve_ode } from './diffsol_wasm.js';
//!
//! await init();
//!
//! // Solve exponential decay: dy/dt = -a*y
//! // Without num_points: returns all internal solver steps
//! const result = solve_ode(
//!     1,  // nstates
//!     new Float64Array([0.1]),  // parameters [a]
//!     10.0,  // final time
//!     // RHS function: dy/dt = -a*y
//!     (x, p, t, y) => { y[0] = -p[0] * x[0]; },
//!     // Initial condition function
//!     (p, t, y) => { y[0] = 1.0; },
//!     null  // num_points: null means use all internal steps
//! );
//!
//! // With num_points: returns exactly 100 evenly spaced points from 0 to 2400 hours
//! const result2 = solve_ode(
//!     1,
//!     new Float64Array([0.1]),
//!     2400.0,  // 2400 hours
//!     (x, p, t, y) => { y[0] = -p[0] * x[0]; },
//!     (p, t, y) => { y[0] = 1.0; },
//!     100  // num_points: exactly 100 output points
//! );
//! console.log(result2.times.length);  // 100
//! console.log(result2.times, result2.states);
//! ```

mod error;

use std::cell::RefCell;
use std::rc::Rc;

use diffsol::{
    NalgebraContext, NalgebraMat, NalgebraVec, OdeBuilder, OdeEquations, OdeSolverMethod, Op,
    Vector,
};
use js_sys::{Float64Array, Function};
use wasm_bindgen::prelude::*;

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;

/// Initialize the WASM module (called automatically on load).
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Convert any error to JsValue
fn to_js_error<E: std::fmt::Display>(err: E) -> JsValue {
    JsValue::from_str(&err.to_string())
}

/// Convert Float64Array to Vec<f64>
fn js_to_vec(arr: &Float64Array) -> Vec<f64> {
    arr.to_vec()
}

/// Convert slice to Float64Array
fn vec_to_js(v: &[f64]) -> Float64Array {
    Float64Array::from(v)
}

/// Solve an ODE system using an explicit Runge-Kutta method (TSIT45).
///
/// This is suitable for non-stiff ODEs. For stiff problems, use `solve_ode_bdf`.
///
/// # Arguments
/// * `nstates` - Number of state variables
/// * `params` - Parameter array as Float64Array
/// * `t_final` - Final time to solve to
/// * `rhs_fn` - RHS function: `(x, p, t, y) => void` where y is output
/// * `init_fn` - Initial condition: `(p, t, y) => void` where y is output
/// * `num_points` - Optional number of evenly spaced output points. If None, returns all internal solver steps.
///
/// # Returns
/// Object with `times` (Float64Array) and `states` (array of Float64Array)
#[wasm_bindgen]
pub fn solve_ode(
    nstates: usize,
    params: Float64Array,
    t_final: f64,
    rhs_fn: Function,
    init_fn: Function,
    num_points: Option<usize>,
) -> Result<JsValue, JsValue> {
    let params_vec = js_to_vec(&params);

    // Store JS functions in Rc<RefCell> so closures can share them
    let rhs_js = Rc::new(RefCell::new(rhs_fn));
    let init_js = Rc::new(RefCell::new(init_fn));

    // Create wrapper closures that call JS functions
    let rhs_js_clone = rhs_js.clone();
    let rhs_closure = move |x: &V, p: &V, t: f64, y: &mut V| {
        let x_js = vec_to_js(&x.clone_as_vec());
        let p_js = vec_to_js(&p.clone_as_vec());
        let y_js = Float64Array::new_with_length(y.len() as u32);

        let _ = rhs_js_clone.borrow().call4(
            &JsValue::NULL,
            &x_js,
            &p_js,
            &JsValue::from_f64(t),
            &y_js,
        );

        let result = js_to_vec(&y_js);
        for (i, val) in result.into_iter().enumerate() {
            y[i] = val;
        }
    };

    let init_js_clone = init_js.clone();
    let init_closure = move |p: &V, t: f64, y: &mut V| {
        let p_js = vec_to_js(&p.clone_as_vec());
        let y_js = Float64Array::new_with_length(y.len() as u32);

        let _ = init_js_clone.borrow().call3(
            &JsValue::NULL,
            &p_js,
            &JsValue::from_f64(t),
            &y_js,
        );

        let result = js_to_vec(&y_js);
        for (i, val) in result.into_iter().enumerate() {
            y[i] = val;
        }
    };

    // Build the ODE problem
    let problem = OdeBuilder::<M>::new()
        .rhs(rhs_closure)
        .init(init_closure, nstates)
        .p(params_vec)
        .build()
        .map_err(to_js_error)?;

    // Solve using TSIT45 (explicit Runge-Kutta, suitable for non-stiff problems)
    let mut solver = problem.tsit45().map_err(to_js_error)?;
    
    // Get initial time
    let t0 = solver.state().t;
    
    // Solve with specified number of points or default behavior
    let (ys, ts) = if let Some(n) = num_points {
        // Generate evenly spaced time points from t0 to t_final
        let t_eval: Vec<f64> = if n == 1 {
            vec![t_final]
        } else {
            (0..n)
                .map(|i| t0 + (t_final - t0) * (i as f64) / ((n - 1) as f64))
                .collect()
        };
        
        // Use solve_dense to get solution at specific time points
        let ys = solver.solve_dense(&t_eval).map_err(to_js_error)?;
        (ys, t_eval)
    } else {
        // Default: use solve() which returns all internal steps
        solver.solve(t_final).map_err(to_js_error)?
    };

    // Convert results
    let times = vec_to_js(&ts);
    let nstates = problem.eqn.rhs().nstates();
    let ntimes = ts.len();

    let states = js_sys::Array::new();
    for i in 0..ntimes {
        let state_vec: Vec<f64> = (0..nstates).map(|j| ys[(j, i)]).collect();
        states.push(&vec_to_js(&state_vec));
    }

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &JsValue::from_str("times"), &times)?;
    js_sys::Reflect::set(&result, &JsValue::from_str("states"), &states)?;

    Ok(result.into())
}

/// Solve a stiff ODE system using BDF (Backward Differentiation Formula).
///
/// This requires providing a Jacobian function. For non-stiff problems,
/// prefer `solve_ode` which doesn't require a Jacobian.
///
/// # Arguments
/// * `nstates` - Number of state variables
/// * `params` - Parameter array as Float64Array
/// * `t_final` - Final time to solve to
/// * `rhs_fn` - RHS function: `(x, p, t, y) => void`
/// * `jac_fn` - Jacobian-vector product: `(x, p, t, v, y) => void` computes y = J*v
/// * `init_fn` - Initial condition: `(p, t, y) => void`
/// * `num_points` - Optional number of evenly spaced output points. If None, returns all internal solver steps.
///
/// # Returns
/// Object with `times` (Float64Array) and `states` (array of Float64Array)
#[wasm_bindgen]
pub fn solve_ode_bdf(
    nstates: usize,
    params: Float64Array,
    t_final: f64,
    rhs_fn: Function,
    jac_fn: Function,
    init_fn: Function,
    num_points: Option<usize>,
) -> Result<JsValue, JsValue> {
    let params_vec = js_to_vec(&params);

    // Store JS functions in Rc<RefCell>
    let rhs_js = Rc::new(RefCell::new(rhs_fn));
    let jac_js = Rc::new(RefCell::new(jac_fn));
    let init_js = Rc::new(RefCell::new(init_fn));

    // RHS closure
    let rhs_js_clone = rhs_js.clone();
    let rhs_closure = move |x: &V, p: &V, t: f64, y: &mut V| {
        let x_js = vec_to_js(&x.clone_as_vec());
        let p_js = vec_to_js(&p.clone_as_vec());
        let y_js = Float64Array::new_with_length(y.len() as u32);

        let _ = rhs_js_clone.borrow().call4(
            &JsValue::NULL,
            &x_js,
            &p_js,
            &JsValue::from_f64(t),
            &y_js,
        );

        let result = js_to_vec(&y_js);
        for (i, val) in result.into_iter().enumerate() {
            y[i] = val;
        }
    };

    // Jacobian closure
    let jac_js_clone = jac_js.clone();
    let jac_closure = move |x: &V, p: &V, t: f64, v: &V, y: &mut V| {
        let x_js = vec_to_js(&x.clone_as_vec());
        let p_js = vec_to_js(&p.clone_as_vec());
        let v_js = vec_to_js(&v.clone_as_vec());
        let y_js = Float64Array::new_with_length(y.len() as u32);

        // Call JS function: jac(x, p, t, v, y)
        let args = js_sys::Array::new();
        args.push(&x_js);
        args.push(&p_js);
        args.push(&JsValue::from_f64(t));
        args.push(&v_js);
        args.push(&y_js);

        let _ = jac_js_clone.borrow().apply(&JsValue::NULL, &args);

        let result = js_to_vec(&y_js);
        for (i, val) in result.into_iter().enumerate() {
            y[i] = val;
        }
    };

    // Init closure
    let init_js_clone = init_js.clone();
    let init_closure = move |p: &V, t: f64, y: &mut V| {
        let p_js = vec_to_js(&p.clone_as_vec());
        let y_js = Float64Array::new_with_length(y.len() as u32);

        let _ = init_js_clone.borrow().call3(
            &JsValue::NULL,
            &p_js,
            &JsValue::from_f64(t),
            &y_js,
        );

        let result = js_to_vec(&y_js);
        for (i, val) in result.into_iter().enumerate() {
            y[i] = val;
        }
    };

    // Build the ODE problem with Jacobian for implicit solver
    let problem = OdeBuilder::<M>::new()
        .rhs_implicit(rhs_closure, jac_closure)
        .init(init_closure, nstates)
        .p(params_vec)
        .build()
        .map_err(to_js_error)?;

    // Solve using BDF (suitable for stiff problems)
    let mut solver = problem.bdf::<diffsol::NalgebraLU<f64>>().map_err(to_js_error)?;
    
    // Get initial time
    let t0 = solver.state().t;
    
    // Solve with specified number of points or default behavior
    let (ys, ts) = if let Some(n) = num_points {
        // Generate evenly spaced time points from t0 to t_final
        let t_eval: Vec<f64> = if n == 1 {
            vec![t_final]
        } else {
            (0..n)
                .map(|i| t0 + (t_final - t0) * (i as f64) / ((n - 1) as f64))
                .collect()
        };
        
        // Use solve_dense to get solution at specific time points
        let ys = solver.solve_dense(&t_eval).map_err(to_js_error)?;
        (ys, t_eval)
    } else {
        // Default: use solve() which returns all internal steps
        solver.solve(t_final).map_err(to_js_error)?
    };

    // Convert results
    let times = vec_to_js(&ts);
    let nstates = problem.eqn.rhs().nstates();
    let ntimes = ts.len();

    let states = js_sys::Array::new();
    for i in 0..ntimes {
        let state_vec: Vec<f64> = (0..nstates).map(|j| ys[(j, i)]).collect();
        states.push(&vec_to_js(&state_vec));
    }

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &JsValue::from_str("times"), &times)?;
    js_sys::Reflect::set(&result, &JsValue::from_str("states"), &states)?;

    Ok(result.into())
}

/// Example: Solve the Lotka-Volterra (predator-prey) system.
///
/// This is a demonstration function showing how to use the solver.
/// The system is: dx/dt = ax - bxy, dy/dt = cxy - dy
///
/// # Arguments
/// * `t_final` - Final time to solve to
///
/// # Returns
/// Object with `times` and `states`
#[wasm_bindgen]
pub fn solve_lotka_volterra(t_final: f64) -> Result<JsValue, JsValue> {
    // Lotka-Volterra system
    let problem = OdeBuilder::<M>::new()
        .rhs(|x, p, _t, y| {
            // dx/dt = a*x - b*x*y
            // dy/dt = c*x*y - d*y
            y[0] = p[0] * x[0] - p[1] * x[0] * x[1];
            y[1] = p[2] * x[0] * x[1] - p[3] * x[1];
        })
        .init(|_p, _t, y| {
            y[0] = 1.0;  // prey
            y[1] = 1.0;  // predator
        }, 2)
        .p(vec![2.0 / 3.0, 4.0 / 3.0, 1.0, 1.0])  // a, b, c, d
        .build()
        .map_err(to_js_error)?;

    let mut solver = problem.tsit45().map_err(to_js_error)?;
    let (ys, ts) = solver.solve(t_final).map_err(to_js_error)?;

    let times = vec_to_js(&ts);
    let ntimes = ts.len();

    let states = js_sys::Array::new();
    for i in 0..ntimes {
        let state_vec: Vec<f64> = (0..2).map(|j| ys[(j, i)]).collect();
        states.push(&vec_to_js(&state_vec));
    }

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &JsValue::from_str("times"), &times)?;
    js_sys::Reflect::set(&result, &JsValue::from_str("states"), &states)?;

    Ok(result.into())
}

/// Solve a stiff ODE system using TR-BDF2 (SDIRK method).
///
/// TR-BDF2 is a combination of the trapezoidal rule and BDF2.
/// Requires providing a Jacobian function.
///
/// # Arguments
/// * `nstates` - Number of state variables
/// * `params` - Parameter array as Float64Array
/// * `t_final` - Final time to solve to
/// * `rhs_fn` - RHS function: `(x, p, t, y) => void`
/// * `jac_fn` - Jacobian-vector product: `(x, p, t, v, y) => void` computes y = J*v
/// * `init_fn` - Initial condition: `(p, t, y) => void`
/// * `num_points` - Optional number of evenly spaced output points. If None, returns all internal solver steps.
///
/// # Returns
/// Object with `times` (Float64Array) and `states` (array of Float64Array)
#[wasm_bindgen]
pub fn solve_ode_tr_bdf2(
    nstates: usize,
    params: Float64Array,
    t_final: f64,
    rhs_fn: Function,
    jac_fn: Function,
    init_fn: Function,
    num_points: Option<usize>,
) -> Result<JsValue, JsValue> {
    solve_with_implicit_solver(nstates, params, t_final, rhs_fn, jac_fn, init_fn, ImplicitSolverType::TrBdf2, num_points)
}

/// Solve a stiff ODE system using ESDIRK34 (4th order SDIRK method).
///
/// ESDIRK34 is a 4th order diagonally implicit Runge-Kutta method.
/// Requires providing a Jacobian function.
///
/// # Arguments
/// * `nstates` - Number of state variables
/// * `params` - Parameter array as Float64Array
/// * `t_final` - Final time to solve to
/// * `rhs_fn` - RHS function: `(x, p, t, y) => void`
/// * `jac_fn` - Jacobian-vector product: `(x, p, t, v, y) => void` computes y = J*v
/// * `init_fn` - Initial condition: `(p, t, y) => void`
/// * `num_points` - Optional number of evenly spaced output points. If None, returns all internal solver steps.
///
/// # Returns
/// Object with `times` (Float64Array) and `states` (array of Float64Array)
#[wasm_bindgen]
pub fn solve_ode_esdirk34(
    nstates: usize,
    params: Float64Array,
    t_final: f64,
    rhs_fn: Function,
    jac_fn: Function,
    init_fn: Function,
    num_points: Option<usize>,
) -> Result<JsValue, JsValue> {
    solve_with_implicit_solver(nstates, params, t_final, rhs_fn, jac_fn, init_fn, ImplicitSolverType::Esdirk34, num_points)
}

/// Internal enum for implicit solver type selection
enum ImplicitSolverType {
    TrBdf2,
    Esdirk34,
}

/// Internal helper for implicit solvers (TR-BDF2, ESDIRK34)
fn solve_with_implicit_solver(
    nstates: usize,
    params: Float64Array,
    t_final: f64,
    rhs_fn: Function,
    jac_fn: Function,
    init_fn: Function,
    solver_type: ImplicitSolverType,
    num_points: Option<usize>,
) -> Result<JsValue, JsValue> {
    let params_vec = js_to_vec(&params);

    // Store JS functions in Rc<RefCell>
    let rhs_js = Rc::new(RefCell::new(rhs_fn));
    let jac_js = Rc::new(RefCell::new(jac_fn));
    let init_js = Rc::new(RefCell::new(init_fn));

    // RHS closure
    let rhs_js_clone = rhs_js.clone();
    let rhs_closure = move |x: &V, p: &V, t: f64, y: &mut V| {
        let x_js = vec_to_js(&x.clone_as_vec());
        let p_js = vec_to_js(&p.clone_as_vec());
        let y_js = Float64Array::new_with_length(y.len() as u32);

        let _ = rhs_js_clone.borrow().call4(
            &JsValue::NULL,
            &x_js,
            &p_js,
            &JsValue::from_f64(t),
            &y_js,
        );

        let result = js_to_vec(&y_js);
        for (i, val) in result.into_iter().enumerate() {
            y[i] = val;
        }
    };

    // Jacobian closure
    let jac_js_clone = jac_js.clone();
    let jac_closure = move |x: &V, p: &V, t: f64, v: &V, y: &mut V| {
        let x_js = vec_to_js(&x.clone_as_vec());
        let p_js = vec_to_js(&p.clone_as_vec());
        let v_js = vec_to_js(&v.clone_as_vec());
        let y_js = Float64Array::new_with_length(y.len() as u32);

        let args = js_sys::Array::new();
        args.push(&x_js);
        args.push(&p_js);
        args.push(&JsValue::from_f64(t));
        args.push(&v_js);
        args.push(&y_js);

        let _ = jac_js_clone.borrow().apply(&JsValue::NULL, &args);

        let result = js_to_vec(&y_js);
        for (i, val) in result.into_iter().enumerate() {
            y[i] = val;
        }
    };

    // Init closure
    let init_js_clone = init_js.clone();
    let init_closure = move |p: &V, t: f64, y: &mut V| {
        let p_js = vec_to_js(&p.clone_as_vec());
        let y_js = Float64Array::new_with_length(y.len() as u32);

        let _ = init_js_clone.borrow().call3(
            &JsValue::NULL,
            &p_js,
            &JsValue::from_f64(t),
            &y_js,
        );

        let result = js_to_vec(&y_js);
        for (i, val) in result.into_iter().enumerate() {
            y[i] = val;
        }
    };

    // Build the ODE problem with Jacobian
    let problem = OdeBuilder::<M>::new()
        .rhs_implicit(rhs_closure, jac_closure)
        .init(init_closure, nstates)
        .p(params_vec)
        .build()
        .map_err(to_js_error)?;

    // Solve using the selected method
    let (ys, ts) = match solver_type {
        ImplicitSolverType::TrBdf2 => {
            let mut solver = problem.tr_bdf2::<diffsol::NalgebraLU<f64>>().map_err(to_js_error)?;
            let t0 = solver.state().t;
            
            if let Some(n) = num_points {
                // Generate evenly spaced time points from t0 to t_final
                let t_eval: Vec<f64> = if n == 1 {
                    vec![t_final]
                } else {
                    (0..n)
                        .map(|i| t0 + (t_final - t0) * (i as f64) / ((n - 1) as f64))
                        .collect()
                };
                
                // Use solve_dense to get solution at specific time points
                let ys = solver.solve_dense(&t_eval).map_err(to_js_error)?;
                (ys, t_eval)
            } else {
                solver.solve(t_final).map_err(to_js_error)?
            }
        }
        ImplicitSolverType::Esdirk34 => {
            let mut solver = problem.esdirk34::<diffsol::NalgebraLU<f64>>().map_err(to_js_error)?;
            let t0 = solver.state().t;
            
            if let Some(n) = num_points {
                // Generate evenly spaced time points from t0 to t_final
                let t_eval: Vec<f64> = if n == 1 {
                    vec![t_final]
                } else {
                    (0..n)
                        .map(|i| t0 + (t_final - t0) * (i as f64) / ((n - 1) as f64))
                        .collect()
                };
                
                // Use solve_dense to get solution at specific time points
                let ys = solver.solve_dense(&t_eval).map_err(to_js_error)?;
                (ys, t_eval)
            } else {
                solver.solve(t_final).map_err(to_js_error)?
            }
        }
    };

    // Convert results
    let times = vec_to_js(&ts);
    let nstates_out = problem.eqn.rhs().nstates();
    let ntimes = ts.len();

    let states = js_sys::Array::new();
    for i in 0..ntimes {
        let state_vec: Vec<f64> = (0..nstates_out).map(|j| ys[(j, i)]).collect();
        states.push(&vec_to_js(&state_vec));
    }

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &JsValue::from_str("times"), &times)?;
    js_sys::Reflect::set(&result, &JsValue::from_str("states"), &states)?;

    Ok(result.into())
}

