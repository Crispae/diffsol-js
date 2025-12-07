# diffsol-wasm

JavaScript/WebAssembly bindings for [diffsol](https://github.com/martinjrobins/diffsol), enabling you to solve ordinary differential equations (ODEs) directly in the browser or Node.js using JavaScript callback functions.

## Features

- 🚀 **High Performance**: Leverages Rust's diffsol library compiled to WebAssembly
- 🔧 **Flexible**: Define ODE equations as JavaScript functions
- 📊 **Multiple Solvers**: Support for both non-stiff and stiff ODE problems
- 🎯 **Adaptive Step Size**: Automatic step size control with configurable tolerances
- 📈 **Dense Output**: Interpolate solutions at any time points you specify


### Building from Source

```bash
# Install wasm-pack if you haven't already
cargo install wasm-pack

# Build the WASM package
cd diffsol-wasm
wasm-pack build --target web --out-dir pkg
```

## Quick Start

```javascript
import init, { solve_ode } from './pkg/diffsol_wasm.js';

// Initialize the WASM module
await init();

// Solve exponential decay: dy/dt = -a*y, y(0) = 1
const result = solve_ode(
    1,                          // nstates: number of state variables
    new Float64Array([0.1]),    // parameters: [a]
    10.0,                       // t_final: final time
    // RHS function: dy/dt = -a*y
    (x, p, t, y) => {
        y[0] = -p[0] * x[0];
    },
    // Initial condition function
    (p, t, y) => {
        y[0] = 1.0;
    },
    null  // num_points: null means use all internal solver steps
);

console.log('Times:', result.times);
console.log('States:', result.states);
```

## API Reference

### `solve_ode(nstates, params, t_final, rhs_fn, init_fn, num_points?)`

Solve a non-stiff ODE system using the TSIT45 explicit Runge-Kutta method.

**Parameters:**
- `nstates` (number): Number of state variables
- `params` (Float64Array): Parameter array
- `t_final` (number): Final time to solve to
- `rhs_fn` (function): Right-hand side function `(x, p, t, y) => void`
  - `x`: Current state vector (Float64Array)
  - `p`: Parameter vector (Float64Array)
  - `t`: Current time (number)
  - `y`: Output vector to fill (Float64Array)
- `init_fn` (function): Initial condition function `(p, t, y) => void`
  - `p`: Parameter vector (Float64Array)
  - `t`: Initial time (number)
  - `y`: Initial state vector to fill (Float64Array)
- `num_points` (number, optional): Number of evenly spaced output points. If `null` or `undefined`, returns all internal solver steps.

**Returns:**
- Object with:
  - `times` (Float64Array): Time points
  - `states` (Array<Float64Array>): State vectors at each time point

**Example:**
```javascript
const result = solve_ode(
    2,                          // 2 state variables
    new Float64Array([1.0, 2.0]), // parameters
    10.0,                       // solve to t=10
    (x, p, t, y) => {
        y[0] = -p[0] * x[0];
        y[1] = -p[1] * x[1];
    },
    (p, t, y) => {
        y[0] = 1.0;
        y[1] = 2.0;
    },
    100  // return 100 evenly spaced points
);
```

### `solve_ode_bdf(nstates, params, t_final, rhs_fn, jac_fn, init_fn, num_points?)`

Solve a stiff ODE system using BDF (Backward Differentiation Formula). Requires a Jacobian function.

**Parameters:**
- `nstates` (number): Number of state variables
- `params` (Float64Array): Parameter array
- `t_final` (number): Final time to solve to
- `rhs_fn` (function): Right-hand side function `(x, p, t, y) => void`
- `jac_fn` (function): Jacobian-vector product function `(x, p, t, v, y) => void`
  - Computes `y = J*v` where `J` is the Jacobian matrix
  - `x`: Current state vector (Float64Array)
  - `p`: Parameter vector (Float64Array)
  - `t`: Current time (number)
  - `v`: Input vector (Float64Array)
  - `y`: Output vector to fill (Float64Array)
- `init_fn` (function): Initial condition function `(p, t, y) => void`
- `num_points` (number, optional): Number of evenly spaced output points

**Returns:**
- Object with `times` and `states` arrays

**Example:**
```javascript
// Stiff ODE: dy/dt = -λ*(y - cos(t))
const result = solve_ode_bdf(
    1,
    new Float64Array([1000.0]),  // λ = 1000
    1.0,
    (x, p, t, y) => {
        y[0] = p[0] * (Math.cos(t) - x[0]);
    },
    (x, p, t, v, y) => {
        y[0] = -p[0] * v[0];  // Jacobian: -λ
    },
    (p, t, y) => {
        y[0] = 1.0;
    }
);
```

### `solve_ode_tr_bdf2(nstates, params, t_final, rhs_fn, jac_fn, init_fn, num_points?)`

Solve a stiff ODE system using TR-BDF2 (Trapezoidal Rule - BDF2 combination), a 2nd order SDIRK method.

**Parameters:** Same as `solve_ode_bdf`

**Returns:** Object with `times` and `states` arrays

### `solve_ode_esdirk34(nstates, params, t_final, rhs_fn, jac_fn, init_fn, num_points?)`

Solve a stiff ODE system using ESDIRK34, a 4th order diagonally implicit Runge-Kutta method.

**Parameters:** Same as `solve_ode_bdf`

**Returns:** Object with `times` and `states` arrays

### `solve_lotka_volterra(t_final)`

Example function demonstrating the Lotka-Volterra (predator-prey) system.

**Parameters:**
- `t_final` (number): Final time to solve to

**Returns:** Object with `times` and `states` arrays

**Example:**
```javascript
const result = solve_lotka_volterra(40.0);
// result.states[0] contains [prey, predator] at each time point
```

## Examples

### Exponential Decay

```javascript
import init, { solve_ode } from './pkg/diffsol_wasm.js';

await init();

// dy/dt = -a*y, y(0) = 1
const a = 0.5;
const result = solve_ode(
    1,
    new Float64Array([a]),
    5.0,
    (x, p, t, y) => { y[0] = -p[0] * x[0]; },
    (p, t, y) => { y[0] = 1.0; },
    100  // 100 output points
);

// Verify against exact solution: y(t) = exp(-a*t)
const exact = Math.exp(-a * 5.0);
console.log('Computed:', result.states[result.states.length-1][0]);
console.log('Exact:', exact);
```

### Lotka-Volterra (Predator-Prey)

```javascript
import init, { solve_ode } from './pkg/diffsol_wasm.js';

await init();

// dx/dt = a*x - b*x*y
// dy/dt = c*x*y - d*y
const result = solve_ode(
    2,
    new Float64Array([2/3, 4/3, 1.0, 1.0]),  // a, b, c, d
    40.0,
    (x, p, t, y) => {
        y[0] = p[0] * x[0] - p[1] * x[0] * x[1];  // prey
        y[1] = p[2] * x[0] * x[1] - p[3] * x[1];  // predator
    },
    (p, t, y) => {
        y[0] = 1.0;  // initial prey
        y[1] = 1.0;  // initial predator
    },
    200
);
```

### Stiff ODE with BDF

```javascript
import init, { solve_ode_bdf } from './pkg/diffsol_wasm.js';

await init();

// Stiff system: dy/dt = -1000*(y - cos(t))
const lambda = 1000.0;
const result = solve_ode_bdf(
    1,
    new Float64Array([lambda]),
    1.0,
    (x, p, t, y) => {
        y[0] = p[0] * (Math.cos(t) - x[0]);
    },
    (x, p, t, v, y) => {
        y[0] = -p[0] * v[0];  // Jacobian: d/dy = -λ
    },
    (p, t, y) => {
        y[0] = 1.0;
    }
);
```

## Choosing the Right Solver

- **Non-stiff problems**: Use `solve_ode` (TSIT45 explicit Runge-Kutta)
- **Stiff problems**: Use `solve_ode_bdf`, `solve_ode_tr_bdf2`, or `solve_ode_esdirk34`
  - BDF: Variable order, best for very stiff problems
  - TR-BDF2: 2nd order, good balance of accuracy and efficiency
  - ESDIRK34: 4th order, higher accuracy for moderately stiff problems

## Output Points

The `num_points` parameter controls how many time points are returned:

- `null` or `undefined`: Returns all internal solver steps (may be many points)
- `number`: Returns exactly that many evenly spaced points from `t0` to `t_final`

```javascript
// Get all internal steps
const result1 = solve_ode(..., null);

// Get exactly 100 evenly spaced points
const result2 = solve_ode(..., 100);
```

## Error Handling

All functions return a `Result` type. In JavaScript, errors are thrown as exceptions:

```javascript
try {
    const result = solve_ode(...);
} catch (error) {
    console.error('ODE solver error:', error);
}
```

## Building

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

### Build Commands

```bash
# Build for web browsers
wasm-pack build --target web --out-dir pkg

# Build for Node.js
wasm-pack build --target nodejs --out-dir pkg

# Build for bundlers (webpack, etc.)
wasm-pack build --target bundler --out-dir pkg
```

## License

This project is licensed under the MIT License - see the [LICENSE.txt](../LICENSE.txt) file for details.

## Related Projects

- [diffsol](https://github.com/martinjrobins/diffsol) - Core ODE solver library in Rust
- [pydiffsol](https://github.com/alexallmont/pydiffsol) - Python bindings for diffsol


