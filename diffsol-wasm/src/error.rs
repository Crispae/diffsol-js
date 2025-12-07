//! Error handling utilities for WASM bindings.

use wasm_bindgen::prelude::*;

/// Convert any error type to a JsValue for JavaScript consumption.
pub fn to_js_error<E: std::fmt::Display>(err: E) -> JsValue {
    JsValue::from_str(&err.to_string())
}
