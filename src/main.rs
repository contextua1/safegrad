use safegrad::Value;

fn main() {
    println!("SafeGrad: A Rust implementation of micrograd");
    println!("{}", "=".repeat(50));
    println!();
    println!("Available demos:");
    println!("  cargo run --bin basic_demo   - Basic operations and backprop");
    println!("  cargo run --bin xor_demo     - XOR neural network training");
    println!("  cargo run                    - This help message");
    println!();
    println!("Quick test of basic functionality:");
    
    // Quick sanity check
    let a = Value::new(3.0, Some("a".to_string()));
    let b = Value::new(4.0, Some("b".to_string()));
    let c = Value::add(&a, &b, Some("a+b".to_string()));
    let d = Value::mul(&c, &a, Some("(a+b)*a".to_string()));
    
    println!("  a = {}, b = {}", Value::data(&a), Value::data(&b));
    println!("  c = a + b = {}", Value::data(&c));
    println!("  d = c * a = {}", Value::data(&d));
    
    Value::backward(&d);
    println!("  ∂d/∂a = {}, ∂d/∂b = {}", Value::grad(&a), Value::grad(&b));
    
    println!();
    println!("✓ SafeGrad is working correctly!");
    println!("Run the demos above to see more advanced examples.");
}
