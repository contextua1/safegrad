use safegrad::Value;

fn main() {
    println!("SafeGrad: A Rust implementation of micrograd");

    // Toy example: build a small graph
    let a = Value::new(-2.0, Some("a".to_string()));
    let b = Value::new(3.0, Some("b".to_string()));
    let e = Value::add(&a, &b, Some("e =a+b".to_string()));
    let d = Value::mul(&a, &b, Some("d = a*b".to_string()));
    let f = Value::mul(&d, &e, Some("f = d*e".to_string()));
    println!("f = d*e = {}", Value::data(&f));

    // Backward pass
    Value::backward(&f);
    println!("Gradient of a: {}", Value::grad(&a));
    println!("Gradient of b: {}", Value::grad(&b));
    println!("Gradient of e: {}", Value::grad(&e));
    println!("Gradient of d: {}", Value::grad(&d));
    println!("Gradient of f: {}", Value::grad(&f));
    // Export graph to DOT format
    let dot = Value::export_dot(&f);
    std::fs::write("graph.dot", &dot).expect("Failed to write graph.dot");
    println!("Graph exported to graph.dot. You can render it with: dot -Tpng graph.dot -o graph.png");
}
