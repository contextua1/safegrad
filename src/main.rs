use safegrad::Value;

fn main() {
    println!("SafeGrad: A Rust implementation of micrograd");

    // Toy example: build a small graph
    let a = Value::new(2.0, Some("a".to_string()));
    let b = Value::new(-3.0, Some("b".to_string()));
    let c = Value::add(
        &a, 
        &Value::mul(&b, &a, Some("b*a".to_string())), 
        Some("c = a+(b*a)".to_string())
    );
    let d = Value::mul(&c, &b, Some("d = c*b".to_string()));

    println!("d = b*a + b*(b * a) = {}", Value::data(&d));

    // Backward pass
    Value::backward(&d);
    println!("Gradient of a: {}", Value::grad(&a));
    println!("Gradient of b: {}", Value::grad(&b));
    println!("Gradient of c: {}", Value::grad(&c));
    println!("Gradient of d: {}", Value::grad(&c));

    // Export graph to DOT format
    let dot = Value::export_dot(&d);
    std::fs::write("graph.dot", &dot).expect("Failed to write graph.dot");
    println!("Graph exported to graph.dot. You can render it with: dot -Tpng graph.dot -o graph.png");
}
