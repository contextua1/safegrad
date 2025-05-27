use safegrad::Value;

fn main() {
    println!("SafeGrad: A Rust implementation of micrograd");

    // Demonstrate all operations: add, sub, mul, div, relu
    let a = Value::new(8.0, Some("a".to_string()));
    let b = Value::new(2.0, Some("b".to_string()));
    
    let add_result = Value::add(&a, &b, Some("a+b".to_string()));
    let sub_result = Value::sub(&a, &b, Some("a-b".to_string()));
    let mul_result = Value::mul(&add_result, &sub_result, Some("(a+b)*(a-b)".to_string()));
    let div_result = Value::div(&mul_result, &b, Some("result/b".to_string()));
    let final_result = Value::relu(&div_result, Some("relu(final)".to_string()));
    
    println!("a = {}, b = {}", Value::data(&a), Value::data(&b));
    println!("a+b = {}", Value::data(&add_result));
    println!("a-b = {}", Value::data(&sub_result));
    println!("(a+b)*(a-b) = {}", Value::data(&mul_result));
    println!("result/b = {}", Value::data(&div_result));
    println!("relu(final) = {}", Value::data(&final_result));

    // Backward pass
    Value::backward(&final_result);
    println!("\nGradients:");
    println!("∂a = {}", Value::grad(&a));
    println!("∂b = {}", Value::grad(&b));
    
    // Export graph
    let dot = Value::export_dot(&final_result);
    std::fs::write("graph.dot", &dot).expect("Failed to write graph.dot");
    println!("\nGraph exported to graph.dot");
}
