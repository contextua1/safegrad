use safegrad::Value;

fn main() {
    println!("SafeGrad: Basic Operations Demo");
    println!("{}", "=".repeat(40));
    
    let a = Value::new(8.0, Some("a".to_string()));
    let b = Value::new(2.0, Some("b".to_string()));
    
    let add_result = Value::add(&a, &b, Some("a+b".to_string()));
    let sub_result = Value::sub(&a, &b, Some("a-b".to_string()));
    let mul_result = Value::mul(&add_result, &sub_result, Some("(a+b)*(a-b)".to_string()));
    let div_result = Value::div(&mul_result, &b, Some("result/b".to_string()));
    let relu_result = Value::relu(&div_result, Some("relu(final)".to_string()));
    let sigmoid_result = Value::sigmoid(&a, Some("sigmoid(a)".to_string()));
    let tanh_result = Value::tanh(&a, Some("tanh(a)".to_string()));
    
    println!("Forward pass:");
    println!("a = {}, b = {}", Value::data(&a), Value::data(&b));
    println!("a+b = {}", Value::data(&add_result));
    println!("a-b = {}", Value::data(&sub_result));
    println!("(a+b)*(a-b) = {}", Value::data(&mul_result));
    println!("result/b = {}", Value::data(&div_result));
    println!("relu(final) = {}", Value::data(&relu_result));
    println!("sigmoid(a) = {:.4}", Value::data(&sigmoid_result));
    println!("tanh(a) = {:.4}", Value::data(&tanh_result));

    println!("\nRunning backward pass...");
    Value::backward(&relu_result);
    
    println!("\nGradients:");
    println!("∂a = {}", Value::grad(&a));
    println!("∂b = {}", Value::grad(&b));
    
    let dot = Value::export_dot(&relu_result);
    std::fs::write("basic_operations_graph.dot", &dot).expect("Failed to write graph.dot");
    println!("\nComputation graph exported to basic_operations_graph.dot");
    
    println!("\n{}", "=".repeat(40));
    println!("Testing individual activation functions:");
    
    let negative_val = Value::new(-2.5, Some("negative".to_string()));
    let positive_val = Value::new(3.7, Some("positive".to_string()));
    
    let relu_neg = Value::relu(&negative_val, Some("relu(negative)".to_string()));
    let relu_pos = Value::relu(&positive_val, Some("relu(positive)".to_string()));
    
    println!("ReLU(-2.5) = {}", Value::data(&relu_neg));
    println!("ReLU(3.7) = {}", Value::data(&relu_pos));
    
    let small_val = Value::new(-5.0, Some("small".to_string()));
    let large_val = Value::new(5.0, Some("large".to_string()));
    
    let sigmoid_small = Value::sigmoid(&small_val, Some("sigmoid(small)".to_string()));
    let sigmoid_large = Value::sigmoid(&large_val, Some("sigmoid(large)".to_string()));
    
    println!("Sigmoid(-5.0) = {:.6}", Value::data(&sigmoid_small));
    println!("Sigmoid(5.0) = {:.6}", Value::data(&sigmoid_large));
    
    let tanh_small = Value::tanh(&small_val, Some("tanh(small)".to_string()));
    let tanh_large = Value::tanh(&large_val, Some("tanh(large)".to_string()));
    
    println!("Tanh(-5.0) = {:.6}", Value::data(&tanh_small));
    println!("Tanh(5.0) = {:.6}", Value::data(&tanh_large));
    
    println!("\nBasic operations demo completed!");
}
