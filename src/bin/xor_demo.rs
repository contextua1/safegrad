use safegrad::{Value, MLP, Activation, mse_loss};

fn main() {
    println!("SafeGrad: XOR Neural Network Demo");
    println!("{}", "=".repeat(40));
    
    // XOR dataset
    let xor_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    println!("XOR Truth Table:");
    for (inputs, targets) in &xor_data {
        println!("  [{:.0}, {:.0}] -> {:.0}", inputs[0], inputs[1], targets[0]);
    }
    println!();
    
    // Create MLP: 2 inputs -> 4 hidden (Sigmoid) -> 1 output (Sigmoid)
    // Using Sigmoid for hidden layer for smooth non-linear decision boundaries
    let mlp = MLP::new(&[2, 4, 1], &[Activation::Sigmoid, Activation::Sigmoid]);
    
    println!("Network Architecture: 2-4-1 (Sigmoid-Sigmoid)");
    println!("Total parameters: {}", mlp.get_parameters().len());
    println!("Starting training...\n");
    
    // Training loop
    let epochs = 5000;
    let learning_rate = 0.5;
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        
        // Training on all samples
        for (inputs, targets) in &xor_data {
            // Convert to ValueRef
            let input_values: Vec<_> = inputs.iter().enumerate()
                .map(|(i, &x)| Value::new(x, Some(format!("x{}", i))))
                .collect();
            
            // Forward pass
            let outputs = mlp.forward(&input_values);
            
            // Compute loss
            let loss = mse_loss(&outputs, targets);
            total_loss += Value::data(&loss);
            
            // Zero gradients
            mlp.zero_grad();
            
            // Backward pass
            Value::backward(&loss);
            
            // Update parameters
            mlp.update_parameters(learning_rate);
        }
        
        let avg_loss = total_loss / xor_data.len() as f64;
        
        if epoch % 500 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, avg_loss);
        }
        
        // Early stopping if loss is very low
        if avg_loss < 0.0001 {
            println!("Converged at epoch {} with loss {:.6}", epoch, avg_loss);
            break;
        }
    }
    
    println!("\n{}", "=".repeat(40));
    println!("Training completed! Testing network...\n");
    
    println!("Final XOR Test Results:");
    println!("Input    | Target | Prediction | Correct?");
    println!("---------+--------+------------+---------");
    
    let mut correct_predictions = 0;
    for (inputs, targets) in &xor_data {
        let input_values: Vec<_> = inputs.iter().enumerate()
            .map(|(i, &x)| Value::new(x, Some(format!("test_x{}", i))))
            .collect();
        
        let outputs = mlp.forward(&input_values);
        let prediction = Value::data(&outputs[0]);
        
        // Check if prediction is correct (threshold at 0.5)
        let predicted_class = if prediction > 0.5 { 1.0 } else { 0.0 };
        let is_correct = (predicted_class - targets[0]).abs() < 0.1;
        if is_correct { correct_predictions += 1; }
        
        println!("[{:.0}, {:.0}]    | {:.0}      | {:.4}     | {}", 
                 inputs[0], inputs[1], targets[0], prediction,
                 if is_correct { "✓" } else { "✗" });
    }
    
    println!("---------+--------+------------+---------");
    println!("Accuracy: {}/{} ({:.1}%)", 
             correct_predictions, xor_data.len(), 
             (correct_predictions as f64 / xor_data.len() as f64) * 100.0);
    
    // Export final computation graph for visualization
    let test_input = vec![
        Value::new(1.0, Some("test_x0".to_string())),
        Value::new(0.0, Some("test_x1".to_string())),
    ];
    let final_output = mlp.forward(&test_input);
    let dot = Value::export_dot(&final_output[0]);
    std::fs::write("xor_network_graph.dot", &dot).expect("Failed to write graph.dot");
    println!("\nComputation graph exported to xor_network_graph.dot");
    
    // Show some network insights
    println!("\n{}", "=".repeat(40));
    println!("Network Analysis:");
    
    // Test decision boundary points
    let test_points = vec![
        (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75),
        (0.5, 0.5), (0.0, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0)
    ];
    
    println!("Decision boundary exploration:");
    for (x, y) in test_points {
        let test_input = vec![
            Value::new(x, Some("boundary_x".to_string())),
            Value::new(y, Some("boundary_y".to_string())),
        ];
        let output = mlp.forward(&test_input);
        let prediction = Value::data(&output[0]);
        println!("  ({:.2}, {:.2}) -> {:.4}", x, y, prediction);
    }
    
    println!("\nXOR demo completed!");
}
