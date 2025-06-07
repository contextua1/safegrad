use safegrad::{Value, MLP, Activation, mse_loss};
use rand::Rng;

fn main() {
    println!("SafeGrad: Sine Function Approximation Demo");
    println!("{}", "=".repeat(50));
      // Generate training data: f(x) = sin(x) for x in [0, 2π]
    let mut rng = rand::thread_rng();
    let n_samples = 200; // More samples for better coverage
    let mut training_data = Vec::new();
    
    println!("Generating {} training samples for f(x) = sin(x)", n_samples);
    
    // Mix random samples with evenly spaced samples for better coverage
    for i in 0..n_samples {
        let x = if i < n_samples / 2 {
            // First half: evenly spaced points
            (i as f64) * 2.0 * std::f64::consts::PI / (n_samples / 2) as f64
        } else {
            // Second half: random points
            rng.gen::<f64>() * 2.0 * std::f64::consts::PI
        };
        let y = x.sin(); // Target: sin(x)
        training_data.push((vec![x], vec![y]));
    }
    
    // Create MLP: 1 input -> 16 hidden (Tanh) -> 8 hidden (Tanh) -> 1 output (Linear)
    // Tanh is good for smooth function approximation, linear output for regression
    let mlp = MLP::new(&[1, 16, 8, 1], &[Activation::Tanh, Activation::Tanh, Activation::Linear]);
    
    println!("Network Architecture: 1-16-8-1 (Tanh-Tanh-Linear)");
    println!("Total parameters: {}", mlp.get_parameters().len());
    println!("Target function: f(x) = sin(x) for x ∈ [0, 2π]");
    println!("Starting training...\n");
    
    // Training loop
    let epochs = 10000;
    let learning_rate = 0.01; // Lower learning rate for stability
    let mut best_loss = f64::INFINITY;
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        
        // Shuffle training data each epoch for better convergence
        let mut epoch_data = training_data.clone();
        for i in (1..epoch_data.len()).rev() {
            let j = rng.gen_range(0..=i);
            epoch_data.swap(i, j);
        }
        
        // Training on all samples
        for (inputs, targets) in &epoch_data {
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
        
        let avg_loss = total_loss / training_data.len() as f64;
        
        if avg_loss < best_loss {
            best_loss = avg_loss;
        }
        
        if epoch % 1000 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, avg_loss);
        }
        
        // Early stopping if loss is very low
        if avg_loss < 0.00001 {
            println!("Converged at epoch {} with loss {:.6}", epoch, avg_loss);
            break;
        }
    }
    
    println!("\n{}", "=".repeat(50));
    println!("Training completed! Testing network...\n");
    
    // Test on regular grid points
    let test_points = 20;
    println!("Testing on {} evenly spaced points:", test_points);
    println!("Input (x)  | Target sin(x) | Prediction | Error");
    println!("-----------+---------------+------------+-------");
    
    let mut total_test_error = 0.0;
    let mut max_error = 0.0;
    
    for i in 0..test_points {
        let x = (i as f64) * 2.0 * std::f64::consts::PI / (test_points - 1) as f64;
        let target = x.sin();
        
        let input_values = vec![Value::new(x, Some("test_x".to_string()))];
        let outputs = mlp.forward(&input_values);
        let prediction = Value::data(&outputs[0]);
          let error = (prediction - target).abs();
        total_test_error += error;
        if error > max_error { max_error = error; }
        
        println!("{:8.4}   | {:11.4}   | {:8.4}   | {:5.4}", 
                 x, target, prediction, error);
    }
    
    let avg_test_error = total_test_error / test_points as f64;
    
    println!("-----------+---------------+------------+-------");
    println!("Average absolute error: {:.6}", avg_test_error);
    println!("Maximum absolute error: {:.6}", max_error);
    println!("Best training loss: {:.6}", best_loss);
    
    println!("\n{}", "=".repeat(50));
    println!("Testing on specific mathematical points:");
    
    let special_points = vec![
        (0.0, "0"),
        (std::f64::consts::PI / 6.0, "π/6"),
        (std::f64::consts::PI / 4.0, "π/4"),
        (std::f64::consts::PI / 3.0, "π/3"),
        (std::f64::consts::PI / 2.0, "π/2"),
        (std::f64::consts::PI, "π"),
        (3.0 * std::f64::consts::PI / 2.0, "3π/2"),
        (2.0 * std::f64::consts::PI, "2π"),
    ];
    
    for (x, label) in special_points {
        let target = x.sin();
        let input_values = vec![Value::new(x, Some("special_x".to_string()))];
        let outputs = mlp.forward(&input_values);
        let prediction = Value::data(&outputs[0]);
        let error = (prediction - target).abs();
        
        println!("{:>4}: sin({:6.4}) = {:8.4}, predicted = {:8.4}, error = {:6.4}", 
                 label, x, target, prediction, error);
    }
    
    // Export computation graph for one test case
    let test_input = vec![Value::new(std::f64::consts::PI / 4.0, Some("test_x_π/4".to_string()))];
    let final_output = mlp.forward(&test_input);
    let dot = Value::export_dot(&final_output[0]);
    std::fs::write("sin_network_graph.dot", &dot).expect("Failed to write sin_network_graph.dot");
    println!("\nComputation graph exported to sin_network_graph.dot");
    
    // Function analysis
    println!("\n{}", "=".repeat(50));
    println!("Function Approximation Analysis:");
    
    // Test smoothness by checking derivative approximation
    let h = 0.01; // Small step for numerical derivative
    println!("Checking derivative approximation (should be close to cos(x)):");
    
    for i in 0..5 {
        let x = i as f64 * std::f64::consts::PI / 2.0;
        let target_derivative = x.cos();
        
        // Numerical derivative: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        let x_plus = Value::new(x + h, Some("x_plus".to_string()));
        let x_minus = Value::new(x - h, Some("x_minus".to_string()));
        
        let f_plus = mlp.forward(&vec![x_plus]);
        let f_minus = mlp.forward(&vec![x_minus]);
        
        let numerical_derivative = (Value::data(&f_plus[0]) - Value::data(&f_minus[0])) / (2.0 * h);
        let derivative_error = (numerical_derivative - target_derivative).abs();
        
        println!("x = {:6.3}: cos(x) = {:7.4}, f'(x) ≈ {:7.4}, error = {:6.4}", 
                 x, target_derivative, numerical_derivative, derivative_error);
    }
    
    println!("\nSine approximation demo completed!");
    println!("The network has learned to approximate sin(x) with high accuracy!");
}
