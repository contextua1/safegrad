use crate::{Value, ValueRef};
use rand::Rng;

#[derive(Debug, Clone)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Linear,
}

#[derive(Debug)]
pub struct Layer {
    pub weights: Vec<Vec<ValueRef>>,
    pub biases: Vec<ValueRef>,
    pub activation: Activation,
}

impl Layer {    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        
        let weight_std = match activation {
            Activation::Relu => (2.0 / input_size as f64).sqrt(), 
            Activation::Sigmoid | Activation::Tanh => (1.0 / input_size as f64).sqrt(),
            Activation::Linear => (1.0 / input_size as f64).sqrt(),
        };
        
        let mut weights = Vec::new();
        for i in 0..input_size {
            let mut weight_row = Vec::new();
            for j in 0..output_size {
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let weight_val = normal * weight_std;
                weight_row.push(Value::new(weight_val, Some(format!("w{}_{}", i, j))));
            }
            weights.push(weight_row);
        }
        
        let mut biases = Vec::new();
        for i in 0..output_size {
            biases.push(Value::new(0.0, Some(format!("b{}", i))));
        }
        
        Layer {
            weights,
            biases,
            activation,
        }
    }
    
    pub fn forward(&self, inputs: &[ValueRef]) -> Vec<ValueRef> {
        let mut outputs = Vec::new();
        
        for j in 0..self.biases.len() {
            let mut sum = self.biases[j].clone();
            
            for (i, input) in inputs.iter().enumerate() {
                let weighted = Value::mul(input, &self.weights[i][j], 
                    Some(format!("x{}*w{}_{}", i, i, j)));
                sum = Value::add(&sum, &weighted, Some(format!("sum_{}", j)));
            }
            
            let activated = match self.activation {
                Activation::Relu => Value::relu(&sum, Some(format!("relu_{}", j))),
                Activation::Sigmoid => Value::sigmoid(&sum, Some(format!("sigmoid_{}", j))),
                Activation::Tanh => Value::tanh(&sum, Some(format!("tanh_{}", j))),
                Activation::Linear => sum,
            };
            
            outputs.push(activated);
        }
        
        outputs
    }
    
    pub fn get_parameters(&self) -> Vec<ValueRef> {
        let mut params = Vec::new();
        
        for weight_row in &self.weights {
            for weight in weight_row {
                params.push(weight.clone());
            }
        }
        
        for bias in &self.biases {
            params.push(bias.clone());
        }
        
        params
    }
}

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(layer_sizes: &[usize], activations: &[Activation]) -> Self {
        assert_eq!(layer_sizes.len() - 1, activations.len(), 
                   "Number of activations must be one less than number of layer sizes");
        
        let mut layers = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(
                layer_sizes[i], 
                layer_sizes[i + 1], 
                activations[i].clone()
            ));
        }
        
        MLP { layers }
    }
    
    pub fn forward(&self, inputs: &[ValueRef]) -> Vec<ValueRef> {
        let mut current_inputs = inputs.to_vec();
        
        for layer in &self.layers {
            current_inputs = layer.forward(&current_inputs);
        }
        
        current_inputs
    }
    
    pub fn get_parameters(&self) -> Vec<ValueRef> {
        let mut params = Vec::new();
        
        for layer in &self.layers {
            params.extend(layer.get_parameters());
        }
        
        params
    }
    
    pub fn zero_grad(&self) {
        for param in self.get_parameters() {
            param.borrow_mut().grad = 0.0;
        }
    }
    
    pub fn update_parameters(&self, learning_rate: f64) {
        for param in self.get_parameters() {
            let mut param_ref = param.borrow_mut();
            param_ref.data -= learning_rate * param_ref.grad;
        }
    }
}

// Loss functions
pub fn mse_loss(predictions: &[ValueRef], targets: &[f64]) -> ValueRef {
    assert_eq!(predictions.len(), targets.len(), "Predictions and targets must have same length");
    
    let mut total_loss = Value::new(0.0, Some("loss_init".to_string()));
    
    for (i, (pred, &target)) in predictions.iter().zip(targets.iter()).enumerate() {
        let target_val = Value::new(target, Some(format!("target_{}", i)));
        let diff = Value::sub(pred, &target_val, Some(format!("diff_{}", i)));
        let squared = Value::mul(&diff, &diff, Some(format!("squared_{}", i)));
        total_loss = Value::add(&total_loss, &squared, Some(format!("loss_sum_{}", i)));
    }
    
    let n = Value::new(predictions.len() as f64, Some("n".to_string()));
    Value::div(&total_loss, &n, Some("mse_loss".to_string()))
}

pub fn binary_cross_entropy_loss(predictions: &[ValueRef], targets: &[f64]) -> ValueRef {
    use crate::Value;
    assert_eq!(predictions.len(), targets.len(), "Predictions and targets must have same length");
    
    let mut total_loss = Value::new(0.0, Some("bce_init".to_string()));
    
    for (i, (pred, &target)) in predictions.iter().zip(targets.iter()).enumerate() {
        let target_val = Value::new(target, Some(format!("target_{}", i)));
        
        let eps = 1e-7;
        let pred_data = Value::data(pred).max(eps).min(1.0 - eps);
        let pred_clamped = Value::new(pred_data, Some(format!("pred_clamped_{}", i)));
        
        let one_minus_pred = Value::sub(&Value::new(1.0, None), &pred_clamped, Some(format!("1-pred_{}", i)));
        let one_minus_target = Value::new(1.0 - target, Some(format!("1-target_{}", i)));
        
        let log_pred = Value::ln(&pred_clamped, Some(format!("log_pred_{}", i)));
        let log_one_minus_pred = Value::ln(&one_minus_pred, Some(format!("log_1-pred_{}", i)));
        
        let term1 = Value::mul(&target_val, &log_pred, Some(format!("y*log_p_{}", i)));
        let term2 = Value::mul(&one_minus_target, &log_one_minus_pred, Some(format!("(1-y)*log(1-p)_{}", i)));
        
        let loss_term = Value::sub(&Value::new(0.0, None), 
            &Value::add(&term1, &term2, Some(format!("bce_terms_{}", i))), 
            Some(format!("neg_bce_{}", i)));
        
        total_loss = Value::add(&total_loss, &loss_term, Some(format!("bce_sum_{}", i)));
    }
    
    let n = Value::new(predictions.len() as f64, Some("n".to_string()));
    Value::div(&total_loss, &n, Some("avg_bce_loss".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(2, 3, Activation::Relu);
        assert_eq!(layer.weights.len(), 2);
        assert_eq!(layer.weights[0].len(), 3);
        assert_eq!(layer.biases.len(), 3);
    }

    #[test]
    fn test_mlp_creation() {
        let mlp = MLP::new(&[2, 4, 1], &[Activation::Relu, Activation::Sigmoid]);
        assert_eq!(mlp.layers.len(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let mlp = MLP::new(&[2, 4, 1], &[Activation::Relu, Activation::Sigmoid]);
        let inputs = vec![
            Value::new(1.0, Some("x1".to_string())),
            Value::new(0.5, Some("x2".to_string())),
        ];
        
        let outputs = mlp.forward(&inputs);
        assert_eq!(outputs.len(), 1);
        
        // Output should be between 0 and 1 due to sigmoid
        let output_val = Value::data(&outputs[0]);
        assert!(output_val >= 0.0 && output_val <= 1.0);
    }    #[test]
    fn test_binary_cross_entropy_loss() {
        let pred1 = Value::new(0.9, Some("pred1".to_string())); // Close to 1
        let pred2 = Value::new(0.1, Some("pred2".to_string())); // Close to 0
        let predictions = vec![pred1, pred2];
        let targets = vec![1.0, 0.0]; // Perfect targets
        
        let loss = binary_cross_entropy_loss(&predictions, &targets);
        let loss_val = Value::data(&loss);
        
        assert!(loss_val < 0.5, "Loss should be small for good predictions, got {}", loss_val);
        assert!(loss_val > 0.0, "Loss should be positive");
    }

    #[test]
    fn test_mse_loss() {
        let pred1 = Value::new(0.8, Some("pred1".to_string()));
        let pred2 = Value::new(0.3, Some("pred2".to_string()));
        let predictions = vec![pred1, pred2];
        let targets = vec![1.0, 0.0];
        
        let loss = mse_loss(&predictions, &targets);
        let loss_val = Value::data(&loss);
        
        // Expected: ((0.8-1.0)^2 + (0.3-0.0)^2) / 2 = (0.04 + 0.09) / 2 = 0.065
        assert!((loss_val - 0.065).abs() < 1e-10);
    }
}
