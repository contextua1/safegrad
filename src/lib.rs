use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

pub mod mlp;
pub use mlp::*;

pub type ValueRef = Rc<RefCell<Value>>;

#[derive(Debug, Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub parents: Vec<ValueRef>,
    pub op: Option<String>,
    pub label: Option<String>,
}

impl Value {
    pub fn new(data: f64, label: Option<String>) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            parents: vec![],
            op: None,
            label,
        }))
    }
    
    pub fn data(val: &ValueRef) -> f64 {
        val.borrow().data
    }
    
    pub fn grad(val: &ValueRef) -> f64 {
        val.borrow().grad
    }
    
    // Operations
    pub fn add(a: &ValueRef, b: &ValueRef, label: Option<String>) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: a.borrow().data + b.borrow().data,
            grad: 0.0,
            parents: vec![a.clone(), b.clone()],
            op: Some("+".to_string()),
            label,
        }))
    }
    
    pub fn mul(a: &ValueRef, b: &ValueRef, label: Option<String>) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: a.borrow().data * b.borrow().data,
            grad: 0.0,
            parents: vec![a.clone(), b.clone()],
            op: Some("*".to_string()),
            label,
        }))
    }
    
    pub fn relu(a: &ValueRef, label: Option<String>) -> ValueRef {
        let data = a.borrow().data.max(0.0);
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            parents: vec![a.clone()],
            op: Some("relu".to_string()),
            label,
        }))
    }
    
    pub fn sigmoid(a: &ValueRef, label: Option<String>) -> ValueRef {
        let data = 1.0 / (1.0 + (-a.borrow().data).exp());
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            parents: vec![a.clone()],
            op: Some("sigmoid".to_string()),
            label,
        }))
    }
    
    pub fn tanh(a: &ValueRef, label: Option<String>) -> ValueRef {
        let data = a.borrow().data.tanh();
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            parents: vec![a.clone()],
            op: Some("tanh".to_string()),
            label,
        }))
    }
    
    pub fn sub(a: &ValueRef, b: &ValueRef, label: Option<String>) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: a.borrow().data - b.borrow().data,
            grad: 0.0,
            parents: vec![a.clone(), b.clone()],
            op: Some("-".to_string()),
            label,
        }))
    }
    
    pub fn div(a: &ValueRef, b: &ValueRef, label: Option<String>) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: a.borrow().data / b.borrow().data,
            grad: 0.0,
            parents: vec![a.clone(), b.clone()],
            op: Some("/".to_string()),
            label,
        }))
    }
    
    pub fn exp(a: &ValueRef, label: Option<String>) -> ValueRef {
        let data = a.borrow().data.exp();
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            parents: vec![a.clone()],
            op: Some("exp".to_string()),
            label,
        }))
    }
    
    pub fn ln(a: &ValueRef, label: Option<String>) -> ValueRef {
        let data = a.borrow().data.ln();
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            parents: vec![a.clone()],
            op: Some("ln".to_string()),
            label,
        }))
    }
    
    pub fn pow(a: &ValueRef, exponent: f64, label: Option<String>) -> ValueRef {
        let data = a.borrow().data.powf(exponent);
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            parents: vec![a.clone()],
            op: Some(format!("pow_{}", exponent)),
            label,
        }))
    }

    // Backpropagation
    pub fn backward(val: &ValueRef) {
        // Topological sort
        let mut topo = vec![];
        let mut visited = std::collections::HashSet::new();
        
        fn build_topo(v: &ValueRef, topo: &mut Vec<ValueRef>, visited: &mut std::collections::HashSet<usize>) {
            let addr = Rc::as_ptr(v) as usize;
            if !visited.contains(&addr) {
                visited.insert(addr);
                for p in &v.borrow().parents {
                    build_topo(p, topo, visited);
                }
                topo.push(Rc::clone(v));
            }
        }
        
        build_topo(val, &mut topo, &mut visited);
        
        // Set output node grad to 1.0
        val.borrow_mut().grad = 1.0;
        
        // Traverse in reverse topological order
        for v in topo.into_iter().rev() {
            let grad = v.borrow().grad;
            let op = v.borrow().op.clone();
              match op.as_deref() {
                Some("+") => {
                    let left = v.borrow().parents[0].clone();
                    let right = v.borrow().parents[1].clone();
                    left.borrow_mut().grad += grad;
                    right.borrow_mut().grad += grad;
                }                Some("*") => {
                    let left = v.borrow().parents[0].clone();
                    let right = v.borrow().parents[1].clone();
                    let left_data = left.borrow().data;
                    let right_data = right.borrow().data;
                    left.borrow_mut().grad += grad * right_data;
                    right.borrow_mut().grad += grad * left_data;
                }Some("relu") => {
                    let parent = v.borrow().parents[0].clone();
                    // Derivative of ReLU: 1 if input > 0, else 0
                    let relu_grad = if parent.borrow().data > 0.0 { 1.0 } else { 0.0 };
                    parent.borrow_mut().grad += grad * relu_grad;
                }
                Some("sigmoid") => {
                    let parent = v.borrow().parents[0].clone();
                    let sigmoid_val = v.borrow().data;
                    // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
                    let sigmoid_grad = sigmoid_val * (1.0 - sigmoid_val);
                    parent.borrow_mut().grad += grad * sigmoid_grad;
                }
                Some("tanh") => {
                    let parent = v.borrow().parents[0].clone();
                    let tanh_val = v.borrow().data;
                    // Derivative of tanh: 1 - tanh²(x)
                    let tanh_grad = 1.0 - tanh_val * tanh_val;
                    parent.borrow_mut().grad += grad * tanh_grad;
                }                Some("-") => {
                    let left = v.borrow().parents[0].clone();
                    let right = v.borrow().parents[1].clone();
                    left.borrow_mut().grad += grad;
                    right.borrow_mut().grad -= grad;
                }                Some("/") => {
                    let numerator = v.borrow().parents[0].clone();
                    let denominator = v.borrow().parents[1].clone();
                    let denom_data = denominator.borrow().data;
                    let numer_data = numerator.borrow().data;
                    
                    numerator.borrow_mut().grad += grad / denom_data;
                    denominator.borrow_mut().grad -= grad * numer_data / (denom_data * denom_data);
                }                Some("exp") => {
                    let parent = v.borrow().parents[0].clone();
                    let exp_val = v.borrow().data;
                    // Derivative of exp(x) is exp(x)
                    parent.borrow_mut().grad += grad * exp_val;
                }
                Some("ln") => {
                    let parent = v.borrow().parents[0].clone();
                    let parent_data = parent.borrow().data;
                    // Derivative of ln(x) is 1/x
                    parent.borrow_mut().grad += grad / parent_data;
                }
                Some(op) if op.starts_with("pow_") => {
                    let parent = v.borrow().parents[0].clone();
                    let parent_data = parent.borrow().data;
                    let exponent: f64 = op.strip_prefix("pow_").unwrap().parse().unwrap();
                    // Derivative of x^n is n * x^(n-1)
                    parent.borrow_mut().grad += grad * exponent * parent_data.powf(exponent - 1.0);
                }
                _ => {}
            }
        }
    }
}

// Graph visualization code
use petgraph::dot::{Dot, Config};
use petgraph::graph::{Graph, NodeIndex};

impl Value {
    pub fn export_dot(val: &ValueRef) -> String {
        let mut graph = Graph::<String, String>::new();
        let mut node_map = HashMap::new();
        let mut visited = std::collections::HashSet::new();

        fn build_graph(
            v: &ValueRef,
            graph: &mut Graph<String, String>,
            node_map: &mut HashMap<usize, NodeIndex>,
            visited: &mut std::collections::HashSet<usize>,
        ) -> NodeIndex {
            let addr = Rc::as_ptr(v) as usize;
            if let Some(&idx) = node_map.get(&addr) {
                return idx;
            }
            
            let label = {
                let vb = v.borrow();
                let node_label = match &vb.label {
                    Some(l) => format!("{}\ndata: {:.3}\ngrad: {:.3}", l, vb.data, vb.grad),
                    None => format!("data: {:.3}\ngrad: {:.3}", vb.data, vb.grad),
                };
                node_label
            };
            
            let idx = graph.add_node(label);
            node_map.insert(addr, idx);
            
            if !visited.contains(&addr) {
                visited.insert(addr);
                let vb = v.borrow();
                
                for p in vb.parents.iter() {
                    let parent_idx = build_graph(p, graph, node_map, visited);
                    let edge_label = vb.op.clone().unwrap_or_else(|| "leaf".to_string());
                    graph.add_edge(parent_idx, idx, edge_label);
                }
            }
            
            idx
        }
        
        build_graph(val, &mut graph, &mut node_map, &mut visited);
        format!("{}", Dot::with_config(&graph, &[Config::EdgeNoLabel]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_positive() {
        let a = Value::new(2.0, Some("a".to_string()));
        let relu_a = Value::relu(&a, Some("relu(a)".to_string()));
        
        assert_eq!(Value::data(&relu_a), 2.0);
        
        Value::backward(&relu_a);
        assert_eq!(Value::grad(&a), 1.0);
        assert_eq!(Value::grad(&relu_a), 1.0);
    }

    #[test]
    fn test_relu_negative() {
        let a = Value::new(-3.0, Some("a".to_string()));
        let relu_a = Value::relu(&a, Some("relu(a)".to_string()));
        
        assert_eq!(Value::data(&relu_a), 0.0);
        
        Value::backward(&relu_a);
        assert_eq!(Value::grad(&a), 0.0);
        assert_eq!(Value::grad(&relu_a), 1.0);
    }

    #[test]
    fn test_relu_zero() {
        let a = Value::new(0.0, Some("a".to_string()));
        let relu_a = Value::relu(&a, Some("relu(a)".to_string()));
        
        assert_eq!(Value::data(&relu_a), 0.0);
        
        Value::backward(&relu_a);
        assert_eq!(Value::grad(&a), 0.0);
        assert_eq!(Value::grad(&relu_a), 1.0);
    }

    #[test]
    fn test_sub() {
        let a = Value::new(5.0, Some("a".to_string()));
        let b = Value::new(3.0, Some("b".to_string()));
        let c = Value::sub(&a, &b, Some("a - b".to_string()));
        
        assert_eq!(Value::data(&c), 2.0);
        
        Value::backward(&c);
        assert_eq!(Value::grad(&a), 1.0);
        assert_eq!(Value::grad(&b), -1.0);
    }    #[test]
    fn test_div() {
        let a = Value::new(6.0, Some("a".to_string()));
        let b = Value::new(2.0, Some("b".to_string()));
        let c = Value::div(&a, &b, Some("a / b".to_string()));
        
        assert_eq!(Value::data(&c), 3.0);
        
        Value::backward(&c);
        // For c = a/b:
        // ∂c/∂a = 1/b = 1/2 = 0.5
        // ∂c/∂b = -a/b² = -6/4 = -1.5
        assert_eq!(Value::grad(&a), 0.5);
        assert_eq!(Value::grad(&b), -1.5);
    }

    #[test]
    fn test_exp() {
        let a = Value::new(1.0, Some("a".to_string()));
        let exp_a = Value::exp(&a, Some("exp(a)".to_string()));
        
        // exp(1) ≈ 2.718
        assert!((Value::data(&exp_a) - std::f64::consts::E).abs() < 1e-10);
        
        Value::backward(&exp_a);
        // Derivative of exp(x) is exp(x)
        assert!((Value::grad(&a) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_ln() {
        let a = Value::new(std::f64::consts::E, Some("a".to_string()));
        let ln_a = Value::ln(&a, Some("ln(a)".to_string()));
        
        // ln(e) = 1
        assert!((Value::data(&ln_a) - 1.0).abs() < 1e-10);
        
        Value::backward(&ln_a);
        // Derivative of ln(x) is 1/x
        assert!((Value::grad(&a) - 1.0/std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_pow() {
        let a = Value::new(2.0, Some("a".to_string()));
        let pow_a = Value::pow(&a, 3.0, Some("a^3".to_string()));
        
        // 2^3 = 8
        assert_eq!(Value::data(&pow_a), 8.0);
        
        Value::backward(&pow_a);
        // Derivative of x^3 is 3*x^2 = 3*4 = 12
        assert_eq!(Value::grad(&a), 12.0);
    }
}