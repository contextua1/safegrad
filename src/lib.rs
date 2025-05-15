use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

// A shared reference to a Value node for graph construction
pub type ValueRef = Rc<RefCell<Value>>;

// The core computational graph node
#[derive(Debug, Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub parents: Vec<ValueRef>,
    pub op: Option<String>,
    pub label: Option<String>,
}

impl Value {
    // Core functionality
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
                    v.borrow().parents[0].borrow_mut().grad += grad;
                    v.borrow().parents[1].borrow_mut().grad += grad;
                }
                Some("*") => {
                    let left = v.borrow().parents[0].clone();
                    let right = v.borrow().parents[1].clone();
                    left.borrow_mut().grad += grad * right.borrow().data;
                    right.borrow_mut().grad += grad * left.borrow().data;
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