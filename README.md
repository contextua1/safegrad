# SafeGrad

SafeGrad is a lightweight automatic differentiation engine implemented in Rust, inspired by micrograd. It provides a computational graph-based approach to building and differentiating mathematical expressions for machine learning applications.

## Functionality

SafeGrad enables the construction of computational graphs with automatic gradient computation through backpropagation. It currently supports basic operations like addition and multiplication with visualization capabilities for examining the underlying computation graph.

## Next Steps

1. **multi-node gradient handling**: implement proper gradient accumulation for nodes used by multiple children in the computational graph to ensure accurate gradient flow during backpropagation
   
2. **activation functions**: add essential activation functions like tanh, relu, and pow along with a few other useful mathematical operations to enable more complex neural network architectures
   
3. **additional operations**: implement division and subtraction operations to complete the core arithmetic functionality needed for building comprehensive computational graphs
   
4. **mlp implementation**: create a simple multi-layer perceptron using the safegrad engine to demonstrate forward and backward propagation with a practical neural network example
   
5. **loss functions**: incorporate common loss functions such as mse and cross-entropy to enable proper training optimization objectives for neural networks
   
6. **performance metrics**: add built-in performance measurement and benchmarking tools to monitor computational efficiency and identify optimization opportunities
   
7. **optimization algorithms**: implement standard optimization algorithms like sgd and adam to provide comprehensive training capabilities for neural network models

## License

MIT
