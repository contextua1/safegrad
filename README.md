# SafeGrad

SafeGrad is an automatic differentiation engine meticulously crafted in Rust and designed to unravel the core mechanics of deep learning. It offers a hands-on approach to understanding fundamental concepts such as computational graphs, the intricacies of backpropagation, and the core principles of automatic differentiation that power modern neural networks.

Leveraging the strengths of Rust, SafeGrad ensures memory safety without a garbage collector, offers the potential for high performance, and utilizes elegant modern language features. This makes it not only a robust learning platform but also provides a glimpse into the construction of efficient and reliable machine learning tools.

Inspired by Andrej Karpathy's micrograd, SafeGrad embarks on a similar journey of discovery while carving its own path in the Rust ecosystem. Built with the learning process at its core, SafeGrad encourages exploration and a deeper appreciation for the intricate interplay of mathematics and code that underpins intelligent systems. Whether you're a student taking your first steps into neural networks or a developer curious about the inner workings of autograd engines, SafeGrad offers a transparent and engaging experience.

## Functionality

SafeGrad empowers users to define and compute complex mathematical expressions by constructing dynamic computational graphs, offering insight into how these systems function internally. Here's a breakdown of its core functionality:

*   **Computational Graph Construction:** At its heart, SafeGrad allows you to build expressions as a graph of interconnected nodes. Each node, represented by a `Value`, holds a scalar floating-point number (`data`) and its corresponding gradient (`grad`).

*   **Supported Operations:** Currently, SafeGrad supports fundamental arithmetic operations:
    *   **Addition (`+`):** Combines two `Value` nodes.
    *   **Multiplication (`*`):** Multiplies two `Value` nodes.
    These operations form the building blocks for more complex expressions.

*   **Graph Representation in Rust (`Rc<RefCell<Value>>`):**
    Nodes in the computational graph are managed using `Rc<RefCell<Value>>`. This idiomatic Rust pattern enables:
    *   `Rc` (Reference Counting): Allows multiple nodes to share ownership of their parent nodes. This is essential as a single node can be an input to several subsequent operations in the graph.
    *   `RefCell` (Interior Mutability): Provides a mechanism to modify the `grad` field of a node during backpropagation, even when the node is shared via `Rc`. This is crucial for updating gradients in place as the algorithm traverses the graph.

*   **Automatic Gradient Computation (Backpropagation):**
    SafeGrad automates the calculation of gradients using the backpropagation algorithm. Starting from an output node:
    1.  The gradient of the output node with respect to itself is set to 1.0.
    2.  The graph is traversed in reverse topological order (from outputs back to inputs).
    3.  For each node, the chain rule of calculus is applied based on the operation that created it. This process meticulously distributes the gradient from the output back through the graph, ensuring each node receives the correct gradient value reflecting its precise influence on the final output.

*   **Graph Visualization:**
    To aid in understanding and debugging, SafeGrad includes a utility to export the computational graph into the DOT language format. This allows for easy visualization of the graph's structure, node values, gradients, and the operations connecting them, using tools like Graphviz.

## Getting Started

This section provides a concise guide to get you started with SafeGrad, from building a simple computational graph to visualizing it.

SafeGrad is designed to be straightforward to use. You primarily interact with the `Value` type from the `safegrad` crate, which represents a node in the computational graph.

### 1. Example: Building a Simple Expression

Here's a basic example of how to use SafeGrad. This code defines a simple mathematical expression `f = (a*b) * (a+b)`, computes its value, performs a backward pass to calculate gradients, and exports the graph.

This example is also the default program run by `cargo run` in this project.

```rust
use safegrad::Value;

fn main() {
    // Create input values with labels for clarity in the graph
    let a = Value::new(-2.0, Some("a".to_string()));
    let b = Value::new(3.0, Some("b".to_string()));

    // Perform operations:
    // e = a + b
    let e = Value::add(&a, &b, Some("e = a+b".to_string()));
    // d = a * b
    let d = Value::mul(&a, &b, Some("d = a*b".to_string()));
    // f = d * e
    let f = Value::mul(&d, &e, Some("f = d*e".to_string()));

    println!("Value of f: {}", Value::data(&f)); // Output the result of the computation

    // Perform the backward pass to compute gradients for all nodes
    Value::backward(&f);

    // Export the computational graph to a .dot file
    let dot_representation = Value::export_dot(&f);
    std::fs::write("graph.dot", &dot_representation)
        .expect("Failed to write graph.dot");

    println!("Graph exported to graph.dot");
    println!("Run 'dot -Tpng graph.dot -o graph.png' to generate a visual graph.");
}
```

### 2. Running the Example

To run this example:

1.  Ensure you have Rust and Cargo installed on your system. If not, please visit [rust-lang.org](https://www.rust-lang.org/tools/install) for installation instructions.
2.  Clone the SafeGrad repository (if you haven't already):
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL of the SafeGrad repository
    cd safegrad
    ```
3.  Execute the program using Cargo:
    ```bash
    cargo run
    ```
    This command will compile and run the `main.rs` file. You should see output similar to this (the exact gradient values from `main.rs` are included here for completeness):
    ```
    Value of f: -6.0
    Gradient of a: 1.000
    Gradient of b: -8.000
    Gradient of e: -6.000
    Gradient of d: 1.000
    Gradient of f: 1.000
    Graph exported to graph.dot
    Run 'dot -Tpng graph.dot -o graph.png' to generate a visual graph.
    ```

### 3. Visualizing the Computational Graph

After running the example, a file named `graph.dot` will be created in the root directory of the project. This file contains the DOT language representation of the computational graph.

To visualize this graph:

1.  Install Graphviz, which provides the `dot` command-line tool. You can find installation instructions at [graphviz.org](https://graphviz.org/download/).
2.  Open your terminal in the project's root directory and run the following command:
    ```bash
    dot -Tpng graph.dot -o graph.png
    ```
    This will generate a PNG image file named `graph.png` showing the structure of the computation, including the values at each node and their gradients after the backward pass.

## Future Enhancements

SafeGrad is an actively evolving project with an ambitious roadmap focused on expanding its capabilities and exploring sophisticated features. We aim to transform it into a more comprehensive and powerful automatic differentiation engine, while retaining its educational clarity.

### Core Functionality Expansion

*   **Robust Multi-Node Gradient Handling:**
    Currently, gradient accumulation for nodes with multiple children in complex computational graphs requires careful consideration. Future work will focus on implementing automatic and robust gradient accumulation (e.g., ensuring `grad` correctly sums contributions from all child nodes) to guarantee correctness and simplify the construction of intricate models. This is crucial for accurate backpropagation in non-trivial network architectures.

*   **Expanded Set of Operations:**
    To build versatile neural networks, a richer set of operations is essential:
    *   **Basic Arithmetic:** Complete the set with reliable Division and Subtraction operations.
    *   **Activation Functions:** Introduce a comprehensive suite of activation functions critical for neural network layers, including Sigmoid, Tanh, ReLU (Rectified Linear Unit) and its variants (Leaky ReLU, ELU), Softmax (for output layers in classification), and potentially others like Pow and Exp.
    *   **Matrix Operations:** This is a pivotal area for significant enhancement, forming the backbone of modern neural networks. Implement foundational matrix operations such as matrix multiplication (dot product), element-wise operations on matrices, and transposition. These are the bedrock of efficient neural network layer implementations (e.g., dense layers, convolutional layers).

*   **Advanced Layer Implementations & MLP Demonstration:**
    Using the expanded operations, we will develop a simple Multi-Layer Perceptron (MLP). This will serve as a practical demonstration of SafeGrad's ability to define, forward-propagate, and backpropagate through a basic neural network, solidifying its use case for educational deep learning.

*   **Diverse Loss Functions:**
    To train models for various tasks, we'll implement a range of standard loss functions. Beyond Mean Squared Error (MSE), this will include Cross-Entropy Loss (for classification tasks), Binary Cross-Entropy, and potentially others, allowing users to define appropriate training objectives.

### Performance and Advanced Features

*   **Performance Optimization & Metrics:**
    *   **Benchmarking:** Implement built-in tools for measuring performance, such as time per forward/backward pass and memory usage per node or graph.
    *   **Comparative Analysis:** Conduct a comparative benchmark against the original `micrograd` (Python) to quantify the performance benefits of the Rust implementation. Extend this to other simple Rust-based autograd engines if available.

*   **GPU Acceleration:**
    To dramatically speed up computations for larger models, we will investigate leveraging GPU capabilities. This could involve using existing Rust crates like `ndarray` with GPU backends (e.g., via `ndarray-cuda` or `ndarray-opencl`) or, for a more challenging endeavor, exploring the development of custom CUDA or OpenCL kernels for core operations.

*   **Enhanced Optimization Algorithms:**
    Expand the suite of optimization algorithms beyond basic Stochastic Gradient Descent (SGD). Implement popular and effective optimizers such as Adam, Adagrad, RMSprop, and AdamW to provide users with more sophisticated tools for model training.

*   **Support for Dynamic Graphs (Exploratory):**
    While currently focused on static graph definition, we will explore the potential for supporting dynamic graph creation (define-by-run), similar to PyTorch. This would offer greater flexibility in model architecture, especially for models with control flow dependent on input data (e.g., RNNs with variable sequence lengths).

*   **Higher-Order Differentiation (Advanced Challenge):**
    Investigate the complex implementation of higher-order differentiation, allowing the computation of gradients of gradients (e.g., Hessians or Hessian-vector products). This advanced feature opens doors to more sophisticated optimization techniques and model analysis.

*   **Model Persistence (Serialization/Deserialization):**
    Implement mechanisms to save and load the computational graph structures, including the learned parameters (weights and biases). This is essential for saving trained models, resuming training, and deploying models.

*   **Rich Integration with the Rust Ecosystem:**
    Foster integration with the broader Rust machine learning ecosystem. This could involve compatibility with data loading libraries (e.g., for CSV, image data), preprocessing tools, or even frameworks for model serving and deployment, making SafeGrad a more versatile component in a larger Rust-based ML pipeline.

*   **Advanced Graph Visualization:**
    Enhance the current DOT-based graph visualization. Future improvements could include more detailed node information (e.g., shapes for tensor-valued nodes if implemented), interactive graph exploration tools, or better visual distinction for different types of operations and data flow.

## Contributing

We warmly welcome contributions to SafeGrad! As an educational project with ambitious goals, there is always room for new features, refined implementations, and performance enhancements. Your help can make SafeGrad an even better tool for learning and experimentation in the realm of automatic differentiation and neural networks.

### How You Can Help

A great place to start is by looking at our "Future Enhancements" section. Many of the listed items represent concrete areas where contributions would be highly valuable. Whether you're interested in:

*   Implementing new mathematical operations or activation functions.
*   Expanding our set of loss functions or optimization algorithms.
*   Helping to build out MLP examples or other neural network structures.
*   Exploring performance optimizations or benchmarking.
*   Improving documentation or adding more examples.
*   Tackling more advanced and challenging features like GPU acceleration or higher-order differentiation.

...your expertise and enthusiasm are welcome!

### Getting Started

If you're interested in contributing:

1.  **Browse the Issues:** Check the GitHub issues tab for existing bug reports, feature requests, or tasks that you might be able to help with.
2.  **Open an Issue:** If you have a new idea, want to work on something from the "Future Enhancements" list that isn't yet an issue, or want to discuss a potential change, please feel free to open an issue. This is a great way to start a discussion before diving into significant coding work.
3.  **Fork and Pull Request:** Once you've decided on a contribution:
    *   Fork the repository.
    *   Create a new branch for your feature or fix.
    *   Make your changes, including adding tests if applicable.
    *   Ensure your code is well-commented—particularly for educational clarity—and includes appropriate tests.
    *   Submit a pull request for review.

We appreciate all contributions, from simple bug fixes to major feature additions. Let's learn and build together!

## License

This project is licensed under the MIT License.
