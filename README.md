# TensorGrad: C++ Automatic Differentiation Library

A high-performance C++ library for tensor operations with automatic gradient computation support, designed for machine learning applications.

## Features

- **Tensor Operations**
  - Matrix multiplication
  - Element-wise addition and subtraction
  - Dot product
  - Division
  - LeakyReLU activation function
  
- **Automatic Gradient Computation**
  - Backward pass implementation for all operations
  - Gradient clipping for numerical stability
  - Topological sort for correct gradient propagation
  
- **Memory Management**
  - Smart pointer implementation using `boost::intrusive_ptr`
  - RAII-compliant resource management
  - Efficient memory handling for large matrices
  
- **Performance**
  - High-resolution timing utilities
  - Optimized matrix operations
  - Gradient norm clipping for training stability

## Core Components

### Tensor Class (`matrix_mul.h`)
The main class handling tensor operations and gradient computation:
```cpp
class Tensor {
public:
    // Constructors
    Tensor();  // Default 1x1 tensor
    Tensor(int rows, int cols, float32** input_data = nullptr, std::string name = "");
    
    // Core Operations
    Tensor operator+(const Tensor& t) const;  // Addition
    Tensor operator-(const Tensor& t) const;  // Subtraction
    Tensor operator*(const Tensor& t) const;  // Matrix multiplication
    Tensor operator^(const Tensor& t) const;  // Dot product
    Tensor operator/(const Tensor& t) const;  // Division
    Tensor lekyrelu(float leaky = 0.01);     // LeakyReLU activation
    
    // Gradient Operations
    void backward();     // Compute gradients
    void setgradzero(); // Reset gradients to zero
    void update(float32 learning_rate); // Update weights using gradients
};
```

### Timer Class (`timer.h`)
Utility class for performance measurement:
```cpp
class Timer {
public:
    Timer(const std::string& label = "");
    double stop();  // Returns time in milliseconds
    void reset();
    static double measureBlock(const std::string& label, const std::function<void()>& block);
};
```

## Usage Examples

### Basic Matrix Operations
```cpp
// Create tensors
Tensor A(2, 3);  // 2x3 matrix
Tensor B(3, 2);  // 3x2 matrix

// Matrix multiplication
Tensor C = A * B;  // 2x2 result

// Element-wise operations
Tensor D = A + A;  // Element-wise addition
Tensor E = A - A;  // Element-wise subtraction

// Apply activation function
Tensor F = A.lekyrelu(0.01);  // LeakyReLU with slope 0.01
```

### Gradient Computation
```cpp
// Forward pass
Tensor loss = compute_loss();

// Backward pass
loss.backward();  // Computes gradients for all tensors in computational graph

// Update weights
float learning_rate = 0.01;
weights.update(learning_rate);
weights.setgradzero();  // Reset gradients for next iteration
```

### Performance Measurement
```cpp
// Measure operation time
Timer timer("Matrix Multiplication");
Tensor result = A * B;
double time_ms = timer.stop();

// Measure code block
double block_time = Timer::measureBlock("Complex Operation", [&]() {
    // Your operations here
    Tensor result = (A * B + C).lekyrelu();
});
```

## Building the Project

### Prerequisites
- C++17 or later
- Boost library
- UUID library
- CMake 3.10 or later

### Installation
1. Clone the repository
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create build directory and build
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Optimization Features

- Gradient clipping to prevent exploding gradients
- Minimum gradient norm threshold to avoid vanishing gradients
- Efficient memory management using smart pointers
- Automatic cleanup of computation graph after backward pass

## Constants

Important constants for gradient handling:
```cpp
const float CLIP_NORM = 1.0f;      // Maximum gradient norm
const float MIN_GRAD_NORM = 1e-3f; // Minimum gradient norm
const float EPSILON = 1e-6f;       // Small value for numerical stability
```

## Implementation Notes

- Uses `boost::intrusive_ptr` for reference counting
- Implements move semantics for efficient tensor operations
- Provides UUID-based tensor identification for graph operations
- Supports automatic memory management for both data and gradient matrices

## Implementing Custom Operations

To add your own tensor operations, follow these steps:

1. **Add Forward Operation**
```cpp
// In matrix_mul.h
class Tensor {
public:
    // Declare your operation
    Tensor custom_operation(const Tensor& other) const;
    // Declare backward function
    void back_custom_operation();
};
```

2. **Implement Forward Pass**
```cpp
// In matrix_mul.cpp
Tensor Tensor::custom_operation(const Tensor& other) const {
    // Create result tensor
    Tensor result(this->rows, this->cols);
    
    // Set parent tensors for gradient computation
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.right = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(&other));
    
    // Set operation name for debugging
    result.name = this->name + "_custom_" + other.name;
    
    // Implement your forward computation
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = /* Your computation */;
        }
    }
    
    // Set backward function pointer
    result._backward = &Tensor::back_custom_operation;
    return result;
}
```

3. **Implement Backward Pass**
```cpp
// In matrix_mul.cpp
void Tensor::back_custom_operation() {
    // Compute gradients for left tensor (this)
    if (this->left) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Update gradients based on chain rule
                left->grad[i][j] += /* Your gradient computation */;
            }
        }
        clip_gradient(left->grad, this->left->rows, this->left->cols);
    }
    
    // Compute gradients for right tensor
    if (this->right) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Update gradients based on chain rule
                right->grad[i][j] += /* Your gradient computation */;
            }
        }
        clip_gradient(right->grad, this->right->rows, this->right->cols);
    }
}
```

### Example: Implementing Element-wise Multiplication

Here's a complete example implementing element-wise multiplication:

```cpp
// In matrix_mul.h
class Tensor {
public:
    Tensor hadamard(const Tensor& other) const;  // Element-wise multiplication
    void back_hadamard();
};

// In matrix_mul.cpp
Tensor Tensor::hadamard(const Tensor& other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    
    Tensor result(rows, cols);
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.right = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(&other));
    result.name = this->name + "⊙" + other.name;
    
    // Forward pass: element-wise multiplication
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = this->data[i][j] * other.data[i][j];
        }
    }
    
    result._backward = &Tensor::back_hadamard;
    return result;
}

void Tensor::back_hadamard() {
    // Backward pass using product rule
    if (this->left) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                left->grad[i][j] += this->grad[i][j] * right->data[i][j];
            }
        }
        clip_gradient(left->grad, this->left->rows, this->left->cols);
    }
    
    if (this->right) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                right->grad[i][j] += this->grad[i][j] * left->data[i][j];
            }
        }
        clip_gradient(right->grad, this->right->rows, this->right->cols);
    }
}
```

Usage of custom operation:
```cpp
Tensor A(2, 2);  // Initialize with data
Tensor B(2, 2);  // Initialize with data
Tensor C = A.hadamard(B);  // Element-wise multiplication
C.backward();  // Compute gradients
```

Remember to:
- Always implement both forward and backward passes
- Set parent tensors using intrusive_ptr
- Apply gradient clipping for numerical stability
- Handle edge cases and input validation
- Update gradients using the chain rule