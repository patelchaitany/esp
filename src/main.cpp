#include <iostream>
#include <memory>
#include <matrix_mul.h>
#include <sys/resource.h>
#include <cstring>
#include <cstdlib>
#include <set>
#include <string>
#include <cmath>
#include <random>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <value.h>
#include <timer.h>

/**
 * @brief Get the peak memory usage of the current process
 * @return Peak memory usage in KB, or -1 if error
 */
long getPeakMemoryUsage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    std::cerr << "Error in getrusage" << std::endl;
    return -1;
}

/**
 * @brief Calculate Mean Squared Error and its gradient
 * @param y_true True values
 * @param y_pred Predicted values
 * @return MSE loss value
 */
float mmse(Value &y_true, Value &y_pred) {
    float result = 0.0f;
    
    // Check dimensions match
    if (y_true.ptr->rows != y_pred.ptr->rows || y_true.ptr->cols != y_pred.ptr->cols) {
        std::cerr << "Error: y_true and y_pred must have the same shape" << std::endl;
        return result;
    }

    // Calculate MSE
    for (int i = 0; i < y_true.ptr->rows; i++) {
        for (int j = 0; j < y_true.ptr->cols; j++) {
            float diff = y_true.ptr->data[i][j] - y_pred.ptr->data[i][j];
            result += diff * diff;
        }
    }
    result /= static_cast<float>(y_true.ptr->rows * y_true.ptr->cols);

    // Calculate gradients if required
    if (y_pred.ptr->grad) {
        float scale = 2.0f / static_cast<float>(y_pred.ptr->rows * y_pred.ptr->cols);
        for (int i = 0; i < y_pred.ptr->rows; i++) {
            for (int j = 0; j < y_pred.ptr->cols; j++) {
                y_pred.ptr->grad[i][j] = scale * (y_pred.ptr->data[i][j] - y_true.ptr->data[i][j]);
            }
        }
    }
    return result;
}

/**
 * @brief Create a 2D array with initialization function
 * @param rows Number of rows
 * @param cols Number of columns
 * @param init_func Function to initialize each element
 * @return Pointer to the created array
 */
float **create_data_array(int rows, int cols, std::function<float(int, int)> init_func)
{
    float **data = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        data[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++)
        {
            data[i][j] = init_func(i, j);
        }
    }
    return data;
}

int main()
{
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    // Training parameters
    const int num_points = 100;          // Reduced for better visualization
    const float learning_rate = 0.01f; // Increased for faster convergence
    const int epochs = 500;

    std::cout << "Starting neural network training...\n"
              << std::endl;

   // Create training data
   // Input features: x values and bias term
   float **x_data = create_data_array(num_points, 2, [num_points](int i, int j) -> float {
       if (j == 1) {
           return 1.0f;  // Bias term
       }
       return static_cast<float>(i) / static_cast<float>(num_points) * 2.0f * M_PI;  // Input x value
   });

   // Target values: sin(x)
   float **y_data = create_data_array(num_points, 1, [num_points](int i, int j) -> float {
       return std::sin(static_cast<float>(i) / static_cast<float>(num_points) * 2.0f * M_PI);
   });

   // Create Value objects for training
   Value x_train(num_points, 2, x_data, "x_train");  // [x, bias]
   Value y_train(num_points, 1, y_data, "y_train");  // [sin(x)]

    // Initialize model parameters
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Neural network architecture:
    // Input layer (2) -> Hidden layer (6) -> Output layer (1)
    float **w1_data = create_data_array(2, 512, [&dis, &gen](int i, int j) -> float {
        return dis(gen);  // Random initialization for first layer weights
    });

    float **w2_data = create_data_array(512, 1, [&dis, &gen](int i, int j) -> float {
        return dis(gen);  // Random initialization for second layer weights
    });

    float **b_data = create_data_array(1, 1, [&dis, &gen](int i, int j) -> float {
        return dis(gen);  // Random initialization for bias
    });

    // Create Value objects for model parameters
    Value W1(2, 64, w1_data, "W1");  // First layer weights: [2 x 6]
    Value W2(64, 1, w2_data, "W2");  // Second layer weights: [6 x 1]
    Value b(1, 1, b_data, "b");     // Bias: [1 x 1]

    // Training loop - Neural Network with one hidden layer
    std::cout << "\nTraining neural network with architecture:\n";
    std::cout << "Input (2) -> Hidden (6) -> Output (1)\n\n";

    const int max_epochs = 10000;
    Timer total_timer("Total training time");
    double epoch_time = 0.0;
    
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        Timer epoch_timer;
        
        // Forward pass
        Value hidden = x_train * W1;        // [num_points x 6]
        Value hidden_act = hidden.leakyrelu(); // Apply activation
        Value out = hidden_act * W2;        // [num_points x 1]

        // Compute loss and gradients
        float loss = mmse(y_train, out);
        out.backward();  // Backpropagate gradients

        // Update parameters
        W1.update(learning_rate);
        W2.update(learning_rate);
        b.update(learning_rate);

        // Reset gradients
        W1.setgradzero();
        W2.setgradzero();
        b.setgradzero();
        out.setgradzero();

        // Print training progress
        if (epoch % 50 == 0) {
            epoch_time = epoch_timer.stop();
            std::cout << "Epoch " << epoch << "/" << max_epochs << ": ";
            std::cout << "Loss = " << loss << ", ";
            std::cout << "Time = " << epoch_time << " ms, ";
            std::cout << "W1[0,0] = " << W1.orig->data[0][0] << ", ";
            std::cout << "W2[0,0] = " << W2.orig->data[0][0];
            std::cout << std::endl;
        }
    }

    // Print final parameters
    std::cout << "\nFinal parameters:" << std::endl;
    std::cout << "W: ";
    W1.printdata();
    W2.printdata();
    std::cout << "b: ";
    b.printdata();

    // Model evaluation
    std::cout << "\nTesting the model on points from 0 to 2π:" << std::endl;
    std::cout << "----------------------------------------\n";
    
    Timer eval_timer("Model evaluation time");
    double total_inference_time = 0.0;
    int num_test_points = 0;
    
    for (float x = 0; x <= 2 * M_PI; x += M_PI / 4) {
        Timer inference_timer;
        num_test_points++;
        
        // Create test input [x, bias_term]
        float **input_data = create_data_array(1, 2, [x](int i, int j) -> float {
            return j == 1 ? 1.0f : x;
        });
        
        // Forward pass through network
        Value input(1, 2, input_data, "test_input");
        Value hidden = input * W1;
        Value hidden_act = hidden.leakyrelu();
        Value pred = hidden_act * W2;
        
        double inference_time = inference_timer.stop();
        total_inference_time += inference_time;
        
        // Print results
        std::cout << "x = " << x << ", sin(x) = " << std::sin(x) << ", prediction = ";
        pred.printdata();
        std::cout << "Inference time: " << inference_time << " ms\n";
        
        // Cleanup
        free(input_data[0]);
        free(input_data);
    }
    
    std::cout << "\nAverage inference time: " << (total_inference_time / num_test_points) << " ms" << std::endl;

    // Free training data
    for (int i = 0; i < num_points; i++)
    {
        free(x_data[i]);
        free(y_data[i]);
    }
    free(x_data);
    free(y_data);

    // Free parameter data
    free(w1_data[0]);
    free(w1_data);
    free(w2_data[0]);
    free(w2_data);
    free(b_data[0]);
    free(b_data);

    return 0;
}
