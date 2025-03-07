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

long getPeakMemoryUsage()
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        return usage.ru_maxrss;
    }
    else
    {
        std::cerr << "Error in getrusage" << std::endl;
        return -1;
    }
}

class Value
{
public:
    mutable boost::intrusive_ptr<Tensor> ptr;
    boost::intrusive_ptr<Tensor> orig;

public:
    Value()
    {
        ptr = nullptr;
        orig = nullptr;
    };

    explicit Value(Tensor *t)
    {
        ptr = t;
    }

    Value(int row, int cols, float **data, std::string name)
    {
        ptr = boost::intrusive_ptr<Tensor>(new Tensor(row, cols, data, name));
        orig = ptr;
    }

    Value &operator=(const Value &other)
    {
        if (this != &other)
        {
            ptr = other.ptr;
        }
        return *this;
    }

    Value operator+(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr + *other.ptr));
    }

    Value operator*(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr * *other.ptr));
    }

    Value operator^(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr ^ *other.ptr));
    }

    Value operator/(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr / *other.ptr));
    }
    Value operator-(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr - *other.ptr));
    }

    void setgrad(float **grad)
    {
        ptr->setGrad(grad);
    }
    void setgradzero()
    {
        if(this->ptr){ptr->setgradzero();}
        if(this->orig){orig->setgradzero();}
    }
    void backward()
    {
        ptr->backward();
    }

    void printgrad()
    {
        if (orig == nullptr)
        {
            for (int i = 0; i < ptr->rows; i++)
            {
                for (int j = 0; j < ptr->cols; j++)
                {
                    std::cout << ptr->grad[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            for (int i = 0; i < orig->rows; i++)
            {
                for (int j = 0; j < orig->cols; j++)
                {
                    std::cout << orig->grad[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    void printdata()
    {
        if (orig != nullptr)
        {
            std::cout<<"Original Data"<<orig->name<<std::endl;
            for (int i = 0; i < orig->rows; i++)
            {
                for (int j = 0; j < orig->cols; j++)
                {
                    std::cout << orig->data[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            std::cout<<"Data"<<ptr->name<<std::endl;
            for (int i = 0; i < ptr->rows; i++)
            {
                for (int j = 0; j < ptr->cols; j++)
                {
                    std::cout << ptr->data[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    void update(float learning_rate)
    {
        this->ptr = this->orig;
        orig->update(learning_rate);
    }
};

// Helper function to create data array
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
    const int num_points = 2;          // Reduced for better visualization
    const float learning_rate = 0.05f; // Increased for faster convergence
    const int epochs = 500;

    std::cout << "Starting neural network training...\n"
              << std::endl;

    // Create training data (x: num_points x 1, y: num_points x 1)
    float **x_data = create_data_array(num_points, 1, [num_points](int i, int j)
                                       { return (float)i / num_points * 2 * M_PI; });

    float **y_data = create_data_array(num_points, 1, [](int i, int j)
                                       {
                                           return std::sin((float)i / 20 * 2 * M_PI); // Fixed constant to match num_points
                                       });

    Value x_train(num_points, 1, x_data, "x_train");
    Value y_train(num_points, 1, y_data, "y_train");

    // Initialize weights and bias (W: 1 x 1, b: 1 x 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    float **w_data = create_data_array(1, 1, [&dis, &gen](int i, int j)
                                       { return dis(gen); });

    float **b_data = create_data_array(1, 1, [&dis, &gen](int i, int j)
                                       { return dis(gen); });

    Value W(1, 1, w_data, "W"); // Single weight
    Value b(1, 1, b_data, "b"); // Single bias
    x_train.printdata();
    y_train.printdata();
    W.printdata();
    b.printdata();

    // Training loop
    for (int epoch = 0; epoch < 10; epoch++)
    {
        // Forward pass - compute predictions
        // // Forward pass
        // float dW = 0.0f; // Accumulate gradients for W
        // float db = 0.0f; // Accumulate gradients for b
        float loss = 0.0f;
        Value out;
        // Compute predictions and accumulate gradients
        // for(int i = 0; i < num_points; i++) {
        //     float x = x_train.orig->data[i][0];
        //     float y = y_train.orig->data[i][0];

        //     // Forward pass for this point
        //     float pred = W.orig->data[0][0] * x + b.orig->data[0][0];

        //     // Compute error and contribution to loss
        //     float error = pred - y;
        //     loss += error * error;

        //     // Accumulate gradients
        //     dW += error * x;  // d(MSE)/dW = 2 * error * x
        //     db += error;      // d(MSE)/db = 2 * error
        // }

        out = x_train * W; //+ b;
        Value error = (y_train - out)^(y_train - out);
        error.setgrad(create_data_array(num_points, 1, [](int i, int j)
                                      { return 1; }));
        error.backward();
        W.update(learning_rate);
        b.update(learning_rate);
        error.setgradzero();
        W.setgradzero();
        b.setgradzero();
        out.setgradzero();

        
        // Finalize loss and gradients

        // Print progress every 50 epochs
        if (epoch % 50 == 0)
        {
            std::cout << "Epoch " << epoch << ": ";
            std::cout << "Loss = " << loss << ", ";
            std::cout << "W = " << W.orig->data[0][0] << ", ";
            std::cout << "b = " << b.orig->data[0][0];

            // Print predictions for a few points
            if (epoch % 100 == 0)
            {
                std::cout << "\nPredictions vs Actual:";
                for (int i = 0; i < num_points; i += 4)
                { // Sample every 4th point
                    float x = x_train.orig->data[i][0];
                    float y_actual = y_train.orig->data[i][0];
                    float y_pred = W.orig->data[0][0] * x + b.orig->data[0][0];
                    std::cout << "\nx = " << x << ": pred = " << y_pred << ", actual = " << y_actual;
                }
                std::cout << "\n";
            }
            std::cout << std::endl;
        }
    }

    // Print final parameters
    std::cout << "\nFinal parameters:" << std::endl;
    std::cout << "W: ";
    W.printdata();
    std::cout << "b: ";
    b.printdata();

    // Test the model
    std::cout << "\nTesting the model:" << std::endl;
    for (float x = 0; x <= 2 * M_PI; x += M_PI / 4)
    {
        float **input_data = create_data_array(1, 1, [x](int i, int j)
                                               { return x; });
        Value input(1, 1, input_data, "test_input");
        Value pred = (W * input); // + b;
        std::cout << "x = " << x << ", sin(x) = " << std::sin(x) << ", pred = ";
        pred.printdata();

        // Free test data
        free(input_data[0]);
        free(input_data);
    }

    // Free training data
    for (int i = 0; i < num_points; i++)
    {
        free(x_data[i]);
        free(y_data[i]);
    }
    free(x_data);
    free(y_data);

    // Free parameter data
    free(w_data[0]);
    free(w_data);
    free(b_data[0]);
    free(b_data);

    return 0;
}
