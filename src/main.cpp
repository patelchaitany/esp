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

float mmse(Value &y_true, Value &y_pred)
{
    float result = 0.0f;
    if(y_true.ptr->rows != y_pred.ptr->rows || y_true.ptr->cols != y_pred.ptr->cols){
        std::cerr << "Error: y_true and y_pred must have the same shape" << std::endl;
        return result;
    }
    for (int i = 0; i < y_true.ptr->rows; i++)
    {
        for (int j = 0; j < y_true.ptr->cols; j++)
        {
            result += (y_true.ptr->data[i][j] - y_pred.ptr->data[i][j]) * (y_true.ptr->data[i][j] - y_pred.ptr->data[i][j]);
        }
    }
    result /= (float)(y_true.ptr->rows * y_true.ptr->cols);
    if(y_pred.ptr->grad){
        for (int i = 0; i < y_pred.ptr->rows; i++)
        {
            for (int j = 0; j < y_pred.ptr->cols; j++)
            {
                y_pred.ptr->grad[i][j] = (2.0f *( (y_pred.ptr->data[i][j] - y_true.ptr->data[i][j])/(float)(y_pred.ptr->rows * y_pred.ptr->cols)));
            }
        }
    }
    return result;
}

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
    const int num_points = 100;          // Reduced for better visualization
    const float learning_rate = 0.01f; // Increased for faster convergence
    const int epochs = 500;

    std::cout << "Starting neural network training...\n"
              << std::endl;

    // Create training data (x: num_points x 1, y: num_points x 1)
    float **x_data = create_data_array(num_points, 2, [num_points](int i, int j)
                                       { 
                                        // if(j==1) return (float)((float)i / num_points * 2 * M_PI)*(float)((float)i / num_points * 2 * M_PI);
                                        if(j==1) return 1.0f;
                                        return (float)((float)i / num_points * 2.0f* M_PI); 
                                    });

    float **y_data = create_data_array(num_points, 1, [num_points](int i, int j)
                                       {
                                           return std::sin((float)i / num_points * 2 * M_PI); // Fixed constant to match num_points
                                       });

    Value x_train(num_points, 2, x_data, "x_train");
    Value y_train(num_points, 1, y_data, "y_train");

    // Initialize weights and bias (W: 1 x 1, b: 1 x 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    float **w_data = create_data_array(2,6, [&dis, &gen](int i, int j)
                                       { return dis(gen); });

    float **w_data1 = create_data_array(6,1, [&dis, &gen](int i, int j)
                                       { return dis(gen); }); 

    float **b_data = create_data_array(1, 1, [&dis, &gen](int i, int j)
                                       { return dis(gen); });

    Value W1(2, 6, w_data, "W1"); // Single weight
    Value W2(6, 1, w_data1, "W2"); // Single weight
    Value b(1, 1, b_data, "b"); // Single bias
    // x_train.printdata();
    // y_train.printdata();
    // W.printdata();
    // b.printdata();

    // Training loop
    for (int epoch = 0; epoch < 10000; epoch++)
    {
        float loss = 0.0f;
        Value out;

        out = x_train * W1; //+ b;
        out = out.leakyrelu();
        out = out * W2;
        // float array[2][6];
        // for(int i=0;i<2;i++){
        //     for(int j=0;j<6;j++){
        //         array[i][j] = W1.ptr->data[i][j];
        //     }
        // }
        // float array1[6][1];
        // for(int i=0;i<6;i++){
        //     for(int j=0;j<1;j++){
        //         array1[i][j] = W2.ptr->data[i][j];
        //     }
        // }
        // Value error = (y_train - out)^(y_train - out);
        // // out.setgrad(create_data_array(num_points, 2, [](int i, int j)
        //                             //   { return 0.00001; }));
        // error.setgrad(create_data_array(num_points, 2, [](int i, int j)
        //                               { return 0.0000001; }));
        // // out.backward();
        // error.backward();
        loss = mmse(y_train, out);
        out.backward();

        // float grad1[2][6];
        // for(int i=0;i<2;i++){
        //     for(int j=0;j<6;j++){
        //         grad1[i][j] = W1.orig->grad[i][j];
        //     }
        // }
        // float grad2[6][1];  
        // for(int i=0;i<6;i++){
        //     for(int j=0;j<1;j++){
        //         grad2[i][j] = W2.orig->grad[i][j];
        //     }
        // }

        // float grad3[num_points][2];
        // for(int i=0;i<num_points;i++){
        //     for(int j=0;j<2;j++){
        //         grad3[i][j] = out.ptr->grad[i][j];
        //     }
        // }

        W1.update(learning_rate);
        W2.update(learning_rate);
        b.update(learning_rate);
        W1.setgradzero();
        W2.setgradzero();
        b.setgradzero();
        out.setgradzero();

        
        // Finalize loss and gradients

        // Print progress every 50 epochs
        if (epoch % 50 == 0)
        {
            std::cout << "Epoch " << epoch << ": ";
            std::cout << "Loss = " << loss << ", ";
            std::cout << "W = " << W1.orig->data[0][0] << ", ";
            std::cout << "b = " << b.orig->data[0][0];

            // Print predictions for a few points
            // if (epoch % 100 == 0)
            // {
            //     std::cout << "\nPredictions vs Actual:";
            //     for (int i = 0; i < num_points; i += 4)
            //     { // Sample every 4th point
            //         float x = x_train.orig->data[i][0];
            //         float y_actual = y_train.orig->data[i][0];
            //         float y_pred = W.orig->data[0][0] * x + b.orig->data[0][0];
            //         std::cout << "\nx = " << x << ": pred = " << y_pred << ", actual = " << y_actual;
            //     }
            //     std::cout << "\n";
            // }
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

    // Test the model
    std::cout << "\nTesting the model:" << std::endl;
    for (float x = 0; x <= 2 * M_PI; x += M_PI / 4)
    {
        float **input_data = create_data_array(1, 2, [x](int i, int j)
                                       { 
                                        // if(j==1) return (float)((float)i / num_points * 2 * M_PI)*(float)((float)i / num_points * 2 * M_PI);
                                        if(j==1) return 1.0f;
                                        return (float)(x); 
                                    });
        Value input(1, 2, input_data, "test_input");
        // input.printdata();
        Value pred = (input*W1); // + b;
        pred = pred.leakyrelu();
        pred = pred * W2;
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
