#include <iostream>
#include <memory>
#include <matrix_mul.h>
#include <sys/resource.h>
#include <cstring>
#include <cstdlib>
#include <set>
#include <string>
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
    boost::intrusive_ptr<Tensor> ptr; 
    boost::intrusive_ptr<Tensor>orig;
public:
    Value()
    {
        ptr = nullptr;
        orig = nullptr;
        
    };
    explicit Value(Tensor* t)
    {
        ptr = t;
    }
    Value(int row, int cols, float **data, std::string name)
    {
        ptr = boost::intrusive_ptr<Tensor>(new Tensor(row, cols, data, name));
        orig = ptr;
    }
    Value& operator=(const Value &other)
    {
        if (this != &other) {
            ptr = other.ptr;
        }
        return *this;
    }
    Value operator+(const Value &other) const
    {
        return Value(new Tensor(*ptr + *other.ptr));
    }
    Value operator*(const Value &other) const
    {
        return Value(new Tensor(*ptr * *other.ptr));
    }
    Value operator^(const Value &other) const
    {
        return Value(new Tensor(*ptr ^ *other.ptr));
    }
    Value operator/(const Value &other) const
    {
        return Value(new Tensor(*ptr / *other.ptr));
    }

    void setgrad(float **grad)
    {
        ptr->setGrad(grad);
    }

    void backward()
    {
        ptr->backward();
    }
    void printgrad(){
        for(int i = 0 ;i<orig->rows;i++){
            for(int j = 0;j<orig->cols;j++){
                std::cout << orig->grad[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    void printdata(){
        for(int i = 0 ;i<orig->rows;i++){
            for(int j = 0;j<orig->cols;j++){
                std::cout << orig->data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
};

int main()
{

    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;
    float data[1][2] = {{1, 2}};
    float data2[2][1] = {{1}, {2}};
    float **data_ptr = (float **)malloc(1 * sizeof(float *));
    float **data_ptr2 = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 1; i++)
    {
        data_ptr[i] = (float *)malloc(2 * sizeof(float));
        memcpy(data_ptr[i], data[i], 2 * sizeof(float));
    }
    for (int i = 0; i < 2; i++)
    {
        data_ptr2[i] = (float *)malloc(1 * sizeof(float));
        memcpy(data_ptr2[i], data2[i], 1 * sizeof(float));
    }
    float **grad = (float **)malloc(1 * sizeof(float *));
    for (int i = 0; i < 1; i++)
    {
        grad[i] = (float *)malloc(1 * sizeof(float));
        for (int j = 0; j < 1; j++)
        {
            grad[i][j] = 1;
        }
    }
    Value a(1, 2, data_ptr, "t1");
    Value b(2, 1, data_ptr2, "t2");
    Value c;
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;
    // a = a;
    a = a*b;

    // c = c*a;
    a.setgrad(grad);
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    // c.backward();
    a.backward();
    a.printgrad();

    return 0;
}
