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
        return usage.ru_maxrss; // Peak memory usage in kilobytes
    }
    else
    {
        std::cerr << "Error in getrusage" << std::endl;
        return -1;
    }
}
// Wrapper class to manage intrusive_ptr internally
class Value
{
public:
    boost::intrusive_ptr<Tensor> ptr; // Intrusive pointer to manage Tensor object
    // std::shared_ptr<Tensor> ptr; // Shared pointer to manage MyClass object
public:
    // Constructor
    Value()
    {
        ptr = nullptr;
    };
    // Constructor from Tensor pointer
    explicit Value(Tensor* t)
    {
        ptr = t;
    }
    Value(int row, int cols, float **data, std::string name)
    {
        ptr = boost::intrusive_ptr<Tensor>(new Tensor(row, cols, data, name));
    }
    // Overload + operator to create a new object
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
    // Print memory address (for demonstration)
    void printAddress() const
    {
        std::cout << "Address: " << ptr.get() << std::endl;
    }
    void backward()
    {
        ptr->backward();
    }
    void printgrad(){
        for(int i = 0 ;i<ptr->rows;i++){
            for(int j = 0;j<ptr->cols;j++){
                std::cout << ptr->grad[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main()
{

    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;
    float data[2][2] = {{1, 2}, {3, 4}};
    float data2[2][2] = {{5, 6}, {7, 8}};
    float **data_ptr = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        data_ptr[i] = (float *)malloc(2 * sizeof(float));
        memcpy(data_ptr[i], data[i], 2 * sizeof(float));
    }
    float **grad = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        grad[i] = (float *)malloc(2 * sizeof(float));
        for (int j = 0; j < 2; j++)
        {
            grad[i][j] = 1;
        }
    }
    Value a(2, 2, data_ptr, "t1");
    Value b(2, 2, data_ptr, "t2");
    Value c;
    // a.setgrad(grad);
//0x55555559ae80
    // b.setChild(a);
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    a.printAddress();
    c = a;
    c = c*b;
    // for(int i = 0;i<100;i++){
    //     if(i%2 == 0){
            
    //     }
    //     c = c + a;
    // }
    c = c*a;
    c.setgrad(grad);
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    a.printAddress();
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    // std::cout << "After:  a.value = " << a.getValue() << " ";
    a.printAddress();
    c.backward();
    a.printgrad();
    // b.printgrad();
    // std::cout << "b.value = " << b.child.get()  << std::endl;

    return 0;
}
