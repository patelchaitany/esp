// #include <iostream>
// #include <string.h>
// #include <set>
// #include <memory>
// #include "matrix_mul.h"

// using namespace std;

// int main() {
//     float data[2][2] = {{1, 2}, {3, 4}};
//     float data2[2][2] = {{5, 6}, {7, 8}};

//     float **data_ptr = (float **)malloc(2 * sizeof(float *));
//     for (int i = 0; i < 2; i++) {
//         data_ptr[i] = (float *)malloc(2 * sizeof(float));
//         memcpy(data_ptr[i], data[i], 2 * sizeof(float));
//     }

//     Tensor t1(2, 2, data_ptr, std::set<Tensor>(), "t1");
//     Tensor t2(2, 2, data_ptr, std::set<Tensor>(), "t2");

//     Tensor t3;
//     t3 = t1 + t2;
//     t3 = t2 + t3;
//     std::set<std::shared_ptr<Tensor>> child1;
//     for (const auto& c : t3.child) {
//         child1.insert(c);
//     }

//     std::set<std::shared_ptr<Tensor>> visited;
//     cout << "in main ----/" << endl;

//     for (const auto& c : child1) {
//         if (visited.find(c) != visited.end()) {
//             cout << "already visited " << c->name << endl;
//             continue;
//         }
//         for (const auto& c1 : c->child) {
//             cout << c1->name << endl;
//         }
//         cout << "Hello " << endl;
//         cout << c->name << endl;
//         visited.insert(c);
//     }
//     cout << "in main ----/" << endl;

//     for (int i = 0; i < t3.rows; i++) {
//         for (int j = 0; j < t3.cols; j++) {
//             cout << t3.data[i][j] << " ";
//         }
//         cout << endl;
//     }

//     float **grad = (float **)malloc(2 * sizeof(float *));
//     for (int i = 0; i < t3.rows; i++) {
//         grad[i] = (float *)malloc(2 * sizeof(float));
//         for (int j = 0; j < t3.cols; j++) {
//             grad[i][j] = 1;
//         }
//     }

//     t3.setGrad(grad);
//     t3.backward();

//     for (int i = 0; i < t1.rows; i++) {
//         for (int j = 0; j < t1.cols; j++) {
//             cout << t1.grad[i][j] << " ";
//         }
//         cout << endl;
//     }

//     // Cleanup allocated memory
//     for (int i = 0; i < 2; i++) {
//         free(data_ptr[i]);
//         free(grad[i]);
//     }
//     free(data_ptr);
//     free(grad);

//     return 0;
// }

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
    Value(int row, int cols, float **data, std::string name)
    {
        ptr = boost::intrusive_ptr<Tensor>(new Tensor(row, cols, data, name));
    }
    // Overload + operator to create a new object
    Value operator=(const Value &other)
    {
        ptr = other.ptr;
        return *this;
    }
    Value operator+(const Value &other) const
    {
        Value obj;
        Tensor result = *ptr + *other.ptr;
        obj.ptr = boost::intrusive_ptr<Tensor>(new Tensor(result));
        return obj;
    }
    Value operator*(const Value &other) const
    {
        Value obj;
        Tensor result = *ptr * *other.ptr;
        obj.ptr = boost::intrusive_ptr<Tensor>(new Tensor(result));
        return obj;
    }
    Value operator^(const Value &other) const
    {
        Value obj;
        Tensor result = *ptr ^ *other.ptr;
        obj.ptr = boost::intrusive_ptr<Tensor>(new Tensor(result));
        return obj;
    }

    Value operator/(const Value &other) const
    {
        Value obj;
        Tensor result = *ptr / *other.ptr;
        obj.ptr = boost::intrusive_ptr<Tensor>(new Tensor(result));
        return obj;
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
    Value c(2, 2, data_ptr, "t3");
    a.setgrad(grad);
//0x55555559ae80
    // b.setChild(a);
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    a.printAddress();

    a = a + c; // Creates a new object, like Python behavior
    c = a + c;
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    a.printAddress();
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    // std::cout << "After:  a.value = " << a.getValue() << " ";
    a.printAddress();
    a.backward();
    // std::cout << "b.value = " << b.child.get()  << std::endl;

    return 0;
}
