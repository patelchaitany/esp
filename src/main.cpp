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
#include <memory>  // For shared_ptr
#include <matrix_mul.h>
#include <sys/resource.h>  // For getrusage
#include <cstring>  // For memcpy
#include <cstdlib>  // For malloc and free
#include <set>
#include <string>

long getPeakMemoryUsage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;  // Peak memory usage in kilobytes
    } else {
        std::cerr << "Error in getrusage" << std::endl;
        return -1;
    }
}

// Wrapper class to manage shared_ptr internally
class Value{
public:
    std::shared_ptr<Tensor> ptr;  // Shared pointer to manage MyClass object
public:
    // Constructor
    Value(){
        ptr = nullptr;
    };
    Value(int row,int cols,float **data,std::string name){
        ptr = std::make_shared<Tensor>(row,cols,data,std::set<Tensor>(), name);
    }
    // Overload + operator to create a new object
    Value operator+(const Value &other) const {
        Tensor result = *ptr + *other.ptr;  // Calls MyClass operator+
        Value obj;
        obj.ptr = std::make_shared<Tensor>(result);  // Create new shared_ptr
        return obj;
    }
    Value operator*(const Value &other) const {
        Tensor result = *ptr * *other.ptr;  // Calls MyClass operator+
        Value obj;
        obj.ptr = std::make_shared<Tensor>(result);  // Create new shared_ptr
        return obj;
    }
    Value operator^(const Value &other) const {
        Tensor result = *ptr ^ *other.ptr;  // Calls MyClass operator+
        Value obj;
        obj.ptr = std::make_shared<Tensor>(result);  // Create new shared_ptr
        return obj;
    }

    Value operator/(const Value &other) const {
        Tensor result = *ptr / *other.ptr;  // Calls MyClass operator+
        Value obj;
        obj.ptr = std::make_shared<Tensor>(result);  // Create new shared_ptr
        return obj;
    }
    // Print memory address (for demonstration)
    void printAddress() const {
        std::cout << "Address: " << ptr.get() << std::endl;
    }
};

int main() {

    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;
    float data[2][2] = {{1, 2}, {3, 4}};
    float data2[2][2] = {{5, 6}, {7, 8}};
    float **data_ptr = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++) {
        data_ptr[i] = (float *)malloc(2 * sizeof(float));
        memcpy(data_ptr[i], data[i], 2 * sizeof(float));
    }

    Value a(2, 2, data_ptr,"t1");
    Value b(2, 2, data_ptr,"t2");
    Value c(2, 2, data_ptr,"t3");
    // b.setChild(a);
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    a.printAddress();
    

    a = a + c;  // Creates a new object, like Python behavior
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    a.printAddress();
    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    std::cout << "Peak Memory Usage: " << getPeakMemoryUsage() << " KB" << std::endl;

    // std::cout << "After:  a.value = " << a.getValue() << " ";
    a.printAddress();
    // std::cout << "b.value = " << b.child.get()  << std::endl;

    return 0;
}
