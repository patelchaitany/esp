#include <iostream>
#include <memory>  // For shared_ptr
#include <matrix_mul.h>

// Wrapper class to manage shared_ptr internally
class MyObject {
public:
    std::shared_ptr<Tensor> ptr;  // Shared pointer to manage MyClass object
public:
    // Constructor
    MyObject(){
        ptr = nullptr;
    };
    MyObject(int row,int cols,float **data){
        ptr = std::make_shared<Tensor>(row,cols,data,std::set<Tensor>(), "t1");
    }

    // Overload + operator to create a new object
    MyObject operator+(const MyObject &other) const {
        Tensor result = *ptr + *other.ptr;  // Calls MyClass operator+
        MyObject obj;
        obj.ptr = std::make_shared<Tensor>(result);  // Create new shared_ptr
        return obj;
    }
    
    // Print memory address (for demonstration)
    void printAddress() const {
        std::cout << "Address: " << ptr.get() << std::endl;
    }
};

int main() {
    float data[2][2] = {{1, 2}, {3, 4}};
    float data2[2][2] = {{5, 6}, {7, 8}};

    float **data_ptr = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++) {
        data_ptr[i] = (float *)malloc(2 * sizeof(float));
        memcpy(data_ptr[i], data[i], 2 * sizeof(float));
    }
    MyObject a(2, 2, data_ptr);
    MyObject b(2, 2, data_ptr);
    MyObject c(2, 2, data_ptr);
    // b.setChild(a);
    a.printAddress();
    

    a = a + c;  // Creates a new object, like Python behavior

    // std::cout << "After:  a.value = " << a.getValue() << " ";
    a.printAddress();
    // std::cout << "b.value = " << b.child.get()  << std::endl;

    return 0;
}
