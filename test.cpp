#include <iostream>
#include <boost/intrusive_ptr.hpp>

// First class: Tensor
class Tensor {
private:
    int id;
    int* data;
    
public:
    int ref_count;  // Each class needs its own reference count
    
    Tensor(int id) : id(id), ref_count(0) {
        std::cout << "Tensor " << id << " Constructor\n";
        data = new int[10];
        for (int i = 0; i < 10; ++i) data[i] = i;
    }

    ~Tensor() {
        std::cout << "Tensor " << id << " Destructor\n";
        delete[] data;
    }

    void printData() {
        for (int i = 0; i < 10; ++i)
            std::cout << data[i] << " ";
        std::cout << "\n";
    }
};

// Second class: Matrix
class Matrix {
private:
    int id;
    double* data;
    
public:
    int ref_count;  // Each class needs its own reference count
    
    Matrix(int id) : id(id), ref_count(0) {
        std::cout << "Matrix " << id << " Constructor\n";
        data = new double[4];
        for (int i = 0; i < 4; ++i) data[i] = i * 1.5;
    }

    ~Matrix() {
        std::cout << "Matrix " << id << " Destructor\n";
        delete[] data;
    }

    void printData() {
        std::cout << "Matrix " << id << ": ";
        for (int i = 0; i < 4; ++i)
            std::cout << data[i] << " ";
        std::cout << "\n";
    }
};

// Required for intrusive_ptr with Tensor
void intrusive_ptr_add_ref(Tensor* t) {
    ++t->ref_count;
}

void intrusive_ptr_release(Tensor* t) {
    if (--t->ref_count == 0) {
        delete t;
    }
}

// Required for intrusive_ptr with Matrix
void intrusive_ptr_add_ref(Matrix* m) {
    ++m->ref_count;
}

void intrusive_ptr_release(Matrix* m) {
    if (--m->ref_count == 0) {
        delete m;
    }
}

int main() {
    // Using intrusive_ptr with Tensor objects
    Tensor* t = new Tensor(1);
    boost::intrusive_ptr<Tensor> tPtr1(t);
    std::cout << "Tensor Reference count: " << t->ref_count << std::endl;
    boost::intrusive_ptr<Tensor> tPtr2(t);
    std::cout << "Tensor Reference count: " << t->ref_count << std::endl;
    
    // Using intrusive_ptr with Matrix objects
    Matrix* m = new Matrix(1);
    boost::intrusive_ptr<Matrix> mPtr1(m);
    std::cout << "Matrix Reference count: " << m->ref_count << std::endl;
    boost::intrusive_ptr<Matrix> mPtr2(m);
    std::cout << "Matrix Reference count: " << m->ref_count << std::endl;
    
    std::cout << "\nPrinting data:\n";
    tPtr1->printData();
    mPtr1->printData();
    
    // Creating new objects
    boost::intrusive_ptr<Tensor> tPtr3(new Tensor(2));
    boost::intrusive_ptr<Matrix> mPtr3(new Matrix(2));
    
    // Reassigning pointers
    std::cout << "\nBefore reassignment:\n";
    tPtr1->printData();
    mPtr1->printData();
    
    tPtr1 = tPtr3;  // tPtr1 now points to Tensor(2)
    mPtr1 = mPtr3;  // mPtr1 now points to Matrix(2)
    
    std::cout << "\nAfter reassignment, second pointers still valid:\n";
    tPtr2->printData();  // Still points to Tensor(1)
    mPtr2->printData();  // Still points to Matrix(1)
    
    std::cout << "\nProgram ending\n";
    return 0;
}