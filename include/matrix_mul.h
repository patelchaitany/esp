#pragma once

#include <iostream>
#include <string.h>
#include <stdexcept>
#include <set>
#include <vector>
#include <uuid/uuid.h>
#include <functional>

typedef float float32;

//here data refers to genral term it can be any thing from the input to the weights or anything yo u can imagine.


class Tensor {
    typedef float float32;
public:
    uuid_t id;
    int rows, cols, batch;
    std::set< Tensor > child;
    float32 **data;
    float32 **grad;
    void (Tensor::*_backward)() = nullptr; 
    std::string name;
    std::string uuidstr;
    Tensor(){
        uuid_generate(id);
        this->rows = 0;
        this->cols = 0;
        data = NULL;
        _backward  = nullptr;
        grad = NULL;
    }
    Tensor(int rows, int cols,std::set< Tensor > child = std::set< Tensor >(),std::string name = "") {
        uuid_generate(id);
        char uuid_str[37];
         uuid_unparse(id, uuid_str);
         uuidstr = std::string(uuid_str);
        this->rows = rows;
        this->cols = cols;
        this->child = child;
        this->name = name;
        _backward = nullptr;

        grad= (float32 **)malloc(rows * sizeof(float32*));

            data = (float32 **)malloc(rows * sizeof(float32*));
            for (int j = 0; j < rows; j++) {
                data[j] = (float32 *)malloc(cols * sizeof(float32));
                grad[j] = (float32 *)malloc(cols * sizeof(float32)); 
                memset(data[j], 0, cols * sizeof(float32));
                memset(grad[j], 0, cols * sizeof(float32));
            }
        
    }
    Tensor(int rows,int cols,float32 **data,std::set< Tensor > child = std::set< Tensor>(),std::string name = "") {
        uuid_generate(id);
        char uuid_str[37];
         uuid_unparse(id, uuid_str);
         uuidstr = std::string(uuid_str);
        this->rows = rows;
        this->cols = cols;
        this->data = data;
        this->child = child;
        this->name = name;
        _backward  = nullptr;

        grad = (float32 **)malloc(rows * sizeof(float32*));
        for (int j = 0; j < rows; j++) {
            grad[j] = (float32 *)malloc(cols * sizeof(float32));
            memset(grad[j], 0, cols * sizeof(float32));
        }

    }

    Tensor(const Tensor &t) {
        uuid_copy(this->id, t.id);
        char uuid_str[37];
         uuid_unparse(t.id, uuid_str);
         uuidstr = std::string(uuid_str);
        this->rows = t.rows;
        this->cols = t.cols;
        this->name = t.name;
        _backward = t._backward;
        
        // Deep copy the child set pointers
        this->child = t.child;
        this->data = t.data;
        this->grad = t.grad;
        // Allocate and copy data arrays
        // data = (float32 **)malloc(rows * sizeof(float32*));
        // grad = (float32 **)malloc(rows * sizeof(float32*));
        // for (int j = 0; j < rows; j++) {
        //     data[j] = (float32 *)malloc(cols * sizeof(float32));
        //     grad[j] = (float32 *)malloc(cols * sizeof(float32));
        //     memcpy(data[j], t.data[j], cols * sizeof(float32));
        //     memcpy(grad[j], t.grad[j], cols * sizeof(float32));
        // }
    }

    // ~Tensor() {
    //         for (int j = 0; j < rows; j++) {
    //             free(data[j]);
    //             free(grad[j]);
    //         }
    //         // free(data);
    //         // free(grad);
    
    // }
    

    Tensor& operator=(const Tensor &t);
    Tensor operator+(const Tensor &t) const;
    Tensor operator/(const Tensor &t) const;
    Tensor operator*(const Tensor &t) const;
    Tensor operator^(const Tensor &t) const; // Custom operator for dot multiplication
    void backadd();
    bool operator<(const Tensor &t) const {
        return uuid_compare(this->id, t.id) < 0;
    }
    bool operator<(const Tensor* t) const{
        return uuid_compare(this->id,t->id) <0;
    }
    bool operator==(const Tensor &t) const {
        return uuid_compare(this->id, t.id) == 0;
    }
    bool operator==(const Tensor *t) const {
        return uuid_compare(this->id, t->id) == 0;
    }
    void bacward();

};