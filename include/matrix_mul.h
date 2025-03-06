#pragma once

#include <iostream>
#include <string.h>
#include <stdexcept>
#include <set>
#include <vector>
#include <uuid/uuid.h>
#include <functional>
#include <memory>

typedef float float32;

class Tensor {
    typedef float float32;
public:
    uuid_t id;
    int rows, cols, batch;
    std::set<std::shared_ptr<Tensor>> child;
    float32** data;  // Keep as raw pointer for direct access
    float32** grad;  // Keep as raw pointer for direct access
    void (Tensor::*_backward)() = nullptr; 
    std::string name;
    std::string uuidstr;
private:
    std::shared_ptr<float32*[]> data_holder;  // For memory management
    std::shared_ptr<float32*[]> grad_holder;  // For memory management

public:
    // Default constructor - initialize with 1x1 tensor
    Tensor() {
        uuid_generate(id);
        char uuid_str[37];
        uuid_unparse(id, uuid_str);
        uuidstr = std::string(uuid_str);
        
        this->rows = 1;
        this->cols = 1;
        this->name = "default";
        this->_backward = nullptr;

        // Allocate minimum memory for 1x1 tensor
        data_holder = std::shared_ptr<float32*[]>(new float32*[1],
            [](float32** p) {
                delete[] p[0];
                delete[] p;
            });
        
        grad_holder = std::shared_ptr<float32*[]>(new float32*[1],
            [](float32** p) {
                delete[] p[0];
                delete[] p;
            });

        data = data_holder.get();
        grad = grad_holder.get();

        data[0] = new float32[1]();  // Initialize to zero
        grad[0] = new float32[1]();  // Initialize to zero
    }

    Tensor(int rows, int cols, float32** input_data = nullptr, const std::set<Tensor>& old_child = {}, std::string name = "") {
        uuid_generate(id);
        char uuid_str[37];
        uuid_unparse(id, uuid_str);
        uuidstr = std::string(uuid_str);
        
        this->rows = rows;
        this->cols = cols;
        this->name = name;
        this->_backward = nullptr;

        // Convert old_child to shared_ptr set
        for (const auto& t : old_child) {
            child.insert(std::make_shared<Tensor>(t));
        }

        int r = rows;
        data_holder = std::shared_ptr<float32*[]>(new float32*[r], 
            [r](float32** p) {
                for (int i = 0; i < r; i++) {
                    delete[] p[i];
                }
                delete[] p;
            });
        
        grad_holder = std::shared_ptr<float32*[]>(new float32*[r],
            [r](float32** p) {
                for (int i = 0; i < r; i++) {
                    delete[] p[i];
                }
                delete[] p;
            });

        data = data_holder.get();
        grad = grad_holder.get();

        for (int j = 0; j < rows; j++) {
            data[j] = new float32[cols]();  // Initialize to zero
            grad[j] = new float32[cols]();  // Initialize to zero
            if (input_data) {
                memcpy(data[j], input_data[j], cols * sizeof(float32));
            }
        }
    }

    // Copy constructor
    Tensor(const Tensor& t) {
        uuid_copy(this->id, t.id);
        char uuid_str[37];
        uuid_unparse(t.id, uuid_str);
        uuidstr = std::string(uuid_str);
        
        this->rows = t.rows;
        this->cols = t.cols;
        this->name = t.name;
        this->_backward = t._backward;
        
        // Copy child tensors
        this->child = t.child;

        int r = rows;
        data_holder = std::shared_ptr<float32*[]>(new float32*[r],
            [r](float32** p) {
                for (int i = 0; i < r; i++) {
                    delete[] p[i];
                }
                delete[] p;
            });
        
        grad_holder = std::shared_ptr<float32*[]>(new float32*[r],
            [r](float32** p) {
                for (int i = 0; i < r; i++) {
                    delete[] p[i];
                }
                delete[] p;
            });

        data = data_holder.get();
        grad = grad_holder.get();

        for (int j = 0; j < rows; j++) {
            data[j] = new float32[cols];
            grad[j] = new float32[cols];
            if (t.data && t.data[j]) {
                memcpy(data[j], t.data[j], cols * sizeof(float32));
            }
            if (t.grad && t.grad[j]) {
                memcpy(grad[j], t.grad[j], cols * sizeof(float32));
            }
        }
    }

    // Move constructor
    Tensor(Tensor&& t) noexcept {
        uuid_copy(this->id, t.id);
        this->uuidstr = std::move(t.uuidstr);
        this->rows = t.rows;
        this->cols = t.cols;
        this->name = std::move(t.name);
        this->_backward = t._backward;
        this->child = std::move(t.child);
        this->data_holder = std::move(t.data_holder);
        this->grad_holder = std::move(t.grad_holder);
        this->data = this->data_holder.get();
        this->grad = this->grad_holder.get();
        
        t.data = nullptr;
        t.grad = nullptr;
        t.rows = 0;
        t.cols = 0;
    }

    void setGrad(float32** new_grad) {
        int r = rows;
        grad_holder = std::shared_ptr<float32*[]>(new float32*[r],
            [r](float32** p) {
                for (int i = 0; i < r; i++) {
                    delete[] p[i];
                }
                delete[] p;
            });
        grad = grad_holder.get();
        for (int j = 0; j < rows; j++) {
            grad[j] = new float32[cols];
            memcpy(grad[j], new_grad[j], cols * sizeof(float32));
        }
    }

    Tensor& operator=(const Tensor& t);
    Tensor operator+(const Tensor& t) const;
    Tensor operator/(const Tensor& t) const;
    Tensor operator*(const Tensor& t) const;
    Tensor operator^(const Tensor& t) const;

    void backadd();
    void backward();

    bool operator<(const Tensor& t) const {
        return uuid_compare(this->id, t.id) < 0;
    }

    bool operator<(const Tensor* t) const {
        return uuid_compare(this->id, t->id) < 0;
    }

    bool operator==(const Tensor& t) const {
        return uuid_compare(this->id, t.id) == 0;
    }

    bool operator==(const Tensor* t) const {
        return uuid_compare(this->id, t->id) == 0;
    }
};