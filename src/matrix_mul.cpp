#include "matrix_mul.h"
#include <memory>
#include <unordered_set>
#include <cstring>

Tensor& Tensor::operator=(const Tensor& t) {
    if (this == &t) {
        return *this;
    }

    // Create an entirely new tensor
    Tensor* new_tensor = new Tensor(t.rows, t.cols);
    
    // Copy the data from t
    for (int j = 0; j < t.rows; j++) {
        memcpy(new_tensor->data[j], t.data[j], t.cols * sizeof(float32));
    }

    // Copy properties from t
    new_tensor->name = t.name;
    new_tensor->_backward = t._backward;
    new_tensor->child = t.child;  // Share same children as t
    
    // Clean up current data
    data_holder.reset();
    grad_holder.reset();
    
    // Move the new tensor's contents into this
    // This is similar to Python's reference reassignment
    this->data_holder = std::move(new_tensor->data_holder);
    this->grad_holder = std::move(new_tensor->grad_holder);
    this->data = this->data_holder.get();
    this->grad = this->grad_holder.get();
    this->rows = new_tensor->rows;
    this->cols = new_tensor->cols;
    this->child = std::move(new_tensor->child);
    this->name = std::move(new_tensor->name);
    this->_backward = new_tensor->_backward;
    
    // Generate new UUID for this tensor
    uuid_generate(this->id);
    char uuid_str[37];
    uuid_unparse(this->id, uuid_str);
    this->uuidstr = std::string(uuid_str);
    
    delete new_tensor;
    return *this;
}

Tensor Tensor::operator+(const Tensor &t) const {
    Tensor result(this->rows, this->cols);
    
    // Store shared_ptr to the original tensors
    result.child.insert(std::shared_ptr<Tensor>(const_cast<Tensor*>(this), [](Tensor*) {}));
    result.child.insert(std::shared_ptr<Tensor>(const_cast<Tensor*>(&t), [](Tensor*) {}));
    
    result.name = this->name + "+" + t.name;
    for (int j = 0; j < rows; j++) {
        for (int k = 0; k < cols; k++) {
            result.data[j][k] = this->data[j][k] + t.data[j][k];
        }
    }
    result._backward = &Tensor::backadd;
    return result;
}

void Tensor::backadd() {
    for (auto& c : this->child) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                c->grad[i][j] = c->grad[i][j] + this->grad[i][j];
            }
        }
    }
}

Tensor Tensor::operator/(const Tensor &t) const {
    Tensor result(this->rows, this->cols);
    result.child.insert(std::shared_ptr<Tensor>(const_cast<Tensor*>(this), [](Tensor*) {}));
    result.child.insert(std::shared_ptr<Tensor>(const_cast<Tensor*>(&t), [](Tensor*) {}));
    result.name = this->name + "/" + t.name;

    for (int j = 0; j < rows; j++) {
        for (int k = 0; k < cols; k++) {
            result.data[j][k] = this->data[j][k] / t.data[j][k];
        }
    }
    return result;
}

Tensor Tensor::operator*(const Tensor &t) const {
    if (this->cols != t.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    Tensor result(this->rows, t.cols);
    result.child.insert(std::shared_ptr<Tensor>(const_cast<Tensor*>(this), [](Tensor*) {}));
    result.child.insert(std::shared_ptr<Tensor>(const_cast<Tensor*>(&t), [](Tensor*) {}));
    result.name = this->name + "*" + t.name;

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < t.cols; j++) {
            result.data[i][j] = 0;
            for (int k = 0; k < this->cols; k++) {
                result.data[i][j] += this->data[i][k] * t.data[k][j];
            }
        }
    }
    return result;
}

void Tensor::backmul(){
    
}

Tensor Tensor::operator^(const Tensor &t) const {
    if (this->rows != t.rows || this->cols != t.cols || this->batch != t.batch) {
        throw std::invalid_argument("Matrix dimensions do not match for dot multiplication");
    }

    Tensor result(t.cols, t.rows);
    result.child.insert(std::shared_ptr<Tensor>(const_cast<Tensor*>(this), [](Tensor*) {}));
    result.child.insert(std::shared_ptr<Tensor>(const_cast<Tensor*>(&t), [](Tensor*) {}));
    result.name = this->name + "^" + t.name;

    for (int i = 0; i < t.rows; i++) {
        for (int j = 0; j < t.cols; j++) {
            result.data[j][i] = t.data[i][j];
        }
    }
    return result;
}

void visit_tensor(const std::shared_ptr<Tensor>& t, std::set<std::string>& visited, std::vector<std::shared_ptr<Tensor>>& topo) {
    if (!t) {
        return;
    }
    
    if (visited.find(t->uuidstr) != visited.end()) {
        return;
    }
    
    visited.insert(t->uuidstr);
    
    for (const auto& child : t->child) {
        visit_tensor(child, visited, topo);
    }
    
    topo.push_back(t);
}

void Tensor::backward() {
    std::vector<std::shared_ptr<Tensor>> topo;
    std::set<std::string> visited;
    
    auto self = std::shared_ptr<Tensor>(this, [](Tensor*) {});
    visit_tensor(self, visited, topo);

    for (int i = topo.size() - 1; i >= 0; i--) {
        if (topo[i]->_backward) {
            (topo[i].get()->*(topo[i]->_backward))();
        }
    }
}
