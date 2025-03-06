#include "matrix_mul.h"
#include <memory>
#include <unordered_set>
#include <cstring>
#include <boost/smart_ptr/intrusive_ptr.hpp>

Tensor& Tensor::operator=(const Tensor& t) {
    if (this == &t) {
        return *this;
    }

    // Create an entirely new tensor
    Tensor* new_tensor = new Tensor(t.rows, t.cols);
    // std::cout << "New tensor created\n";
    // Copy the data from t
    for (int j = 0; j < t.rows; j++) {
        memcpy(new_tensor->data[j], t.data[j], t.cols * sizeof(float32));
    }

    // Copy properties from t
    new_tensor->name = t.name;
    new_tensor->_backward = t._backward;
    new_tensor->left = t.left;   // Share same children as t
    new_tensor->right = t.right;
    
    // Clean up current data
    data_holder.reset();
    grad_holder.reset();
    
    // Move the new tensor's contents into this
    // This is similar to Python's reference reassignment
    this->data_holder = std::move(new_tensor->data_holder);
    this->grad_holder = std::move(new_tensor->grad_holder);
    // this->data_holder = new_tensor->data_holder;
    // this->grad_holder = new_tensor->grad_holder;
    this->data = this->data_holder.get();
    this->grad = this->grad_holder.get();
    this->rows = new_tensor->rows;
    this->cols = new_tensor->cols;
    this->left = std::move(new_tensor->left);
    this->right = std::move(new_tensor->right);
    this->name = std::move(new_tensor->name);
    // this->left = new_tensor->left;
    // this->right = new_tensor->right;
    // this->name = new_tensor->name;
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
    
    // Store intrusive_ptr to the original tensors
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.right = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(&t));
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
    if (this->left) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                left->grad[i][j] = left->grad[i][j] + this->grad[i][j];
            }
        }
    }
    if (this->right) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                right->grad[i][j] = right->grad[i][j] + this->grad[i][j];
            }
        }
    }
}

Tensor Tensor::operator/(const Tensor &t) const {
    Tensor result(this->rows, this->cols);
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.right = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(&t));
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
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.right = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(&t));
    result.name = this->name + "*" + t.name;
    result._backward = &Tensor::backmul;
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
    // A = B * C
    // $A = B*C$
    // $B = (3*2)$
    // C = (2x2)
    // dL/dA (3x2)
    // dL/dB = dL/dA * C^T
    // dL/dC = B^T * dL/dA
    if(this->left){
        for(int i = 0; i<this->rows;i++){
            for(int j = 0;j<this->right->rows;j++){
                for(int k = 0;k<this->cols;k++){
                    left->grad[i][j] += this->grad[i][k] * right->data[j][k];
                }
            }
        }
    }
    if(this->right){
        for(int i = 0;i<this->left->cols;i++){
            for(int j = 0;j<this->cols;j++){
                for(int k = 0;k<this->rows;k++){
                    right->grad[i][j] += this->grad[k][j] * left->data[k][i];
                }
            }
        }
    }
}

Tensor Tensor::operator^(const Tensor &t) const {
    if (this->rows != t.rows || this->cols != t.cols || this->batch != t.batch) {
        throw std::invalid_argument("Matrix dimensions do not match for dot multiplication");
    }

    Tensor result(t.cols, t.rows);
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.right = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(&t));
    result.name = this->name + "^" + t.name;

    for (int i = 0; i < t.rows; i++) {
        for (int j = 0; j < t.cols; j++) {
            result.data[j][i] = t.data[i][j];
        }
    }
    return result;
}

void visit_tensor(const boost::intrusive_ptr<Tensor>& t,
                 std::set<boost::intrusive_ptr<Tensor>>& visited,
                 std::vector<boost::intrusive_ptr<Tensor>>& topo) {
    if (!t) {
        return;
    }
    
    if (visited.find(t) != visited.end()) {
        return;
    }
    
    visited.insert(t);
    
    if (t->left) {
        visit_tensor(t->left, visited, topo);
    }
    if (t->right) {
        visit_tensor(t->right, visited, topo);
    }
    
    topo.push_back(t);
}

void Tensor::backward() {
    std::vector<boost::intrusive_ptr<Tensor>> topo;
    std::set<boost::intrusive_ptr<Tensor>> visited;
    
    auto self = boost::intrusive_ptr<Tensor>(this);
    visit_tensor(self, visited, topo);

    for (int i = topo.size() - 1; i >= 0; i--) {
        if (topo[i]->_backward) {
            (topo[i].get()->*(topo[i]->_backward))();
        }
    }
}
