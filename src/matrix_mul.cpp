#include "matrix_mul.h"
#include <memory>
#include <unordered_set>
#include <cstring>
#include <cmath>   
#include <boost/smart_ptr/intrusive_ptr.hpp>

const float CLIP_NORM = 1.0f;
const float MIN_GRAD_NORM = 1e-3f;  
const float EPSILON = 1e-6f;        

void clip_gradient(float** grad, int rows, int cols) {
    float norm = 0.0f;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!std::isfinite(grad[i][j])) { 
                grad[i][j] = 0.0f;
            }
            norm += grad[i][j] * grad[i][j];
        }
    }
    norm = sqrt(norm + EPSILON); 
    if (norm > CLIP_NORM) {
        float scale = CLIP_NORM / norm;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                grad[i][j] *= scale;
            }
        }
    }
    else if (norm < MIN_GRAD_NORM) {
        float scale = MIN_GRAD_NORM / (norm + EPSILON);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                grad[i][j] *= scale;  
            }
        }
    }
}

Tensor& Tensor::operator=(const Tensor& t) {
    if (this == &t) {
        return *this;
    }

    Tensor* new_tensor = new Tensor(t.rows, t.cols);

    for (int j = 0; j < t.rows; j++) {
        memcpy(new_tensor->data[j], t.data[j], t.cols * sizeof(float32));
    }

    new_tensor->name = t.name;
    new_tensor->_backward = t._backward;
    new_tensor->left = t.left;   
    new_tensor->right = t.right;
    
    data_holder.reset();
    grad_holder.reset();

    this->data_holder = std::move(new_tensor->data_holder);
    this->grad_holder = std::move(new_tensor->grad_holder);

    this->data = this->data_holder.get();
    this->grad = this->grad_holder.get();
    this->rows = new_tensor->rows;
    this->cols = new_tensor->cols;
    this->left = std::move(new_tensor->left);
    this->right = std::move(new_tensor->right);
    this->name = std::move(new_tensor->name);

    this->_backward = new_tensor->_backward;

    uuid_generate(this->id);
    char uuid_str[37];
    uuid_unparse(this->id, uuid_str);
    this->uuidstr = std::string(uuid_str);
    
    delete new_tensor;
    return *this;
}

Tensor Tensor::operator+(const Tensor &t) const {
    Tensor result(this->rows, this->cols);

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

Tensor Tensor::operator-(const Tensor &t) const {
    Tensor result(this->rows, this->cols);
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.right = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(&t));
    result.name = this->name + "-" + t.name;

    for (int j = 0; j < rows; j++) {
        for (int k = 0; k < cols; k++) {
            result.data[j][k] = this->data[j][k] - t.data[j][k];
        }
    }
    result._backward = &Tensor::backsub;
    return result;
}

void Tensor::backsub(){
    if(this->left){
        for(int i = 0;i<this->rows;i++){
            for(int j = 0;j<this->cols;j++){
                left->grad[i][j] = left->grad[i][j] + this->grad[i][j];
            }
        }
        clip_gradient(left->grad,this->left->rows, this->left->cols);
    }
    if(this->right){
        for(int i = 0;i<this->rows;i++){
            for(int j = 0;j<this->cols;j++){
                right->grad[i][j] = right->grad[i][j] - this->grad[i][j];
            }
        }
        clip_gradient(right->grad, this->right->rows, this->right->cols);
    }
}

void Tensor::backadd() {
    if (this->left) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                left->grad[i][j] = left->grad[i][j] + this->grad[i][j];
            }
        }
        clip_gradient(left->grad,this->left->rows, this->left->cols);
    }
    if (this->right) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                right->grad[i][j] = right->grad[i][j] + this->grad[i][j];
            }
        }
        clip_gradient(right->grad, this->right->rows, this->right->cols);
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
        clip_gradient(left->grad,this->left->rows, this->left->cols);
    }
    if(this->right){
        for(int i = 0;i<this->left->cols;i++){
            for(int j = 0;j<this->cols;j++){
                for(int k = 0;k<this->rows;k++){
                    right->grad[i][j] += this->grad[k][j] * left->data[k][i];
                }
            }
        }
        clip_gradient(right->grad, this->right->rows, this->right->cols);
    }
}

Tensor Tensor::operator^(const Tensor &t) const {
    if (this->cols != 1 || t.cols !=1 || this ->rows != t.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for dot multiplication");
    }

    Tensor result(t.cols, t.cols);
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.right = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(&t));
    result.name = this->name + "^" + t.name;

    for (int i = 0;i<this->rows;i++){
        result.data[0][0] += this->data[i][0] * t.data[i][0];
    }
    result._backward = &Tensor::backdot;
    return result;
}

void Tensor::backdot(){
    if(this->left){
        for(int i = 0;i<this->left->rows;i++){
            left->grad[i][0] += ((this->grad[0][0] * right->data[i][0]));
        }
        clip_gradient(left->grad,this->left->rows, this->left->cols);
    }
    if(this->right){
        for(int i = 0;i<this->right->rows;i++){
            right->grad[i][0] += (this->grad[0][0] * left->data[i][0]);
        }
        clip_gradient(right->grad, this->right->rows, this->right->cols);
    }
}

Tensor Tensor::lekyrelu(float leaky){
    Tensor result(this->rows, this->cols);
    result.left = boost::intrusive_ptr<Tensor>(const_cast<Tensor*>(this));
    result.name = this->name + "leakyrelu";
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            result.data[i][j] = this->data[i][j] > 0 ? this->data[i][j] : leaky * this->data[i][j];
        }
    }
    result._backward = &Tensor::backleakyrelu;
    return result;
}

void Tensor::backleakyrelu() {
    if (this->left) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                left->grad[i][j] += (this->data[i][j]) > 0 ? this->grad[i][j] : 0.01 * this->grad[i][j];
            }
        }
        clip_gradient(left->grad,this->left->rows, this->left->cols);
    }
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
        // Free left and right child tensors after computation
    }
    for(int i = 0;i<topo.size();i++){
        topo[i]->left = nullptr;
        topo[i]->right = nullptr;
        topo[i]->_backward = nullptr;
    }
}


