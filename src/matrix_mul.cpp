#include "matrix_mul.h"
#include <memory>
#include <unordered_set>
#include <cstring>


Tensor &Tensor::operator=(const Tensor &t)
{
    if (this == &t)
        return *this;

    for (int j = 0; j < rows; j++)
    {
        free(data[j]);
    }
    free(data);
    uuid_copy(this->id, t.id);
    this->rows = t.rows;
    this->cols = t.cols;
    this->batch = t.batch;
    this->name = t.name;
    this->uuidstr = t.uuidstr;
    this->_backward = t._backward; // Clear existing children
    this->grad = (float32 **)malloc(rows * sizeof(float32 *));
    for(auto& c : t.child) {
        this->child.insert(c);  // Insert each child individually
    }
    data = (float32 **)malloc(rows * sizeof(float32 **));
    for (int j = 0; j < rows; j++)
    {
        data[j] = (float32 *)malloc(cols * sizeof(float32));
        grad[j] = (float32 *)malloc(cols * sizeof(float32));
        memcpy(data[j], t.data[j], cols * sizeof(float32));
        memcpy(grad[j], t.grad[j], cols * sizeof(float32));
    }

    return *this;
}

Tensor Tensor::operator+(const Tensor &t) const
{
    Tensor result(this->rows, this->cols);
    result.child = {*this, t};
    result.name = this->name + "+" + t.name;
    std::cout<<"-----------------"<<std::endl;
    std::cout<<result.name<<" "<<result.uuidstr<<" "<<t.uuidstr<<" "<<this->uuidstr<<std::endl;
    for(const auto &c : result.child){
        std::cout<<c.name<<" "<<c.uuidstr<<std::endl;
    }
    std::cout<<"-----------------"<<std::endl;

    for (int j = 0; j < rows; j++)
    {
        for (int k = 0; k < cols; k++)
        {
            result.data[j][k] = this->data[j][k] + t.data[j][k];
        }
    }
    // result._backward = [&](){
    //     std::cout<<"Backward called : >"<<this->name<<"< >"<<t.name<<" <"<<std::endl;

    //     for(int i = 0;i<this->rows;i++){
    //         for(int j = 0;j<this->cols;j++){
    //             std::cout<<result.grad[i][j]<<" ";
    //             this->grad[i][j] = this->grad[i][j] + result.grad[i][j];
    //             t.grad[i][j] = t.grad[i][j] + result.grad[i][j];
    //         }
    //     }
    //     std::cout<<"-----*--------"<<std::endl;
    // };
    result._backward = &Tensor::backadd;
    return result;
}
void Tensor::backadd(){
    for(auto &c : this->child){
        for(int i = 0;i<this->rows;i++){
            for(int j = 0;j<this->cols;j++){
                c.grad[i][j] = c.grad[i][j] + this->grad[i][j];
            }
        }
    }
}
Tensor Tensor::operator/(const Tensor &t) const
{
    Tensor result(this->rows, this->cols, {this, &t});
    result.name = this->name + "/" + t.name;
    for (int j = 0; j < rows; j++)
    {
        for (int k = 0; k < cols; k++)
        {
            result.data[j][k] = this->data[j][k] / t.data[j][k];
        }
    }
    return result;
}

Tensor Tensor::operator*(const Tensor &t) const
{
    if (this->cols != t.rows)
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    Tensor result(this->rows, t.cols, {this, &t});
    result.name = this->name + "*" + t.name;
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < t.cols; j++)
        {
            result.data[i][j] = 0;
            for (int k = 0; k < this->cols; k++)
            {
                result.data[i][j] += this->data[i][k] * t.data[k][j];
            }
        }
    }
    //A = B*C 
    // B = (3x2)
    // C = (2x2)
    //dL/dA (3x2)
    //dL/dB = dL/dA * C^T
    //dL/dC = B^T * dL/dA
    float32 **temp_grad = this->grad;
    // result._backward = [&](){

    //     for(int i = 0;i<result.rows;i++){
    //         for(int j = 0;j<t.rows;j++){
    //             for(int k = 0;k<t.cols;k++){
    //                 temp_grad[i][j] += result.grad[i][k] * t.data[k][j];
    //              }
    //         }
    //     }
    //     for (int i = 0; i < this->rows; i++) {
    //         for (int j = 0; j < this->cols; j++) {
    //             this->grad[i][j] = temp_grad[i][j];
    //         }
    //     }
    //     temp_grad = t.grad;

    //     for(int i = 0;i<this->cols;i++){
    //         for(int j = 0;j<result.cols;j++){
    //             for(int k = 0;k<result.rows;k++){
    //                 temp_grad[i][j] += this->data[k][i] * result.grad[k][j];
    //              }
    //         }
    //     }

    //     for (int i = 0; i < t.rows; i++) {
    //         for (int j = 0; j < t.cols; j++) {
    //             t.grad[i][j] = temp_grad[i][j];
    //         }
    //     }
        
    // };
    return result;
}


Tensor Tensor::operator^(const Tensor &t) const
{
    if (this->rows != t.rows || this->cols != t.cols || this->batch != t.batch)
    {
        throw std::invalid_argument("Matrix dimensions do not match for dot multiplication");
    };

    Tensor result(t.cols, t.rows, {this, &t});
    result.name = this->name + "^" + t.name;
    for (int i = 0; i < t.rows; i++)
    {
        for (int j = 0; j < t.cols; j++)
        {
            result.data[j][i] = t.data[i][j];
        }
    }

    // result._backward = [&](){
    //     std::cout<<"Backward called"<<std::endl;
    // };

    return result;
}


void visit_tensor(Tensor* t, std::set<std::string>& visited, std::vector<Tensor*>& topo) {
    if (!t) {
        return;
    }
    
    if (visited.find(t->uuidstr) != visited.end()) {
        return;
    }
    
    visited.insert(t->uuidstr);
    
    for (auto& child : t->child) {
        visit_tensor(const_cast<Tensor*>(&child), visited, topo);
    }
    
    topo.push_back(t);
}
void Tensor::bacward()
{

    std::vector<Tensor*> topo;
    std::set<std::string> visited;
    
    // Call the helper function to build the topology
    visit_tensor(this, visited, topo);
    for (int i = topo.size() - 1; i >= 0; i--) {
        if (topo[i]->_backward) {
            // (topo[i]->*(_backward))(); 
            (topo[i]->*(topo[i]->_backward))();
    }
    }
}
