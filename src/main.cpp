#include <iostream>
#include <string.h>
#include<set>
#include "matrix_mul.h"
using namespace std;


//this cause the unexpected behaviour becuse the intermidiate tensor is not stored.
// the error is due to the there is no way to compare the pointer so that's why we are getting error.
int main() {
  float data[2][2] = {{1, 2}, {3, 4}};
  float data2[2][2] = {{5, 6}, {7, 8}};

  float **data_ptr = (float **)malloc(2 * sizeof(float *));
  for (int i = 0; i < 2; i++) {
    data_ptr[i] = (float *)malloc(2 * sizeof(float));
    memcpy(data_ptr[i], data[i], 2 * sizeof(float));
  }

  Tensor t1(2, 2, data_ptr,{},"t1");
  Tensor t2(2, 2, data_ptr,{},"t2");

  Tensor t3 ;
  t3 = t1 + t2;
  t3 = t2 + t3;
  // for(int i = 0;i<10;i++){
  //   t3 = t3 + t2;
  // }
  set<Tensor > child1;
  for(const auto &c : t3.child){
    child1.insert(c);
  }
  set<Tensor > visited;
  cout<<"in main ----/"<<endl;
  
  for(auto &c :child1){
    if(visited.find(c) != visited.end()){
      cout<<"already visited "<<c.name<<endl;
      continue;
    }
    for(auto c1 : c.child){
      cout<<c1.name<<endl;

    }
      cout<<"Hello "<<endl;

    
    cout<<c.name<<endl;
    visited.insert(c);
  }
  cout<<"in main ----/"<<endl;


  for(int i = 0;i<t3.rows;i++){
    for(int j = 0;j<t3.cols;j++){
      cout<<t3.data[i][j]<<" ";
    }
    cout<<endl;
  }
  float **grad = (float ** )malloc(2 * sizeof(float *));

  for(int i = 0;i<t3.rows;i++){
    grad[i] = (float *)malloc(2 * sizeof(float));
    for(int j = 0;j<t3.cols;j++){
      grad[i][j] = 1;
    }
  }

  t3.grad = grad;

  t3.bacward();
  for(int i = 0;i<t1.rows;i++){
    for(int j = 0;j<t1.cols;j++){
      cout<<t2.grad[i][j]<<" ";
    }
    cout<<endl;
  }
  return 0;
}
