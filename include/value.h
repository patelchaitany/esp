#pragma once

#include <iostream>
#include <memory>
#include <matrix_mul.h>
#include <sys/resource.h>
#include <cstring>
#include <cstdlib>
#include <set>
#include <string>
#include <cmath>
#include <random>
#include <boost/smart_ptr/intrusive_ptr.hpp>
class Value
{
public:
    mutable boost::intrusive_ptr<Tensor> ptr;
    boost::intrusive_ptr<Tensor> orig;

public:
    Value()
    {
        ptr = nullptr;
        orig = nullptr;
    };

    explicit Value(Tensor *t)
    {
        ptr = t;
    }

    Value(int row, int cols, float **data, std::string name)
    {
        ptr = boost::intrusive_ptr<Tensor>(new Tensor(row, cols, data, name));
        orig = ptr;
    }

    Value &operator=(const Value &other)
    {
        if (this != &other)
        {
            ptr = other.ptr;
        }
        return *this;
    }

    Value operator+(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr + *other.ptr));
    }

    Value operator*(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr * *other.ptr));
    }

    Value operator^(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr ^ *other.ptr));
    }

    Value operator/(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr / *other.ptr));
    }
    Value operator-(const Value &other) const
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(*ptr - *other.ptr));
    }
    Value leakyrelu(float leaky = 0.01)
    {
        if (ptr->_backward == nullptr && orig != nullptr)
        {
            this->ptr = this->orig;
        }
        return Value(new Tensor(ptr->lekyrelu(leaky)));
    }
    void setgrad(float **grad)
    {
        ptr->setGrad(grad);
    }
    void setgradzero()
    {
        if(this->ptr){ptr->setgradzero();}
        if(this->orig){orig->setgradzero();}
    }
    void backward()
    {
        ptr->backward();
    }

    void printgrad()
    {
        if (orig == nullptr)
        {
            for (int i = 0; i < ptr->rows; i++)
            {
                for (int j = 0; j < ptr->cols; j++)
                {
                    std::cout << ptr->grad[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            for (int i = 0; i < orig->rows; i++)
            {
                for (int j = 0; j < orig->cols; j++)
                {
                    std::cout << orig->grad[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    void printdata()
    {
        if (orig != nullptr)
        {
            std::cout<<"Original Data \n"<<orig->name<<std::endl;
            for (int i = 0; i < orig->rows; i++)
            {
                for (int j = 0; j < orig->cols; j++)
                {
                    std::cout << orig->data[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            std::cout<<"Data "<<ptr->name<<std::endl;
            for (int i = 0; i < ptr->rows; i++)
            {
                for (int j = 0; j < ptr->cols; j++)
                {
                    std::cout << ptr->data[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    void update(float learning_rate)
    {
        this->ptr = this->orig;
        orig->update(learning_rate);
    }
};