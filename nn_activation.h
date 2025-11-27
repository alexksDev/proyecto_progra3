#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>
#include <algorithm>

namespace utec {
namespace neural_network {

// Función de activación ReLU: f(x) = max(0, x)
template<typename T>
class ReLU final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> input_;  // Guardamos la entrada para backward
    
public:
    // Forward: aplica ReLU elemento a elemento
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
        input_ = z;  // Guardamos para usar en backward
        
        auto result = z;
        // ReLU: si x < 0 entonces 0, sino x
        for (auto& val : result) {
            val = std::max(T{0}, val);
        }
        return result;
    }
    
    // Backward: derivada de ReLU es 1 si x > 0, sino 0
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) override {
        auto result = gradient;
        
        auto it_grad = result.begin();
        auto it_input = input_.cbegin();
        
        while (it_grad != result.end()) {
            // Si la entrada era <= 0, el gradiente se hace 0
            if (*it_input < T{0}) {
                *it_grad = T{0};
            }
            ++it_grad;
            ++it_input;
        }
        
        return result;
    }
};

// Función de activación Sigmoid: f(x) = 1 / (1 + e^(-x))
template<typename T>
class Sigmoid final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> output_;  // Guardamos la salida para backward
    
public:
    // Forward: aplica sigmoid elemento a elemento
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
        auto result = z;
        
        for (auto& val : result) {
            // Sigmoid: 1 / (1 + e^(-x))
            val = T{1} / (T{1} + std::exp(-val));
        }
        
        output_ = result;  // Guardamos para usar en backward
        return result;
    }
    
    // Backward: derivada de sigmoid es sigmoid(x) * (1 - sigmoid(x))
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) override {
        auto result = gradient;
        
        auto it_grad = result.begin();
        auto it_output = output_.cbegin();
        
        while (it_grad != result.end()) {
            // Derivada: sigmoid(x) * (1 - sigmoid(x))
            T sigmoid_val = *it_output;
            *it_grad = (*it_grad) * sigmoid_val * (T{1} - sigmoid_val);
            ++it_grad;
            ++it_output;
        }
        
        return result;
    }
};

} // namespace neural_network
} // namespace utec

#endif // NN_ACTIVATION_H