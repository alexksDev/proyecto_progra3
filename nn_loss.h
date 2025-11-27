#ifndef NN_LOSS_H
#define NN_LOSS_H

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec {
namespace neural_network {

// Mean Squared Error Loss: MSE = (1/n) * Σ(y_pred - y_true)²
template<typename T>
class MSELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> y_predicted_;
    utec::algebra::Tensor<T, 2> y_true_;
    
public:
    MSELoss(const utec::algebra::Tensor<T, 2>& y_prediction, 
            const utec::algebra::Tensor<T, 2>& y_true) 
        : y_predicted_(y_prediction), y_true_(y_true) {
        
        // Verificar que las dimensiones coincidan
        if (y_predicted_.shape() != y_true_.shape()) {
            throw std::invalid_argument("Predictions and true values must have the same shape");
        }
    }
    
    // Calcula el MSE
    T loss() const override {
        auto diff = y_predicted_ - y_true_;  // (y_pred - y_true)
        
        T sum = T{0};
        for (const auto& val : diff) {
            sum += val * val;  // Suma de cuadrados
        }
        
        return sum / static_cast<T>(y_predicted_.size());  // Promedio
    }
    
    // Gradiente: dL/dy_pred = (2/n) * (y_pred - y_true)
    utec::algebra::Tensor<T, 2> loss_gradient() const override {
        auto gradient = y_predicted_ - y_true_;
        T factor = T{2} / static_cast<T>(y_predicted_.size());
        
        gradient = gradient * factor;
        return gradient;
    }
};

// Binary Cross Entropy Loss: BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
template<typename T>
class BCELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> y_predicted_;
    utec::algebra::Tensor<T, 2> y_true_;
    static constexpr T epsilon = T{1e-15};  // Para evitar log(0)
    
public:
    BCELoss(const utec::algebra::Tensor<T, 2>& y_prediction, 
            const utec::algebra::Tensor<T, 2>& y_true) 
        : y_predicted_(y_prediction), y_true_(y_true) {
        
        // Verificar que las dimensiones coincidan
        if (y_predicted_.shape() != y_true_.shape()) {
            throw std::invalid_argument("Predictions and true values must have the same shape");
        }
    }
    
    // Calcula el BCE
    T loss() const override {
        T sum = T{0};
        
        auto it_pred = y_predicted_.cbegin();
        auto it_true = y_true_.cbegin();
        
        while (it_pred != y_predicted_.cend()) {
            T p = std::max(epsilon, std::min(T{1} - epsilon, *it_pred));  // Clip para estabilidad
            T y = *it_true;
            
            // BCE: -[y*log(p) + (1-y)*log(1-p)]
            sum += -(y * std::log(p) + (T{1} - y) * std::log(T{1} - p));
            
            ++it_pred;
            ++it_true;
        }
        
        return sum / static_cast<T>(y_predicted_.size());  // Promedio
    }
    
    // Gradiente: dL/dp = (1/n) * [(p - y) / (p * (1 - p))]
    utec::algebra::Tensor<T, 2> loss_gradient() const override {
        auto gradient = y_predicted_;
        
        auto it_grad = gradient.begin();
        auto it_pred = y_predicted_.cbegin();
        auto it_true = y_true_.cbegin();
        
        while (it_grad != gradient.end()) {
            T p = std::max(epsilon, std::min(T{1} - epsilon, *it_pred));  // Clip
            T y = *it_true;
            
            // Gradiente: (p - y) / (p * (1 - p))
            *it_grad = (p - y) / (p * (T{1} - p));
            *it_grad /= static_cast<T>(y_predicted_.size());  // Dividir por n
            
            ++it_grad;
            ++it_pred;
            ++it_true;
        }
        
        return gradient;
    }
};

} // namespace neural_network
} // namespace utec

#endif // NN_LOSS_H