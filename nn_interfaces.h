#ifndef NN_INTERFACES_H
#define NN_INTERFACES_H

#include "tensor.h"

namespace utec {
namespace neural_network {

template<typename T> class IOptimizer;

// Interfaz base para capas de la red neuronal
template<typename T>
class ILayer {
public:
    virtual ~ILayer() = default;
    
    // Propagación hacia adelante
    virtual utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) = 0;
    
    // Propagación hacia atrás (backpropagation)
    virtual utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) = 0;
    
    // Actualización de parámetros (solo para capas con parámetros entrenables)
    virtual void update_params(IOptimizer<T>& optimizer) {}
};

// Interfaz base para funciones de pérdida (loss functions)
template<typename T, int N>
class ILoss {
public:
    virtual ~ILoss() = default;
    
    // Calcula el valor de la pérdida
    virtual T loss() const = 0;
    
    // Calcula el gradiente de la pérdida
    virtual utec::algebra::Tensor<T, N> loss_gradient() const = 0;
};

// Interfaz base para optimizadores
template<typename T>
class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    
    // Actualiza los parámetros usando los gradientes
    virtual void update(utec::algebra::Tensor<T, 2>& params, 
                       const utec::algebra::Tensor<T, 2>& grads) = 0;
    
    // Incrementa el paso del optimizador (usado en Adam)
    virtual void step() {}
};

} // namespace neural_network
} // namespace utec

#endif // NN_INTERFACES_H