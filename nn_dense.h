#ifndef NN_DENSE_H
#define NN_DENSE_H

#include "nn_interfaces.h"
#include "tensor.h"

// Alias global para facilitar el uso
template<typename T, size_t N>
using Tensor = utec::algebra::Tensor<T, N>;

namespace utec {
namespace neural_network {

// Capa Dense (Fully Connected): Y = X * W + b
template<typename T>
class Dense final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> weights_;      // Matriz de pesos (in_features x out_features)
    utec::algebra::Tensor<T, 2> biases_;       // Vector de sesgos (1 x out_features)
    
    utec::algebra::Tensor<T, 2> input_;        // Guardamos entrada para backward
    utec::algebra::Tensor<T, 2> grad_weights_; // Gradiente de pesos
    utec::algebra::Tensor<T, 2> grad_biases_;  // Gradiente de sesgos
    
    size_t in_features_;
    size_t out_features_;
    
public:
    // Constructor con inicializadores personalizados
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
        : in_features_(in_f), out_features_(out_f),
          weights_(in_f, out_f),
          biases_(1, out_f),
          grad_weights_(in_f, out_f),
          grad_biases_(1, out_f) {
        
        // Inicializar pesos y sesgos usando las funciones proporcionadas
        init_w_fun(weights_);
        init_b_fun(biases_);
        
        grad_weights_.fill(T{0});
        grad_biases_.fill(T{0});
    }
    
    // Forward: Y = X * W + b
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) override {
        input_ = x;  // Guardar para backward
        
        // Multiplicación matricial: (batch_size x in_features) * (in_features x out_features)
        auto output = utec::algebra::matrix_product(x, weights_);
        
        // Sumar bias (broadcasting automático)
        output = output + biases_;
        
        return output;
    }
    
    // Backward: calcula gradientes respecto a entrada, pesos y sesgos
    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& dZ) override {
        // dZ tiene forma (batch_size x out_features)
        
        // Gradiente respecto a los pesos: dW = X^T * dZ
        // X tiene forma (batch_size x in_features)
        // X^T tiene forma (in_features x batch_size)
        // dZ tiene forma (batch_size x out_features)
        // Resultado: (in_features x out_features)
        auto input_T = utec::algebra::transpose_2d(input_);
        grad_weights_ = utec::algebra::matrix_product(input_T, dZ);
        
        // Gradiente respecto a los sesgos: db = sum(dZ, axis=0)
        // Sumamos a lo largo del batch
        const auto shape_dZ = dZ.shape();
        size_t batch_size = shape_dZ[0];
        size_t out_f = shape_dZ[1];
        
        grad_biases_.fill(T{0});
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_f; ++j) {
                grad_biases_(0, j) += dZ(i, j);
            }
        }
        
        // Gradiente respecto a la entrada: dX = dZ * W^T
        // dZ tiene forma (batch_size x out_features)
        // W^T tiene forma (out_features x in_features)
        // Resultado: (batch_size x in_features)
        auto weights_T = utec::algebra::transpose_2d(weights_);
        auto dX = utec::algebra::matrix_product(dZ, weights_T);
        
        return dX;
    }
    
    // Actualiza pesos y sesgos usando el optimizador
    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(weights_, grad_weights_);
        optimizer.update(biases_, grad_biases_);
    }
};

} // namespace neural_network
} // namespace utec

#endif // NN_DENSE_H