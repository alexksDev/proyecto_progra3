#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_activation.h"
#include "nn_dense.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include "tensor.h"
#include <vector>
#include <memory>
#include <iostream>

namespace utec {
namespace neural_network {

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;
    
public:
    // Agregar una capa a la red
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }
    
    // Entrenamiento de la red
    template<template<typename> class LossType, 
             template<typename> class OptimizerType = SGD>
    void train(const utec::algebra::Tensor<T, 2>& X,
               const utec::algebra::Tensor<T, 2>& Y,
               const size_t epochs,
               const size_t batch_size,
               T learning_rate) {
        
        // Crear optimizador
        OptimizerType<T> optimizer(learning_rate);
        
        const auto shape_X = X.shape();
        const size_t total_samples = shape_X[0];
        
        // Entrenar por épocas
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            T total_loss = T{0};
            size_t num_batches = 0;
            
            // Procesar por batches
            for (size_t batch_start = 0; batch_start < total_samples; batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, total_samples);
                size_t current_batch_size = batch_end - batch_start;
                
                // Extraer batch
                auto X_batch = extract_batch(X, batch_start, current_batch_size);
                auto Y_batch = extract_batch(Y, batch_start, current_batch_size);
                
                // Forward pass
                auto predictions = forward(X_batch);
                
                // Calcular pérdida
                LossType<T> loss_fn(predictions, Y_batch);
                T loss = loss_fn.loss();
                total_loss += loss;
                ++num_batches;
                
                // Backward pass
                auto gradient = loss_fn.loss_gradient();
                backward(gradient);
                
                // Actualizar parámetros
                update_parameters(optimizer);
            }
            
            /*
            // Imprimir pérdida cada 100 épocas
            if ((epoch + 1) % 100 == 0) {
                T avg_loss = total_loss / static_cast<T>(num_batches);
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                          << " - Loss: " << avg_loss << std::endl;
            }*/
        }
    }
    
    // Realizar predicciones
    utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
        return forward(X);
    }
    
private:
    // Forward pass a través de todas las capas
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) {
        auto output = input;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }
    
    // Backward pass a través de todas las capas (en orden inverso)
    void backward(const utec::algebra::Tensor<T, 2>& gradient) {
        auto grad = gradient;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
    }
    
    // Actualizar parámetros de todas las capas
    void update_parameters(IOptimizer<T>& optimizer) {
        for (auto& layer : layers_) {
            layer->update_params(optimizer);
        }
        optimizer.step();  // Incrementar paso del optimizador (para Adam)
    }
    
    // Extraer un batch del tensor
    utec::algebra::Tensor<T, 2> extract_batch(const utec::algebra::Tensor<T, 2>& data, 
                                               size_t start, 
                                               size_t size) {
        const auto shape = data.shape();
        size_t features = shape[1];
        
        utec::algebra::Tensor<T, 2> batch(size, features);
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < features; ++j) {
                batch(i, j) = data(start + i, j);
            }
        }
        
        return batch;
    }
};

} // namespace neural_network
} // namespace utec

#endif // NEURAL_NETWORK_H