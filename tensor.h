#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <initializer_list>
#include <numeric>
#include <algorithm>

namespace utec {
namespace algebra {

template<typename T, size_t N>
class Tensor {
private:
    std::array<size_t, N> dims_;  // Almacena las dimensiones del tensor
    std::vector<T> data_;         // Almacena los datos en memoria contigua
    
    // Calcula el tamaño total del tensor (producto de todas las dimensiones)
    size_t calculate_size() const {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) {
            total *= dims_[i];
        }
        return total;
    }
    
    // Convierte índices multidimensionales a un índice lineal
    // Ejemplo: (i,j,k) -> índice único en el vector data_
    size_t get_index(const std::array<size_t, N>& indices) const {
        size_t index = 0;
        size_t multiplier = 1;
        for (int i = N - 1; i >= 0; --i) {
            index += indices[i] * multiplier;
            multiplier *= dims_[i];
        }
        return index;
    }
    
    // Verifica si dos tensores son compatibles para broadcasting
    // Broadcasting: operar tensores de diferentes formas
    // Ejemplo: (3,4) + (3,1) es válido, (3,4) + (2,4) no lo es
    bool is_broadcastable(const Tensor<T, N>& other) const {
        for (int i = N - 1; i >= 0; --i) {
            if (dims_[i] != other.dims_[i] && dims_[i] != 1 && other.dims_[i] != 1) {
                return false;
            }
        }
        return true;
    }
    
    // Calcula la forma resultante después de broadcasting
    // Toma el máximo de cada dimensión
    std::array<size_t, N> broadcast_shape(const Tensor<T, N>& other) const {
        std::array<size_t, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = std::max(dims_[i], other.dims_[i]);
        }
        return result;
    }
    
    // Mapea un índice del tensor expandido al tensor original
    // Si una dimensión es 1, siempre usa índice 0 (broadcasting)
    size_t broadcast_index(size_t linear_idx, const std::array<size_t, N>& original_dims, 
                          const std::array<size_t, N>& broadcast_dims) const {
        std::array<size_t, N> indices;
        size_t temp = linear_idx;
        
        // Convierte índice lineal a índices multidimensionales
        for (int i = N - 1; i >= 0; --i) {
            indices[i] = temp % broadcast_dims[i];
            temp /= broadcast_dims[i];
        }
        
        // Ajusta índices para dimensiones de tamaño 1 (broadcasting)
        for (size_t i = 0; i < N; ++i) {
            if (original_dims[i] == 1) {
                indices[i] = 0;
            }
        }
        
        // Convierte de vuelta a índice lineal
        size_t result = 0;
        size_t multiplier = 1;
        for (int i = N - 1; i >= 0; --i) {
            result += indices[i] * multiplier;
            multiplier *= original_dims[i];
        }
        return result;
    }

public:
    // Constructor: crea tensor con dimensiones especificadas
    // Ejemplo: Tensor<int, 3> t(2, 3, 4) crea tensor 2x3x4

    Tensor() {
        dims_.fill(0);
        data_.clear();
    }

    explicit Tensor(const std::array<size_t, N>& dimensions) : dims_(dimensions) {
        data_.resize(calculate_size(), T{});
    }

    template<typename Func>
    Tensor<T, N> apply(Func func) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) {
            val = func(val);
        }
        return result;
    }

    template<typename... Dims>
    Tensor(Dims... dimensions) {
        static_assert(sizeof...(Dims) == N, 
                    "Number of constructor arguments must match tensor dimensions");
        
        size_t temp_dims[] = {static_cast<size_t>(dimensions)...};
        for (size_t i = 0; i < N; ++i) {
            dims_[i] = temp_dims[i];
        }
        
        data_.resize(calculate_size(), T{});
    }

    //agregamos este nuevo metodo
    size_t size() const {
        return data_.size();
    }
    
    // Retorna las dimensiones del tensor
    const std::array<size_t, N>& shape() const {
        return dims_;
    }
    
    // Llena todo el tensor con un valor específico
    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }
    
    // Asigna valores desde una lista inicializadora
    // Verifica que el tamaño coincida
    Tensor<T, N>& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size()) {
            throw std::invalid_argument("Data size does not match tensor size");
        }
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }
    
    // Acceso a elementos: t(i, j, k)
    template<typename... Indices>
    T& operator()(Indices... indices) {
        std::array<size_t, N> idx_array = {static_cast<size_t>(indices)...};
        return data_[get_index(idx_array)];
    }
    
    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        std::array<size_t, N> idx_array = {static_cast<size_t>(indices)...};
        return data_[get_index(idx_array)];
    }
    
    // Cambia las dimensiones del tensor manteniendo los datos
    // Puede aumentar o disminuir el tamaño total
    template<typename... Dims>
    void reshape(Dims... new_dimensions) {
        constexpr size_t num_dims = sizeof...(Dims);
        if (num_dims != N) {
            throw std::invalid_argument("Number of dimensions do not match with 2");
        }
        
        size_t temp_dims[] = {static_cast<size_t>(new_dimensions)...};
        size_t new_size = 1;
        for (size_t i = 0; i < N; ++i) {
            new_size *= temp_dims[i];
        }
        
        for (size_t i = 0; i < N; ++i) {
            dims_[i] = temp_dims[i];
        }
        
        data_.resize(new_size, T{});  // Ajusta tamaño, rellena con T{} si crece
    }
    
    // Suma de tensores: elemento a elemento
    // Soporta broadcasting si las formas son compatibles
    Tensor<T, N> operator+(const Tensor<T, N>& other) const {
        // Caso simple: mismas dimensiones
        if (dims_ == other.dims_) {
            Tensor<T, N> result = *this;
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] += other.data_[i];
            }
            return result;
        }
        
        // Broadcasting
        if (!is_broadcastable(other)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        
        auto new_dims = broadcast_shape(other);
        Tensor<T, N> result(new_dims);
        
        size_t total_size = 1;
        for (size_t d : new_dims) total_size *= d;
        
        for (size_t i = 0; i < total_size; ++i) {
            size_t idx1 = broadcast_index(i, dims_, new_dims);
            size_t idx2 = broadcast_index(i, other.dims_, new_dims);
            result.data_[i] = data_[idx1] + other.data_[idx2];
        }
        
        return result;
    }
    
    // Resta de tensores: similar a la suma
    Tensor<T, N> operator-(const Tensor<T, N>& other) const {
        if (dims_ == other.dims_) {
            Tensor<T, N> result = *this;
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] -= other.data_[i];
            }
            return result;
        }
        
        if (!is_broadcastable(other)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        
        auto new_dims = broadcast_shape(other);
        Tensor<T, N> result(new_dims);
        
        size_t total_size = 1;
        for (size_t d : new_dims) total_size *= d;
        
        for (size_t i = 0; i < total_size; ++i) {
            size_t idx1 = broadcast_index(i, dims_, new_dims);
            size_t idx2 = broadcast_index(i, other.dims_, new_dims);
            result.data_[i] = data_[idx1] - other.data_[idx2];
        }
        
        return result;
    }
    
    // Multiplicación elemento a elemento (Hadamard product)
    // NO es multiplicación matricial
    Tensor<T, N> operator*(const Tensor<T, N>& other) const {
        if (dims_ == other.dims_) {
            Tensor<T, N> result = *this;
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] *= other.data_[i];
            }
            return result;
        }
        
        if (!is_broadcastable(other)) {
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        
        auto new_dims = broadcast_shape(other);
        Tensor<T, N> result(new_dims);
        
        size_t total_size = 1;
        for (size_t d : new_dims) total_size *= d;
        
        for (size_t i = 0; i < total_size; ++i) {
            size_t idx1 = broadcast_index(i, dims_, new_dims);
            size_t idx2 = broadcast_index(i, other.dims_, new_dims);
            result.data_[i] = data_[idx1] * other.data_[idx2];
        }
        
        return result;
    }
    
    // Operaciones con escalares: aplica operación a todos los elementos
    Tensor<T, N> operator+(const T& scalar) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) {
            val += scalar;
        }
        return result;
    }
    
    Tensor<T, N> operator-(const T& scalar) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) {
            val -= scalar;
        }
        return result;
    }
    
    Tensor<T, N> operator*(const T& scalar) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) {
            val *= scalar;
        }
        return result;
    }
    
    Tensor<T, N> operator/(const T& scalar) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) {
            val /= scalar;
        }
        return result;
    }
    
    // Iteradores para usar con STL (std::iota, std::accumulate, etc.)
    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator cbegin() const { return data_.cbegin(); }
    typename std::vector<T>::const_iterator cend() const { return data_.cend(); }
    
    // Imprime el tensor en formato anidado
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T, N>& tensor) {
        tensor.print_recursive(os, 0, 0);
        return os;
    }
    
private:
    // Función auxiliar recursiva para imprimir el tensor
    // Maneja la indentación y formato de corchetes
    void print_recursive(std::ostream& os, size_t dim, size_t offset) const {
        if (dim == N - 1) {
            // Última dimensión: imprime los valores
            
            for (size_t i = 0; i < dims_[dim]; ++i) {
                os << data_[offset + i];
                if (i < dims_[dim] - 1) os << " ";
            }
            os << "\n";
        } else {
            // Dimensiones intermedias: llamada recursiva
            
            os << "{\n";
            size_t stride = 1;
            for (size_t i = dim + 1; i < N; ++i) {
                stride *= dims_[i];
            }
            
            for (size_t i = 0; i < dims_[dim]; ++i) {
                
                print_recursive(os, dim + 1, offset + i * stride);
            }
            os << "}\n";
        }
    }
    
    
    template<typename U, size_t M>
    friend Tensor<U, M> operator+(const U& scalar, const Tensor<U, M>& tensor);
    
    template<typename U, size_t M>
    friend Tensor<U, M> transpose_2d(const Tensor<U, M>& tensor);
    
    template<typename U, size_t M>
    friend Tensor<U, M> matrix_product(const Tensor<U, M>& t1, const Tensor<U, M>& t2);
};

// Operadores globales para escalar + tensor
template<typename T, size_t N>
Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor + scalar;
}

template<typename T, size_t N>
Tensor<T, N> operator*(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor * scalar;
}

// Transpone las últimas dos dimensiones del tensor
// Para tensor 3D (batch, rows, cols) -> (batch, cols, rows)
template<typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& tensor) {
    if (N < 2) {
        throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
    }
    
    auto new_dims = tensor.dims_;
    std::swap(new_dims[N-2], new_dims[N-1]);  // Intercambia las últimas dos dimensiones
    
    Tensor<T, N> result(new_dims);
    
    // Calcula el tamaño del batch (todas las dimensiones excepto las últimas dos)
    size_t batch_size = 1;
    for (size_t i = 0; i < N - 2; ++i) {
        batch_size *= tensor.dims_[i];
    }
    
    size_t rows = tensor.dims_[N-2];
    size_t cols = tensor.dims_[N-1];
    size_t matrix_size = rows * cols;
    
    // Transpone cada matriz en el batch
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                size_t src_idx = b * matrix_size + i * cols + j;
                size_t dst_idx = b * matrix_size + j * rows + i;
                result.data_[dst_idx] = tensor.data_[src_idx];
            }
        }
    }
    
    return result;
}

// Multiplicación matricial de tensores
// Para 2D: multiplicación estándar de matrices (m,k) x (k,n) = (m,n)
// Para >2D: aplica multiplicación a cada batch
template<typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
    static_assert(N >= 2, "Matrix product requires at least 2D tensors");
    
    // Extrae dimensiones de las matrices: t1 es (m x k1), t2 es (k2 x n)
    size_t m = t1.dims_[N-2];   // filas de t1
    size_t k1 = t1.dims_[N-1];  // columnas de t1
    size_t k2 = t2.dims_[N-2];  // filas de t2
    size_t n = t2.dims_[N-1];   // columnas de t2
    
    // Verifica que las dimensiones internas coincidan: k1 debe ser igual a k2
    if (k1 != k2) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }
    
    // Para tensores >2D, verifica que las dimensiones de batch coincidan
    if (N > 2) {
        for (size_t i = 0; i < N - 2; ++i) {
            if (t1.dims_[i] != t2.dims_[i]) {
                throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }
        }
    }
    
    // El resultado tiene dimensiones (m x n)
    auto result_dims = t1.dims_;
    result_dims[N-2] = m;
    result_dims[N-1] = n;
    
    Tensor<T, N> result(result_dims);
    
    // Calcula cuántas matrices hay en el batch
    size_t batch_size = 1;
    for (size_t i = 0; i < N - 2; ++i) {
        batch_size *= t1.dims_[i];
    }
    
    size_t k = k1;
    
    // Multiplica cada par de matrices en el batch
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                // Producto punto de fila i de t1 con columna j de t2
                T sum = T{};
                for (size_t p = 0; p < k; ++p) {
                    size_t idx1 = b * (m * k) + i * k + p;      // t1[b][i][p]
                    size_t idx2 = b * (k * n) + p * n + j;      // t2[b][p][j]
                    sum += t1.data_[idx1] * t2.data_[idx2];
                }
                result.data_[b * (m * n) + i * n + j] = sum;
            }
        }
    }
    
    return result;
}

template<typename T, size_t N, typename Func>
Tensor<T, N> apply(const Tensor<T, N>& tensor, Func func) {
    return tensor.apply(func);
}

} // namespace algebra
} // namespace utec

#endif // TENSOR_H