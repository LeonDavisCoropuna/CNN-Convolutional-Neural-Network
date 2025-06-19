#include "layer.hpp"
#include "../../utils/activations.hpp"
#include "../../utils/optimizer.hpp"
#include <stdexcept>

class DenseLayer : public Layer
{
private:
  // Matriz de pesos: [num_neurons x num_inputs]
  std::vector<std::vector<float>> weights;

  // Vector de biases: [num_neurons]
  std::vector<float> biases;

  // Deltas para backpropagation: [num_neurons]
  std::vector<float> deltas;

  // Salidas de la capa: [num_neurons]
  std::vector<float> outputs;

  // Entradas de la capa: [num_inputs]
  std::vector<float> inputs;

  ActivationFunction *activation;
  Optimizer *optimizer;
  int num_neurons;
  int num_inputs;
  bool training;

  std::vector<std::vector<float>> accumulated_grad_weights;
  std::vector<float> accumulated_grad_biases;

public:
  DenseLayer(int _num_inputs, int _num_neurons,
             ActivationFunction *_activation, Optimizer *_optimizer)
      : num_neurons(_num_neurons), num_inputs(_num_inputs),
        activation(_activation),
        optimizer(_optimizer), training(true)
  {

    // Inicialización de pesos (He initialization)
    weights.resize(num_neurons, std::vector<float>(num_inputs));
    biases.resize(num_neurons, 0.01f);
    deltas.resize(num_neurons, 0.0f);
    // gradientes
    accumulated_grad_weights.resize(num_neurons, std::vector<float>(num_inputs, 0.0f));
    accumulated_grad_biases.resize(num_neurons, 0.0f);

    float stddev = sqrt(2.0f / num_inputs);
    for (int i = 0; i < num_neurons; ++i)
    {
      for (int j = 0; j < num_inputs; ++j)
      {
        weights[i][j] = normal_distribution<float>(0.0f, stddev)(Layer::gen);
      }
    }
  }

  // Forward pass matricial
  std::vector<float> forward(const std::vector<float> &batch_inputs)
  {
    inputs = batch_inputs;
    std::vector<float> pre_activations(num_neurons, 0.0f);
    outputs.resize(num_neurons);

// Multiplicación matriz-vector: W·X + b
#pragma omp parallel for
    for (int i = 0; i < num_neurons; ++i)
    {
      float z = biases[i];
      for (int j = 0; j < num_inputs; ++j)
      {
        z += weights[i][j] * inputs[j];
      }
      pre_activations[i] = z;
    }

    // Aplicación de función de activación
    if (dynamic_cast<Softmax *>(activation))
    {
      outputs = activation->activate_vector(pre_activations);
    }
    else
    {
#pragma omp parallel for
      for (int i = 0; i < num_neurons; ++i)
      {
        outputs[i] = activation->activate(pre_activations[i]);
      }
    }

    return outputs;
  }

  // Backward pass optimizado
  void backward(const std::vector<float> *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    if (targets != nullptr)
    {
      // Cálculo para capa de salida
      for (int i = 0; i < num_neurons; ++i)
      {
        float error = outputs[i] - (*targets)[i];
        if (dynamic_cast<Softmax *>(activation))
        {
          deltas[i] = error; // Softmax + Cross-Entropy
        }
        else
        {
          deltas[i] = error * activation->derivative(outputs[i]);
        }
      }
    }
    else if (next_layer != nullptr)
    {
      // Capa oculta
      for (int i = 0; i < num_neurons; ++i)
      {
        float sum = 0.0f;

        if (next_layer->has_weights())
        {
          // Para capas con pesos (Dense)
          const auto &next_weights = next_layer->get_weights();
          const auto &next_deltas = next_layer->get_deltas();
          int next_neurons = next_layer->output_size();

          for (int j = 0; j < next_neurons; ++j)
          {
            sum += next_weights[j][i] * next_deltas[j];
          }
        }
        else
        {
          // Para capas sin pesos (Dropout, etc.)
          const auto &next_deltas = next_layer->get_deltas();
          sum = next_deltas[i]; // Asume misma dimensionalidad
        }

        deltas[i] = sum * activation->derivative(outputs[i]);
      }
    }
    else
    {
      throw std::runtime_error("Invalid backward call");
    }
  }

  // Actualización de pesos con el optimizador
  void update_weights()
  {
    std::vector<std::vector<float>> gradients_weights(num_neurons, std::vector<float>(num_inputs));
    std::vector<float> gradients_biases(num_neurons);

// Calcular gradientes
#pragma omp parallel for
    for (int i = 0; i < num_neurons; ++i)
    {
      for (int j = 0; j < num_inputs; ++j)
      {
        gradients_weights[i][j] = deltas[i] * inputs[j];
      }
      gradients_biases[i] = deltas[i];
    }

    // Actualizar pesos con el optimizador
    optimizer->update(weights, gradients_weights, biases, gradients_biases);
  }

  void accumulate_gradients()
  {
#pragma omp parallel for
    for (int i = 0; i < num_neurons; ++i)
    {
      for (int j = 0; j < num_inputs; ++j)
      {
        accumulated_grad_weights[i][j] += deltas[i] * inputs[j];
      }
      accumulated_grad_biases[i] += deltas[i];
    }
  }

  void apply_gradients(float batch_size)
  {
// Dividir por el tamaño del batch para obtener el promedio
#pragma omp parallel for
    for (int i = 0; i < num_neurons; ++i)
    {
      for (int j = 0; j < num_inputs; ++j)
      {
        accumulated_grad_weights[i][j] /= batch_size;
      }
      accumulated_grad_biases[i] /= batch_size;
    }

    // Pasar los gradientes al optimizador
    optimizer->update(weights, accumulated_grad_weights,
                      biases, accumulated_grad_biases);

    // Reiniciar acumuladores
    zero_grad();
  }

  void zero_grad()
  {
    std::fill(deltas.begin(), deltas.end(), 0.0f);
    for (auto &row : accumulated_grad_weights)
    {
      std::fill(row.begin(), row.end(), 0.0f);
    }
    std::fill(accumulated_grad_biases.begin(), accumulated_grad_biases.end(), 0.0f);
  }

  // Métodos de acceso
  const std::vector<float> &get_outputs() const { return outputs; }
  const std::vector<float> &get_deltas() const { return deltas; }
  void set_weights(const std::vector<std::vector<float>> &new_weights) override
  {
    weights = new_weights;
  }

  const std::vector<std::vector<float>> &get_weights() const { return weights; }
  int input_size() const
  {
    return num_inputs;
  }

  int output_size() const
  {
    return num_neurons;
  }
  bool has_weights() const override { return true; }
  // Activar o desactivar modo entrenamiento
  void set_training(bool is_training) override
  {
    training = is_training;
  }
};