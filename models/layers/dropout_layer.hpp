#include "layer.hpp"
#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
class DropoutLayer : public Layer
{
private:
  float dropout_rate;
  std::vector<float> mask;
  std::vector<float> outputs;
  std::vector<float> inputs;
  bool training;

public:
  DropoutLayer(float rate) : dropout_rate(rate), training(true)
  {
    if (rate < 0.0f || rate >= 1.0f)
    {
      throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
  }

  std::vector<float> forward(const std::vector<float> &inputs) override
  {
    this->inputs = inputs;
    outputs.resize(inputs.size());

    if (training)
    {
      mask.resize(inputs.size());
      float keep_prob = 1.0f - dropout_rate;

      for (size_t i = 0; i < inputs.size(); ++i)
      {
        float rand_val = std::uniform_real_distribution<float>(0.0f, 1.0f)(gen);
        mask[i] = rand_val < keep_prob ? 1.0f / keep_prob : 0.0f;
        outputs[i] = inputs[i] * mask[i];
      }
    }
    else
    {
      outputs = inputs;
    }

    return outputs;
  }

  void backward(const std::vector<float> *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    if (!next_layer)
      throw std::runtime_error("Dropout needs next_layer");

    const auto &next_deltas = next_layer->get_deltas();
    deltas.resize(inputs.size()); // Usar tamaño de entrada, no de siguiente capa

    if (training)
    {
      if (next_layer->has_weights())
      {
        // Capa densa siguiente - propagación normal
        const auto &weights = next_layer->get_weights();
        for (size_t i = 0; i < deltas.size(); ++i)
        {
          float sum = 0.0f;
          for (size_t j = 0; j < next_deltas.size(); ++j)
          {
            sum += weights[j][i] * next_deltas[j];
          }
          deltas[i] = sum * mask[i];
        }
      }
      else
      {
        // Para otras capas sin pesos (como otro Dropout)
        // Solo aplicamos máscara si las dimensiones coinciden
        if (next_deltas.size() == mask.size())
        {
          for (size_t i = 0; i < deltas.size(); ++i)
          {
            deltas[i] = next_deltas[i] * mask[i];
          }
        }
        else
        {
          throw std::runtime_error("Dimension mismatch in Dropout backward");
        }
      }
    }
    else
    {
      // En modo evaluación
      if (next_deltas.size() == inputs.size())
      {
        deltas = next_deltas;
      }
      else
      {
        throw std::runtime_error("Dimension mismatch in Dropout backward");
      }
    }
  }

  // Métodos de acceso
  const std::vector<float> &get_outputs() const override { return outputs; }
  const std::vector<float> &get_deltas() const override { return deltas; }
  const std::vector<std::vector<float>> &get_weights() const override
  {
    static const std::vector<std::vector<float>> empty;
    return empty; // Dropout no tiene pesos
  }
  void set_training(bool is_training) override
  {
    training = is_training;
  }
  void set_weights(std::vector<std::vector<float>> v) {}

  int input_size() const override { return inputs.empty() ? 0 : inputs.size(); }
  int output_size() const override { return outputs.size(); }

  // Métodos vacíos requeridos por la interfaz
  void update_weights() override {}
  void accumulate_gradients() override {}
  void apply_gradients(float) override {}
  void zero_grad() override {}

private:
  std::vector<float> deltas;
};
