#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random> // Para std::uniform_real_distribution y mt19937
using namespace std;

class ActivationFunction
{
public:
  virtual float activate(float x) const = 0;
  virtual float derivative(float x) const = 0;
  virtual void initialize_weights(vector<float> &weights, int num_inputs, mt19937 gen) const = 0;
  virtual vector<float> activate_vector(const vector<float> &x) const
  {
    throw runtime_error("activate_vector no implementado para esta función de activación");
  }
  virtual bool requires_special_output_gradient() const { return false; } // Por defecto, no requiere tratamiento especial
};

// ReLU
class ReLU : public ActivationFunction
{
public:
  float activate(float x) const override
  {
    return (x > 0) ? x : 0;
  }

  float derivative(float x) const override
  {
    return (x > 0) ? 1.0f : 0.0f;
  }

  void initialize_weights(vector<float> &weights, int num_inputs, mt19937 gen) const override
  {
    // Inicialización He (He et al., 2015) para ReLU
    float stddev = sqrt(2.0 / num_inputs);
    normal_distribution<float> dist(0.0, stddev);
    for (auto &w : weights)
    {
      w = dist(gen);
    }
  }
};

// Tanh
class Tanh : public ActivationFunction
{
public:
  float activate(float x) const override
  {
    return tanh(x);
  }

  float derivative(float x) const override
  {
    return 1.0f - tanh(x) * tanh(x); // Derivada de tanh(x)
  }

  void initialize_weights(vector<float> &weights, int num_inputs, mt19937 gen) const override
  {
    // Inicialización Xavier/Glorot (Glorot & Bengio, 2010) para Tanh/Sigmoid
    float limit = sqrt(6.0 / num_inputs);
    uniform_real_distribution<float> dist(-limit, limit);
    for (auto &w : weights)
    {
      w = dist(gen);
    }
  }
};

// Sigmoid
class Sigmoid : public ActivationFunction
{
public:
  float activate(float x) const override
  {
    return 1.0f / (1.0f + exp(-x));
  }

  float derivative(float x) const override
  {
    float sig = activate(x);
    return sig * (1 - sig);
  }

  void initialize_weights(vector<float> &weights, int num_inputs, mt19937 gen) const override
  {
    // Misma inicialización Xavier/Glorot que Tanh
    float limit = sqrt(6.0 / num_inputs);
    uniform_real_distribution<float> dist(-limit, limit);
    for (auto &w : weights)
    {
      w = dist(gen);
    }
  }
};

// Softmax
class Softmax : public ActivationFunction
{
public:
  // Esta función no tiene sentido para softmax, pero debe estar por contrato
  float activate(float x) const override
  {
    return x; // No aplica softmax escalar
  }

  float derivative(float x) const override
  {
    return 1.0f; // No usada directamente (softmax se trata diferente con cross entropy)
  }

  void initialize_weights(vector<float> &weights, int num_inputs, mt19937 gen) const override
  {
    // Softmax suele usar la misma inicialización que sigmoid
    float limit = sqrt(6.0 / num_inputs);
    uniform_real_distribution<float> dist(-limit, limit);
    for (auto &w : weights)
    {
      w = dist(gen);
    }
  }
  bool requires_special_output_gradient() const override { return true; } // ¡Softmax necesita tratamiento especial!

  // Softmax se aplica a un vector de pre-activaciones
  vector<float> activate_vector(const vector<float> &z) const
  {
    vector<float> result(z.size());

    // Estabilización numérica: restar el máximo antes de exponenciar
    float max_val = *max_element(z.begin(), z.end());

    float sum_exp = 0.0f;
    for (size_t i = 0; i < z.size(); ++i)
    {
      result[i] = exp(z[i] - max_val);
      sum_exp += result[i];
    }

    for (size_t i = 0; i < z.size(); ++i)
    {
      result[i] /= sum_exp;
    }

    return result;
  }
};
