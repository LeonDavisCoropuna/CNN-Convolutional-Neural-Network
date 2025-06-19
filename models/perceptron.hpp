#include <iostream>
#include <vector>
#include <random>
#include "../utils/activations.hpp"
#include <sstream>
#include <fstream>
using namespace std;

class Perceptron
{
public:
  float bias;
  vector<float> weights;
  float learning_rate;
  static mt19937 gen;
  float output;

  // gradiente local
  float delta;
  vector<float> grad_w; // ⬅️ Aquí almacenas los gradientes acumulados

public:
  float forward(const vector<float> &inputs)
  {
    float z = bias;
    for (size_t i = 0; i < weights.size(); i++)
    {
      z += weights[i] * inputs[i];
    }
    return z;
  }

  Perceptron(int num_inputs, float _learning_rate)
  {
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    learning_rate = _learning_rate;

    weights.resize(num_inputs);
    float stddev = sqrt(2.0f / num_inputs);
    for (auto &w : weights)
    {
      w = normal_distribution<float>(0.0f, stddev)(gen);
    }
    bias = 0.01f;

    delta = 0.0f;
  }

  void print_weights()
  {
    cout << "Pesos: ";
    for (const auto &w : weights)
    {
      cout << w << "\t";
    }
    cout << endl
         << "Bias: " << bias << endl;
  }

  void set_delta(float d)
  {
    delta = d;
  }
  float get_delta() const
  {
    return delta;
  }
  float get_output() const
  {
    return output;
  }
  vector<float> getWeights()
  {
    return weights;
  }

  void serialize(std::ofstream &file) const
  {
    size_t num_weights = weights.size();
    file.write(reinterpret_cast<const char *>(&num_weights), sizeof(num_weights));

    file.write(reinterpret_cast<const char *>(weights.data()),
               num_weights * sizeof(float));

    file.write(reinterpret_cast<const char *>(&bias), sizeof(bias));
  }

  void deserialize(std::ifstream &file)
  {
    size_t num_weights;
    file.read(reinterpret_cast<char *>(&num_weights), sizeof(num_weights));

    weights.resize(num_weights);
    file.read(reinterpret_cast<char *>(weights.data()),
              num_weights * sizeof(float));

    file.read(reinterpret_cast<char *>(&bias), sizeof(bias));
  }
};
