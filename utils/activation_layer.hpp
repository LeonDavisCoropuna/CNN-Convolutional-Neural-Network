#pragma once
#include "../models/layers/layer.hpp"
#include "activations.hpp"

class ActivationLayer : public Layer {
private:
    ActivationFunction* activation;
    std::vector<float> inputs;
    std::vector<float> outputs;
    std::vector<float> deltas;
    bool is_training;

public:
    ActivationLayer(ActivationFunction* act) : activation(act), is_training(true) {}

    std::vector<float> forward(const std::vector<float>& input) override {
        inputs = input;
        outputs.resize(input.size());
        deltas.resize(input.size());

        // Aplicar activación elemento por elemento
#pragma omp parallel for
        for (int i = 0; i < input.size(); ++i) {
            outputs[i] = activation->activate(input[i]);
        }

        return outputs;
    }

    void backward(const std::vector<float>* targets = nullptr,
                  const Layer* next_layer = nullptr) override {
        const std::vector<float>& next_deltas = next_layer->get_deltas();
        const std::vector<std::vector<float>>& next_weights = next_layer->get_weights();

        deltas.resize(inputs.size());

        // Propagación del gradiente
#pragma omp parallel for
        for (int i = 0; i < inputs.size(); ++i) {
            float grad = 0.0f;
            if (!next_weights.empty()) {
                for (int j = 0; j < next_deltas.size(); ++j) {
                    grad += next_weights[j][i] * next_deltas[j];
                }
            } else {
                grad = next_deltas[i];
            }

            deltas[i] = grad * activation->derivative(inputs[i]);
        }
    }

    void accumulate_gradients() override {}
    void apply_gradients(float batch_size) override {}
    void zero_grad() override {}

    const std::vector<float>& get_outputs() const override { return outputs; }
    const std::vector<float>& get_deltas() const override { return deltas; }

    const std::vector<std::vector<float>>& get_weights() const override {
        static std::vector<std::vector<float>> dummy;
        return dummy;
    }

    void set_weights(const std::vector<std::vector<float>>& new_weights) override {}
    int input_size() const override { return outputs.size(); }
    int output_size() const override { return outputs.size(); }
    bool has_weights() const override { return false; }

    void set_training(bool training) override {
        is_training = training;
    }
    void update_weights() override {} // No hace nada

};
