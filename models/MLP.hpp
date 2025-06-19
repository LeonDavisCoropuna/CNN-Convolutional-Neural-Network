#include <iostream>
#include "layers/layer.hpp"
#include "../utils/loss.hpp"
#include "layers/dense_layer.hpp"
#include "layers/dropout_layer.hpp"
#include "layers/conv2d_layer.hpp"
#include "layers/pooling_layer.hpp"

#include "fstream"
#include "sstream"
class MLP
{
private:
  float learning_rate;
  std::vector<int> num_layers;
  int num_inputs;
  std::vector<Layer *> layers;
  std::vector<std::vector<float>> output_layers;
  std::vector<ActivationFunction *> activations;
  Loss *loss_function;
  int last_output_size = -1;
  Optimizer *optimizer;

public:
  MLP(float _learning_rate, Optimizer *_optimizer)
  {
    learning_rate = _learning_rate;
    optimizer = _optimizer;
  }

  void add_layer(Layer *layer)
  {
    layers.push_back(layer);
  }

  void set_loss(Loss *_loss_function)
  {
    loss_function = _loss_function;
  }
  int predict(const std::vector<float> &input)
  {
    std::vector<float> out = forward(input);
    if (out.size() == 1)
    { // Caso binario
      return out[0];
    }
    else
    { // Caso multiclase
      return static_cast<int>(std::distance(out.begin(),
                                            std::max_element(out.begin(), out.end())));
    }
  }

  std::vector<float> forward(std::vector<float> batch_inputs)
  {
    output_layers.clear();
    std::vector<float> current_input = batch_inputs;
    for (auto &layer : layers)
    {
      current_input = layer->forward(current_input);
      output_layers.push_back(current_input);
    }
    return current_input;
  }

  void train(int num_epochs, const std::vector<std::vector<float>> &X, const std::vector<float> &Y,
             const std::vector<std::vector<float>> &X_test, const std::vector<float> &Y_test,
             int batch_size = 1, const std::string &log_filepath = "")
  {

    bool is_binary = (layers.back()->output_size() == 1);
    std::ofstream log_file;

    if (!log_filepath.empty())
    {
      log_file.open(log_filepath, std::ios::out);
      if (!log_file.is_open())
      {
        std::cerr << "Error al abrir el archivo de logs: " << log_filepath << std::endl;
        return;
      }
    }
    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
      float total_loss = 0.0f;
      int correct_predictions = 0;
      for (auto layer : layers)
      {
        layer->set_training(true);
      }
      // Procesar en batches
      for (size_t batch_start = 0; batch_start < X.size(); batch_start += batch_size)
      {
        size_t batch_end = std::min(batch_start + batch_size, X.size());
        size_t actual_batch_size = batch_end - batch_start;

        // Limpiar gradientes acumulados
        for (auto &layer : layers)
        {
          layer->zero_grad();
        }

        float batch_loss = 0.0f;

        // Procesar cada muestra en el batch
        for (size_t i = batch_start; i < batch_end; i++)
        {

          std::vector<float> outputs = forward(X[i]);
          float y_true = Y[i];

          // Calcular precisión
          int predicted_class;
          if (is_binary)
          {
            predicted_class = (outputs[0] > 0.5f) ? 1 : 0;
            if (predicted_class == static_cast<int>(y_true))
              correct_predictions++;
          }
          else
          {
            predicted_class = static_cast<int>(std::distance(
                outputs.begin(), std::max_element(outputs.begin(), outputs.end())));
            if (predicted_class == static_cast<int>(y_true))
              correct_predictions++;
          }

          // Preparar target std::vector
          std::vector<float> target_vec;
          if (is_binary)
          {
            target_vec = {y_true};
          }
          else
          {
            target_vec.assign(layers.back()->output_size(), 0.0f);
            target_vec[static_cast<int>(y_true)] = 1.0f;
          }

          batch_loss += loss_function->compute(outputs, target_vec);

          // Backward pass y acumulación de gradientes
          layers.back()->backward(&target_vec);
          for (int l = layers.size() - 2; l >= 0; l--)
          {
            layers[l]->backward(nullptr, layers[l + 1]);
          }

          // Acumular gradientes para esta muestra
          for (auto &layer : layers)
          {
            layer->accumulate_gradients();
          }
        }

        // Aplicar gradientes promediados
        for (auto &layer : layers)
        {
          layer->apply_gradients(actual_batch_size);
        }

        total_loss += batch_loss;
        optimizer->increment_t();
      }

      // Calcular métricas
      float avg_loss = total_loss / X.size();
      float accuracy = static_cast<float>(correct_predictions) / X.size() * 100.0f;

      // === NUEVO: evaluación en el conjunto de test ===

      for (auto layer : layers)
      {
        layer->set_training(false);
      }

      float test_loss = 0.0f;
      int test_correct_predictions = 0;

      for (size_t i = 0; i < X_test.size(); ++i)
      {
        std::vector<float> outputs = forward(X_test[i]);
        float y_true = Y_test[i];

        int predicted_class;
        if (is_binary)
        {
          predicted_class = (outputs[0] > 0.5f) ? 1 : 0;
          if (predicted_class == static_cast<int>(y_true))
            test_correct_predictions++;
        }
        else
        {
          predicted_class = static_cast<int>(std::distance(
              outputs.begin(), std::max_element(outputs.begin(), outputs.end())));
          if (predicted_class == static_cast<int>(y_true))
            test_correct_predictions++;
        }

        std::vector<float> target_vec;
        if (is_binary)
        {
          target_vec = {y_true};
        }
        else
        {
          target_vec.assign(layers.back()->output_size(), 0.0f);
          target_vec[static_cast<int>(y_true)] = 1.0f;
        }

        test_loss += loss_function->compute(outputs, target_vec);
      }

      float avg_test_loss = test_loss / X_test.size();
      float test_accuracy = static_cast<float>(test_correct_predictions) / X_test.size() * 100.0f;

      // === Logging extendido ===
      std::ostringstream log_stream;
      log_stream << "Epoch " << epoch + 1
                 << ", Train Loss: " << avg_loss
                 << ", Train Accuracy: " << accuracy << "%"
                 << ", Test Loss: " << avg_test_loss
                 << ", Test Accuracy: " << test_accuracy << "%" << std::endl;

      std::cout << log_stream.str();
      if (log_file.is_open())
      {
        log_file << log_stream.str();
      }
    }

    if (log_file.is_open())
    {
      log_file.close();
    }
  }
  float evaluate(const std::vector<std::vector<float>> &X_test, const std::vector<float> &Y_test)
  {
    int correct_predictions = 0;
    for (auto layer : layers)
    {
      layer->set_training(false);
    }
    for (size_t i = 0; i < X_test.size(); i++)
    {
      std::vector<float> out = forward(X_test[i]);
      int predicted_class;
      float true_class = Y_test[i];

      if (out.size() == 1)
      { // Binario
        predicted_class = out[0] > 0.5f ? 1 : 0;

        if (predicted_class == static_cast<int>(true_class))
        {
          correct_predictions++;
        }
      }
      else
      { // Multiclase
        predicted_class = static_cast<int>(std::distance(out.begin(), std::max_element(out.begin(), out.end())));

        if (predicted_class == static_cast<int>(true_class))
        {
          correct_predictions++;
        }
      }
    }

    float accuracy = static_cast<float>(correct_predictions) / X_test.size() * 100.0f;
    std::cout << "Evaluation Results:" << std::endl;
    std::cout << " - Test samples: " << X_test.size() << std::endl;
    std::cout << " - Correct predictions: " << correct_predictions << std::endl;
    std::cout << " - Accuracy: " << accuracy << "%" << std::endl;

    return accuracy;
  }

  // ~MLP()
  // {
  //   for (auto *layer : layers)
  //   {
  //     delete layer;
  //   }
  //   for (auto *act : activations)
  //   {
  //     delete act;
  //   }
  //   delete loss_function;
  // }
  void save_weights(const std::string &filename)
  {
    std::ofstream file(filename);
    if (!file.is_open())
    {
      throw std::runtime_error("No se pudo abrir el archivo para guardar los pesos.");
    }

    for (size_t l = 0; l < layers.size(); ++l)
    {
      const auto &weights = layers[l]->get_weights();
      if (weights.empty())
        continue; // Capas sin pesos (ej. Dropout)

      file << "Layer " << l << ":\n";
      for (const auto &row : weights)
      {
        for (size_t i = 0; i < row.size(); ++i)
        {
          file << row[i];
          if (i < row.size() - 1)
            file << ",";
        }
        file << "\n";
      }
      file << "\n";
    }

    file.close();
  }

  void load_weights(const std::string &filename)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      throw std::runtime_error("No se pudo abrir el archivo para cargar los pesos.");
    }

    std::string line;
    size_t current_layer = 0;
    std::vector<std::vector<float>> layer_weights;

    while (std::getline(file, line))
    {
      if (line.empty())
      {
        // Fin de capa: aplicar pesos si la capa es válida
        while (current_layer < layers.size() && layers[current_layer]->get_weights().empty())
        {
          current_layer++; // Saltar capas sin pesos (ej. Dropout)
        }

        if (current_layer >= layers.size())
          break;

        layers[current_layer]->set_weights(layer_weights);
        layer_weights.clear();
        current_layer++;
      }
      else if (line.find("Layer") != std::string::npos)
      {
        // Encabezado, lo ignoramos (ya se usa `current_layer`)
        continue;
      }
      else
      {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ','))
        {
          row.push_back(std::stof(value));
        }
        layer_weights.push_back(row);
      }
    }

    // Última capa si no había línea vacía final
    if (!layer_weights.empty() && current_layer < layers.size())
    {
      while (current_layer < layers.size() && layers[current_layer]->get_weights().empty())
      {
        current_layer++;
      }

      if (current_layer < layers.size())
      {
        layers[current_layer]->set_weights(layer_weights);
      }
    }

    file.close();
  }
};
