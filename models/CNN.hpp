#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "layers/layer.hpp"
#include "layers/dense_layer.hpp"
#include "layers/dropout_layer.hpp"
#include "layers/conv2d_layer.hpp"
#include "layers/pooling_layer.hpp"
#include "../utils/loss.hpp"

class CNN {
private:
  float learningRate;
  std::vector<Layer*> layers;
  std::vector<std::vector<float>> outputLayers;
  Loss* lossFunction;
  Optimizer* optimizer;

public:
  CNN(float lr, Optimizer* opt) : learningRate(lr), optimizer(opt) {}

  void addLayer(Layer* layer) {
    layers.push_back(layer);
  }

  void setLoss(Loss* loss) {
    lossFunction = loss;
  }

  int predict(const std::vector<float>& input) {
    std::vector<float> output = forward(input);
    if (output.size() == 1) {
      return output[0];
    } else {
      return static_cast<int>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));
    }
  }

  std::vector<float> forward(const std::vector<float>& input) {
    outputLayers.clear();
    std::vector<float> current = input;
    for (auto& layer : layers) {
      current = layer->forward(current);
      outputLayers.push_back(current);
    }
    return current;
  }

  void train(int epochs,
             const std::vector<std::vector<float>>& trainData,
             const std::vector<std::vector<float>>& trainLabels,
             const std::vector<std::vector<float>>& testData,
             const std::vector<std::vector<float>>& testLabels,
             int batchSize = 1,
             const std::string& logPath = "") {

    bool isBinary = (layers.back()->output_size() == 1);
    std::ofstream logFile;

    if (!logPath.empty()) {
      logFile.open(logPath, std::ios::out);
      if (!logFile.is_open()) {
        std::cerr << "Error al abrir el archivo de logs: " << logPath << std::endl;
        return;
      }
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
      float totalLoss = 0.0f;
      int correct = 0;

      for (auto& layer : layers) {
        layer->set_training(true);
      }

      for (size_t i = 0; i < trainData.size(); i += batchSize) {
        size_t end = std::min(i + batchSize, trainData.size());
        size_t currentBatchSize = end - i;

        for (auto& layer : layers) {
          layer->zero_grad();
        }

        float batchLoss = 0.0f;

        for (size_t j = i; j < end; j++) {
          const std::vector<float>& label = trainLabels[j];
          std::vector<float> output = forward(trainData[j]);

          int pred = isBinary ? (output[0] > 0.5f ? 1 : 0)
                              : static_cast<int>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));

          int trueClass = static_cast<int>(std::distance(label.begin(), std::max_element(label.begin(), label.end())));
          if (pred == trueClass) correct++;

          batchLoss += lossFunction->compute(output, label);

          layers.back()->backward(&label);
          for (int l = layers.size() - 2; l >= 0; l--) {
            layers[l]->backward(nullptr, layers[l + 1]);
          }
          for (auto& layer : layers) {
            layer->accumulate_gradients();
          }
        }

        for (auto& layer : layers) {
          layer->apply_gradients(currentBatchSize);
        }

        totalLoss += batchLoss;
        optimizer->increment_t();
      }

      float avgTrainLoss = totalLoss / trainData.size();
      float trainAccuracy = static_cast<float>(correct) / trainData.size() * 100.0f;

      for (auto& layer : layers) {
        layer->set_training(false);
      }

      float testLoss = 0.0f;
      int testCorrect = 0;

      for (size_t i = 0; i < testData.size(); ++i) {
        const std::vector<float>& label = testLabels[i];
        std::vector<float> output = forward(testData[i]);

        int pred = isBinary ? (output[0] > 0.5f ? 1 : 0)
                            : static_cast<int>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));

        int trueClass = static_cast<int>(std::distance(label.begin(), std::max_element(label.begin(), label.end())));
        if (pred == trueClass) testCorrect++;

        testLoss += lossFunction->compute(output, label);
      }

      float avgTestLoss = testLoss / testData.size();
      float testAccuracy = static_cast<float>(testCorrect) / testData.size() * 100.0f;

      std::ostringstream log;
      log << "[" << epoch + 1 << "/" << epochs << "] Epoch "
          << "Train Loss: " << std::fixed << std::setprecision(4) << avgTrainLoss
          << ", Train Acc: " << std::setprecision(2) << trainAccuracy << "%"
          << " | Test Loss: " << std::setprecision(4) << avgTestLoss
          << ", Test Acc: " << std::setprecision(2) << testAccuracy << "%" << std::endl;

      std::cout << log.str();
      if (logFile.is_open()) logFile << log.str();
    }

    if (logFile.is_open()) logFile.close();
  }

  float evaluate(const std::vector<std::vector<float>>& testData,
                 const std::vector<float>& testLabels) {
    int correct = 0;
    for (auto& layer : layers) {
      layer->set_training(false);
    }

    for (size_t i = 0; i < testData.size(); i++) {
      std::vector<float> output = forward(testData[i]);
      int pred = (output.size() == 1) ? (output[0] > 0.5f ? 1 : 0)
                                      : static_cast<int>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));
      if (pred == static_cast<int>(testLabels[i])) correct++;
    }

    float accuracy = static_cast<float>(correct) / testData.size() * 100.0f;
    std::cout << "Evaluation Results:\n - Test samples: " << testData.size()
              << "\n - Correct predictions: " << correct
              << "\n - Accuracy: " << accuracy << "%\n";
    return accuracy;
  }

  void saveWeights(const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("No se pudo abrir el archivo para guardar los pesos.");
    }
    for (size_t i = 0; i < layers.size(); ++i) {
      const auto& weights = layers[i]->get_weights();
      if (weights.empty()) continue;
      file << "Layer " << i << ":\n";
      for (const auto& row : weights) {
        for (size_t j = 0; j < row.size(); ++j) {
          file << row[j];
          if (j < row.size() - 1) file << ",";
        }
        file << "\n";
      }
      file << "\n";
    }
    file.close();
  }

  void loadWeights(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("No se pudo abrir el archivo para cargar los pesos.");
    }

    std::string line;
    size_t currentLayer = 0;
    std::vector<std::vector<float>> layerWeights;

    while (std::getline(file, line)) {
      if (line.empty()) {
        while (currentLayer < layers.size() && layers[currentLayer]->get_weights().empty()) {
          currentLayer++;
        }
        if (currentLayer >= layers.size()) break;
        layers[currentLayer]->set_weights(layerWeights);
        layerWeights.clear();
        currentLayer++;
      } else if (line.find("Layer") != std::string::npos) {
        continue;
      } else {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
          row.push_back(std::stof(val));
        }
        layerWeights.push_back(row);
      }
    }

    if (!layerWeights.empty() && currentLayer < layers.size()) {
      while (currentLayer < layers.size() && layers[currentLayer]->get_weights().empty()) {
        currentLayer++;
      }
      if (currentLayer < layers.size()) {
        layers[currentLayer]->set_weights(layerWeights);
      }
    }
    file.close();
  }
};
