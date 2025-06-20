#include "models/MLP.hpp"
#include "utils/load_dataset.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

mt19937 Layer::gen(32);

// 🔍 Visualiza la predicción del modelo para una imagen específica
void visualizePrediction(CNN& model,
                         const std::vector<float>& image,
                         const std::vector<float>& oneHotLabel) {
    std::cout << "\n=== Imagen de entrada ===\n";
    MNISTDataset::displayImage(image);

    // Obtener etiqueta real
    int trueLabel = static_cast<int>(
        std::distance(oneHotLabel.begin(),
                      std::max_element(oneHotLabel.begin(), oneHotLabel.end())));

    std::cout << "Etiqueta real: " << trueLabel << "\n";

    // Forward pass
    auto output = model.forward(image);

    std::cout << "Tamaño de la salida: " << output.size() << "\n";
    std::cout << "Vector de salida (softmax): ";
    for (float prob : output) {
        std::cout << std::fixed << std::setprecision(4) << prob << " ";
    }
    std::cout << "\n";

    int predictedLabel = static_cast<int>(
        std::distance(output.begin(),
                      std::max_element(output.begin(), output.end())));
    std::cout << "Predicción del modelo: " << predictedLabel << "\n";
}

int main() {
    // Cargar dataset
    auto trainImages = MNISTDataset::loadImages("mnist_data/train-images.idx3-ubyte", 2000);
    auto trainLabels = MNISTDataset::loadLabels("mnist_data/train-labels.idx1-ubyte", 2000);
    auto testImages  = MNISTDataset::loadImages("mnist_data/t10k-images.idx3-ubyte", 2000);
    auto testLabels  = MNISTDataset::loadLabels("mnist_data/t10k-labels.idx1-ubyte", 2000);

    std::cout << "Cargadas " << trainImages.size() << " imágenes de entrenamiento.\n";
    std::cout << "Cargadas " << testImages.size() << " imágenes de prueba.\n";

    // Inicializar optimizador y red
    float learningRate = 0.001f;
    float weightDecay = 0.0005f;
    Optimizer* adam = new Adam(learningRate, weightDecay);
    CNN cnn(learningRate, adam);

    // Construir arquitectura
    cnn.addLayer(new Conv2DLayer(1, 8, 3, 28, 28));    // -> [8, 26, 26]
    cnn.addLayer(new PoolingLayer(8, 26, 26));         // -> [8, 13, 13]
    cnn.addLayer(new Conv2DLayer(8, 16, 3, 13, 13));   // -> [16, 11, 11]
    cnn.addLayer(new PoolingLayer(16, 11, 11));        // -> [16, 5, 5]
    cnn.addLayer(new Conv2DLayer(16, 32, 3, 5, 5));    // -> [32, 3, 3]
    cnn.addLayer(new PoolingLayer(32, 3, 3));          // -> [32, 1, 1]
    cnn.addLayer(new DenseLayer(32, 16, new ReLU(), adam));
    cnn.addLayer(new DropoutLayer(0.2f));
    cnn.addLayer(new DenseLayer(16, 10, new Softmax(), adam));
    cnn.setLoss(new CrossEntropyLoss());

    // Entrenar modelo
    cnn.train(20, trainImages, trainLabels, testImages, testLabels, 32, "output/test-conv2d.txt");

    // Visualizar una predicción
    visualizePrediction(cnn, trainImages[0], trainLabels[0]);

    return 0;
}
