#pragma once
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace std;

// Clase para manejar datos MNIST con float
class MNISTDataset {
public:
    static vector<vector<float>> loadImages(const string& filename, int max_images = -1);
    static vector<vector<float>> loadLabels(const string& filename, int max_labels = -1);

    static void displayImage(const vector<float>& image, int rows = 28, int cols = 28);
};

// Carga las imágenes MNIST
vector<vector<float>> MNISTDataset::loadImages(const string& filename, int max_images) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("No se pudo abrir el archivo de imágenes");

    int32_t magic = 0, num = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = __builtin_bswap32(magic);
    num = __builtin_bswap32(num);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (max_images > 0 && max_images < num) {
        num = max_images;
    }

    vector<vector<float>> images(num, vector<float>(rows * cols));

    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }

    return images;
}

// Carga las etiquetas MNIST (como float, no one-hot)
vector<vector<float>> MNISTDataset::loadLabels(const string& filename, int max_labels) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("No se pudo abrir el archivo de etiquetas");

    int32_t magic = 0, num_labels = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    if (max_labels > 0 && max_labels < num_labels) {
        num_labels = max_labels;
    }

    vector<vector<float>> labels(num_labels, vector<float>(10, 0.0f));
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i][label] = 1.0f;
    }

    return labels;
}


// Muestra una imagen en texto (modo consola)
void MNISTDataset::displayImage(const vector<float>& image, int rows, int cols) {
    const string shades = " .:-=+*#%@";

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float pixel = image[i * cols + j];
            int level = static_cast<int>(pixel * (shades.size() - 1));
            cout << shades[level] << shades[level];
        }
        cout << endl;
    }
}
