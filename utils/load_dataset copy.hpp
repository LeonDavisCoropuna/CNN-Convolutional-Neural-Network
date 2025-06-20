#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <regex>
#include <string>
#include <iostream>

namespace fs = std::filesystem;
std::pair<std::vector<std::vector<float>>, std::vector<float>>
load_dataset_numbers(const std::string &folder_path, int max_samples = -1)
{
  std::vector<std::vector<float>> X;
  std::vector<float> Y;
  std::regex label_regex(".*?(\\d+)\\.png$");

  // 1. Recolectar archivos en un vector
  std::vector<fs::directory_entry> entries;
  for (const auto &entry : fs::directory_iterator(folder_path))
  {
    if (entry.is_regular_file())
    {
      entries.push_back(entry);
    }
  }

  // 2. Ordenar alfabéticamente por nombre de archivo
  std::sort(entries.begin(), entries.end(), [](const fs::directory_entry &a, const fs::directory_entry &b)
            { return a.path().filename() < b.path().filename(); });

  // 3. Procesar los archivos ordenados
  int count = 0;
  for (const auto &entry : entries)
  {
    if (max_samples != -1 && count >= max_samples)
      break;

    std::string filename = entry.path().filename().string();
    std::smatch match;

    if (std::regex_match(filename, match, label_regex))
    {
      int label = std::stoi(match[1]);
      cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

      if (!img.empty())
      {
        std::vector<float> flattened;
        flattened.reserve(img.total());

        for (int i = 0; i < img.rows; ++i)
        {
          for (int j = 0; j < img.cols; ++j)
          {
            flattened.push_back(static_cast<float>(img.at<uchar>(i, j)) / 255.0f);
          }
        }

        X.push_back(flattened);
        Y.push_back(static_cast<float>(label));
        ++count;
      }
      else
      {
        std::cerr << "No se pudo cargar la imagen: " << entry.path() << std::endl;
      }
    }
    else
    {
      std::cerr << "Nombre no coincide con regex: " << filename << std::endl;
    }
  }

  return {X, Y};
}

void flatten_image_to_vector_and_predict(const std::string &image_path, MLP &mlp)
{
  // 1) Leer imagen en escala de grises
  cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (img.empty())
  {
    std::cerr << "No se pudo cargar la imagen: " << image_path << std::endl;
    return;
  }

  // 2) Redimensionar a 28x28 con interpolación Lanczos4 (similar a PIL.LANCZOS)
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(28, 28), 0, 0, cv::INTER_LANCZOS4);

  // 3) Convertir a blanco y negro con umbral (threshold binario)
  cv::Mat bw_img;
  int umbral = 128; // umbral como en Python
  cv::threshold(resized_img, bw_img, umbral, 255, cv::THRESH_BINARY);

  // 4) Imprimir matriz de píxeles
  std::cout << "Matriz de píxeles (28x28) en blanco y negro:\n";
  for (int i = 0; i < bw_img.rows; ++i)
  {
    for (int j = 0; j < bw_img.cols; ++j)
    {
      std::cout << static_cast<int>(bw_img.at<uchar>(i, j)) << ' ';
    }
    std::cout << '\n';
  }

  // 5) Aplanar y normalizar a [0,1]
  std::vector<float> flattened;
  flattened.reserve(bw_img.total());
  for (int i = 0; i < bw_img.rows; ++i)
  {
    for (int j = 0; j < bw_img.cols; ++j)
    {
      float norm = bw_img.at<uchar>(i, j) / 255.0f; // 0 o 1
      flattened.push_back(norm);
    }
  }

  // 6) Predecir con el MLP y mostrar resultado
  float pred = mlp.predict(flattened);
  std::cout << "Predicción del MLP: " << pred << std::endl;
}


std::pair<std::vector<std::vector<float>>, std::vector<float>>
load_dataset_fashion(const std::string &csv_path) {
    std::ifstream file(csv_path);
    std::string line;
    std::vector<std::vector<float>> images;
    std::vector<float> labels;

    // Leer encabezado y descartarlo
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> pixels;

        // Leer la etiqueta
        std::getline(ss, value, ',');
        labels.push_back(std::stof(value));

        // Leer todos los píxeles
        while (std::getline(ss, value, ',')) {
            pixels.push_back(std::stof(value) / 255.0f);  // Normaliza los valores a [0, 1]
        }

        images.push_back(pixels);
    }

    return {images, labels};
}