#include "models/MLP.hpp"
#include "utils/load_dataset.hpp"
#include <chrono>

std::mt19937 Layer::gen(32); // Semilla para reproducibilidad

int main()
{
  auto train_data = load_dataset_numbers("mnist_data/saved_images/train");
  auto test_data = load_dataset_numbers("mnist_data/saved_images/test");

  std::cout << "Cargadas " << train_data.first.size() << " imágenes de entrenamiento." << std::endl;
  std::cout << "Cargadas " << test_data.first.size() << " imágenes de prueba." << std::endl;

  float learning_rate = 0.001f;
  float wd = 0.0005f;

  Optimizer *adam = new Adam(learning_rate, wd);
  MLP mlp(learning_rate, adam);

  mlp.add_layer(new DenseLayer(784, 64, new ReLU(), adam));
  mlp.add_layer(new DropoutLayer(0.2));
  mlp.add_layer(new DenseLayer(64, 32, new ReLU(), adam));
  mlp.add_layer(new DropoutLayer(0.2));
  mlp.add_layer(new DenseLayer(32, 10, new Softmax(), adam));
  mlp.set_loss(new CrossEntropyLoss());

  auto start_time = std::chrono::high_resolution_clock::now();

  mlp.train(20, train_data.first, train_data.second, test_data.first, test_data.second, 32, "output/drop05-wd-0005-arch784x64x32x10.txt");
  mlp.save_weights("save_models/drop02-wd-0005-arch784x64x32x10.txt");
  // mlp.load_weights("save_models/drop02-arch784x64x32x10.txt");
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;

  std::cout << "Tiempo total de entrenamiento: " << duration.count() << " segundos" << std::endl;

  mlp.evaluate(test_data.first, test_data.second);

  return 0;
}