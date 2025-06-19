#include "models/MLP.hpp"
#include "utils/load_dataset.hpp"
#include <chrono>

mt19937 Layer::gen(32);

int main()
{

  auto train_data = load_dataset_numbers("mnist_data/saved_images/train", 2000);
  auto test_data = load_dataset_numbers("mnist_data/saved_images/test", 2000);

  std::cout << "Cargadas " << train_data.first.size() << " imágenes de entrenamiento." << std::endl;
  std::cout << "Cargadas " << test_data.first.size() << " imágenes de prueba." << std::endl;

  float learning_rate = 0.001f;
  Optimizer *sgd = new SGD(learning_rate);
  MLP mlp(learning_rate, sgd);
  float wd = 0.0005f;
  Optimizer *adam = new Adam(learning_rate, wd);

  mlp.add_layer(new Conv2DLayer(1, 8, 3, 28, 28)); // output: [8, 26, 26]
  mlp.add_layer(new PoolingLayer(8, 26, 26));      // output: [8, 13, 13]

  mlp.add_layer(new Conv2DLayer(8, 16, 3, 13, 13)); // output: [16, 11, 11]
  mlp.add_layer(new PoolingLayer(16, 11, 11));      // output: [16, 5, 5]

  mlp.add_layer(new Conv2DLayer(16, 32, 3, 5, 5)); // output: [32, 3, 3]
  mlp.add_layer(new PoolingLayer(32, 3, 3));       // output: [32, 1, 1]

  mlp.add_layer(new DenseLayer(32, 16, new ReLU(), adam));
  mlp.add_layer(new DropoutLayer(0.2));
  mlp.add_layer(new DenseLayer(16, 10, new Softmax(), adam));
  mlp.set_loss(new CrossEntropyLoss());

  // mlp.train(20, train_data.first, train_data.second, test_data.first, test_data.second, 32, "output/test-conv2d.txt");

  auto ff = mlp.forward(train_data.first[0]);
  std::cout << "suzeee: " << ff.size() << std::endl;

  for (auto qq : ff)
  {
    std::cout << qq;
  }

  std::cout << std::endl;
  return 0;
}