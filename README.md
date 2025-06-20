
# CNN para MNIST con C++ (Convolucional + Pooling + Dense + Dropout + Weight Decay)


Este proyecto implementa una red convolucional (CNN) entrenada para reconocer dígitos (0–9) usando el dataset MNIST. Está desarrollado en C++ con CMake como sistema de construcción y sin dependencias pesadas más allá de librerías estándar para manejo de vectores/binarios. Se incluyeron capas convolucionales, pooling, dense, dropout y soporte a weight decay en el optimizador.

## 🔧 Requisitos

- CMake >= 3.10  
- Compilador C++17 compatible (e.g., gcc/g++ >= 7, clang >= 5)  
- Opcional: OpenCV si quieres visión adicional (p.ej., mostrar imágenes), aunque la carga de MNIST se hace en binario sin OpenCV.  

## 📁 Estructura del repositorio

- `models/`  
  - `CNN.hpp`, `MLP.hpp` (si aún existe MLP),  
  - `layers/`: define `Conv2DLayer`, `PoolingLayer`, `DenseLayer`, `DropoutLayer`, `ActivationLayer`, etc.  
- `utils/`  
  - `load_dataset.hpp`: carga imágenes/etiquetas MNIST en vectores normalizados,  
  - `activations.hpp`, `optimizer.hpp`, `loss.hpp`.  
- `main.cpp` o `test.cpp`: ejemplos de uso, entrenamiento y evaluación.  
- `CMakeLists.txt`, `run.sh`: scripts para compilar y ejecutar.  
- `README.md`: este archivo.

## 🚀 Instalación y compilación

1. Clona el repositorio y entra en la carpeta:

   ``bash
   git clone https://github.com/LeonDavisCoropuna/MLP-Multi-Layer-Perceptron.git
   cd MLP-Multi-Layer-Perceptron
  ``

2. Crea directorio de compilación y compila con CMake:

   ``bash
   mkdir build && cd build
   cmake ..
   make
   ``

   O usa el script:

   ``bash
   chmod +x run.sh
   ./run.sh main   # o ./run.sh test según tu objetivo
   ``
3. Asegúrate de tener los ficheros MNIST (`train-images.idx3-ubyte`, etc.) en la ruta esperada (`mnist_data/`). Si no, descárgalos desde el sitio oficial de Yann LeCun y colócalos en `mnist_data/`.

## ⚙️ Uso

En `main.cpp` (o `test.cpp`), se crea y entrena la CNN. Un ejemplo típico:

```cpp
#include "models/CNN.hpp"
#include "utils/load_dataset.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // 1) Cargar dataset MNIST (normalizado a [0,1])
    auto trainImages = MNISTDataset::loadImages("mnist_data/train-images.idx3-ubyte", 60000);
    auto trainLabels = MNISTDataset::loadLabels("mnist_data/train-labels.idx1-ubyte", 60000);
    auto testImages  = MNISTDataset::loadImages("mnist_data/t10k-images.idx3-ubyte", 10000);
    auto testLabels  = MNISTDataset::loadLabels("mnist_data/t10k-labels.idx1-ubyte", 10000);

    std::cout << "Cargadas " << trainImages.size() << " imágenes de entrenamiento.\n";
    std::cout << "Cargadas " << testImages.size()  << " imágenes de prueba.\n";

    // 2) Configurar optimizador y CNN
    float learningRate = 0.001f;
    float weightDecay  = 0.0005f;               // L2 regularization
    Optimizer* adam    = new Adam(learningRate, weightDecay);
    CNN cnn(learningRate, adam);

    // 3) Construir arquitectura CNN
    //    Conv + ReLU + Pooling  (input 28×28)
    cnn.addLayer(new Conv2DLayer(1,  8, 3, 28, 28, new ReLU(), 1, 0));   // -> [8, 26, 26]
    cnn.addLayer(new PoolingLayer(8, 26, 26));                          // -> [8, 13, 13]

    cnn.addLayer(new Conv2DLayer(8, 16, 3, 13, 13, new ReLU(), 1, 0));  // -> [16, 11, 11]
    cnn.addLayer(new PoolingLayer(16, 11, 11));                        // -> [16, 5, 5]

    cnn.addLayer(new Conv2DLayer(16, 32, 3, 5, 5, new ReLU(), 1, 0));   // -> [32, 3, 3]
    cnn.addLayer(new PoolingLayer(32, 3, 3));                          // -> [32, 1, 1]

    // 4) Flatten implícito en DenseLayer: asume input_size = 32
    cnn.addLayer(new DenseLayer(32, 16, new ReLU(), adam));
    cnn.addLayer(new DropoutLayer(0.2f));                              // Dropout para regularizar
    cnn.addLayer(new DenseLayer(16, 10, new Softmax(), adam));         // 10 clases
    cnn.setLoss(new CrossEntropyLoss());

    // 5) Entrenamiento
    int epochs    = 10;
    int batchSize = 32;
    cnn.train(epochs, trainImages, trainLabels, testImages, testLabels, batchSize, "output/cnn-mnist.txt");

    // 6) Evaluación puntual y visualización de ejemplo
    float accTest = cnn.evaluate(testImages, testLabels);
    std::cout << "Accuracy final en test: " << std::fixed << std::setprecision(2) << accTest << "%\n";

    // (Opcional) mostrar predicción de una imagen
    // visualizePrediction(cnn, testImages[0], testLabels[0]);

    delete adam;
    return 0;
}
```

Ajusta rutas y tamaños según tu dataset y recursos. El método `train` mostrará por época el Train Loss/Acc y Test Loss/Acc en formato:

```
[1/10] Epoch Train Loss: 2.3005, Train Acc: 11.00% | Test Loss: 2.3015, Test Acc: 10.24%
...
```

## 🧪 Experimentación y Regularización

Se recomienda experimentar variaciones en:

* **Dropout**: p.ej., 0.2 ó 0.5 entre capas Dense tras flatten. Reduce sobreajuste.
* **Weight decay (L2)**: ajustar parámetro en el optimizador (`Adam(learningRate, weightDecay)`). Valores típicos: 0.0005, 0.0001.
* **Arquitectura**: cambiar número de filtros en Conv2D (8→16→32…), profundidad, tamaño de kernels, añadir más capas.
* **Tasa de aprendizaje**: probar 0.001, 0.0005, etc.
* **Batch size**: típicamente 32 o 64.

### Ejemplo de experimentos (resultados referenciales)

1. **Arquitectura básica (sin dropout, sin weight decay)**

   * Conv(1→8) + pool → Conv(8→16) + pool → Conv(16→32) + pool → Dense(32→16) → Dense(16→10).
   * learningRate=0.001, weightDecay=0.
   * Puede converger lentamente y tender a sobreajuste si se entrena muchas épocas.

2. **Con Dropout 0.2**

   * Añadir `DropoutLayer(0.2f)` antes de la última Dense.
   * Suele mejorar generalización, especialmente con pocas imágenes o redes grandes.

3. **Con Weight Decay moderado (0.0005)**

   * En `Adam(learningRate, 0.0005f)`.
   * Penaliza pesos grandes, ayuda a generalizar. Ajustar si disminuye demasiado capacidad de aprendizaje.

4. **Combinación Dropout + Weight Decay**

   * Ejemplo: Dropout 0.2 + weightDecay 0.0005.
   * Buena práctica para equilibrar regularización.

Para cada experimento se registra Train/Test Loss y Accuracy por época, y se analiza curva para detectar sobreajuste. Al final, elegir configuración que maximice accuracy en test con pérdida razonable.

## 📈 Visualización de métricas

* El archivo de logs (`output/cnn-mnist.txt`) contendrá líneas por época con métricas.
* Puedes plotear offline (p.ej., exportar a CSV) para ver curvas de loss/accuracy.
* Opcional: implementar en C++ o en Python un script que lea ese log y grafique.

## 🛠 Estructura de código relevante

* **`Conv2DLayer`**: implementa convolución 2D manual, guarda pre-activaciones y calcula backward con derivada correcta.
* **`PoolingLayer`**: max-pooling 2×2 por defecto, con backward que propaga gradiente solo al índice max.
* **`ActivationLayer`**: envuelve función de activación genérica (ReLU, Sigmoid, Tanh, etc.).
* **`DenseLayer`**: capa fully-connected, soporta Softmax en salida.
* **`DropoutLayer`**: en modo entrenamiento descarta unidades aleatoriamente, en evaluación pasa sin cambio.
* **`Optimizer`**: p.ej. `SGD`, `Adam`, implementan update con weight decay.
* **`Loss`**: `CrossEntropyLoss` para clasificación multiclase, usuada junto con Softmax.
* **`load_dataset.hpp`**: carga binario MNIST y normaliza a \[0,1], devuelve vectores `<vector<float>>` para imágenes y `<vector<float>>` one-hot para etiquetas, o índices según convención.

## 💡 Buenas prácticas

* Comprueba tras cambios en backward que no aparezcan `NaN` en loss inicial. Usa assert o prints con `std::isfinite`.
* Verifica que `preActivations` se use en derivadas y no el valor ya activado.
* Ajusta semilla fija (`Layer::gen(32)`) para reproducibilidad.
* Usa batch size >= 16 para mejor convergencia y velocidad.
* Normaliza entrada (ya en \[0,1]). Si quisieras normalizar con media y desviación, ajusta según convenga.
* Monitoriza learning rate: si loss no baja, prueba reducir lr.

## 🔗 Enlaces y código

* Repositorio en GitHub:

  ```
  https://github.com/LeonDavisCoropuna/MLP-Multi-Layer-Perceptron.git
  ```
* Carpeta principal:

  * `models/` contiene implementación de CNN/MLP y capas.
  * `utils/` contiene cargas, funciones de activación, optimizador, pérdida.

