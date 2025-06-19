#include <vector>
#include <cmath>     // Para log(), pow(), etc.
#include <stdexcept> // Para manejar errores

class Loss
{
public:
  // Método virtual puro para calcular la pérdida
  virtual float compute(const std::vector<float> &predictions, const std::vector<float> &targets) = 0;

  // Método virtual puro para calcular el gradiente (derivada de la pérdida)
  virtual std::vector<float> gradient(const std::vector<float> &predictions, const std::vector<float> &targets) = 0;

  // Destructor virtual para evitar memory leaks
  virtual ~Loss() = default;
};

class MSELoss : public Loss
{
public:
  float compute(const std::vector<float> &predictions, const std::vector<float> &targets) override
  {
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i)
    {
      loss += 0.5f * std::pow(targets[i] - predictions[i], 2);
    }
    return loss;
  }

  std::vector<float> gradient(const std::vector<float> &predictions, const std::vector<float> &targets) override
  {
    std::vector<float> grad(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i)
    {
      grad[i] = predictions[i] - targets[i]; // Derivada de MSE: (y_pred - y_true)
    }
    return grad;
  }
};

class CrossEntropyLoss : public Loss
{
public:
  float compute(const std::vector<float> &predictions, const std::vector<float> &targets) override
  {
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i)
    {
      // Evitar log(0) con un pequeño epsilon (1e-10)
      loss += -targets[i] * std::log(predictions[i] + 1e-10f);
    }
    return loss;
  }

  std::vector<float> gradient(const std::vector<float> &predictions, const std::vector<float> &targets) override
  {
    // Asume que la última capa usa Softmax (el gradiente es y_pred - y_true)
    std::vector<float> grad(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i)
    {
      grad[i] = predictions[i] - targets[i];
    }
    return grad;
  }
};

class BCELoss : public Loss {
  public:
      float compute(const std::vector<float>& predictions, const std::vector<float>& targets) override {
          float loss = 0.0f;
          for (size_t i = 0; i < predictions.size(); ++i) {
              // Evitar overflow numérico (clip predictions entre [epsilon, 1-epsilon])
              float y_pred = std::max(1e-10f, std::min(1.0f - 1e-10f, predictions[i]));
              float y_true = targets[i];
              loss += - (y_true * log(y_pred) + (1.0f - y_true) * log(1.0f - y_pred));
          }
          return loss / predictions.size(); // Pérdida promedio
      }
  
      std::vector<float> gradient(const std::vector<float>& predictions, const std::vector<float>& targets) override {
          std::vector<float> grad(predictions.size());
          for (size_t i = 0; i < predictions.size(); ++i) {
              float y_pred = predictions[i];
              float y_true = targets[i];
              // Gradiente de BCE: (y_pred - y_true) / (y_pred * (1 - y_pred))
              grad[i] = (y_pred - y_true) / (y_pred * (1.0f - y_pred) + 1e-10f); // +epsilon para estabilidad
          }
          return grad;
      }
  };