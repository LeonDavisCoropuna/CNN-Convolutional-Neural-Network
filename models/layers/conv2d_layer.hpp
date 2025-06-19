#pragma once
#include "layer.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class Conv2DLayer : public Layer {
private:
    // Parámetros de la convolución
    int input_channels_;
    int output_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    
    // Dimensiones de entrada/salida
    int input_height_;
    int input_width_;
    int output_height_;
    int output_width_;
    
    // Pesos y biases
    std::vector<std::vector<std::vector<std::vector<float>>>> weights_; // [output_ch][input_ch][k_h][k_w]
    std::vector<float> biases_;
    
    // Cache para forward/backward
    std::vector<float> outputs_;
    std::vector<float> inputs_cache_;
    std::vector<float> deltas_;
    
    // Gradientes
    std::vector<std::vector<std::vector<std::vector<float>>>> weights_grad_;
    std::vector<float> biases_grad_;
    
    bool is_training_;

public:
    Conv2DLayer(int input_channels, int output_channels, int kernel_size, 
           int input_height, int input_width, int stride = 1, int padding = 0)
        : input_channels_(input_channels), output_channels_(output_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          input_height_(input_height), input_width_(input_width) {
        
        // Calcular dimensiones de salida
        output_height_ = (input_height_ + 2 * padding_ - kernel_size_) / stride_ + 1;
        output_width_ = (input_width_ + 2 * padding_ - kernel_size_) / stride_ + 1;
        
        // Inicializar pesos y biases
        initialize_weights();
        
        // Inicializar gradientes
        weights_grad_ = std::vector<std::vector<std::vector<std::vector<float>>>>(
            output_channels_, std::vector<std::vector<std::vector<float>>>(
                input_channels_, std::vector<std::vector<float>>(
                    kernel_size_, std::vector<float>(kernel_size_, 0.0f))));
        biases_grad_ = std::vector<float>(output_channels_, 0.0f);
        
        // Reservar espacio para outputs y deltas
        outputs_.resize(output_channels_ * output_height_ * output_width_);
        deltas_.resize(input_channels_ * input_height_ * input_width_);
        
        is_training_ = true;
    }

    void initialize_weights() {
        // Inicialización He (Kaiming)
        float stddev = sqrtf(2.0f / (kernel_size_ * kernel_size_ * input_channels_));
        std::normal_distribution<float> dist(0.0f, stddev);
        
        weights_.resize(output_channels_);
        for (int oc = 0; oc < output_channels_; ++oc) {
            weights_[oc].resize(input_channels_);
            for (int ic = 0; ic < input_channels_; ++ic) {
                weights_[oc][ic].resize(kernel_size_);
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    weights_[oc][ic][kh].resize(kernel_size_);
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        weights_[oc][ic][kh][kw] = dist(gen);
                    }
                }
            }
        }
        
        biases_.resize(output_channels_, 0.0f);
    }

    std::vector<float> forward(const std::vector<float> &input) override {
        inputs_cache_ = input; // Guardar para backward pass
        
        // Aplicar convolución
        for (int oc = 0; oc < output_channels_; ++oc) {
            for (int oh = 0; oh < output_height_; ++oh) {
                for (int ow = 0; ow < output_width_; ++ow) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < input_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                
                                if (ih >= 0 && ih < input_height_ && iw >= 0 && iw < input_width_) {
                                    int input_idx = ic * (input_height_ * input_width_) + ih * input_width_ + iw;
                                    sum += input[input_idx] * weights_[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    
                    sum += biases_[oc];
                    int output_idx = oc * (output_height_ * output_width_) + oh * output_width_ + ow;
                    outputs_[output_idx] = sum;
                }
            }
        }
        
        return outputs_;
    }

    void backward(const std::vector<float> *targets = nullptr,
                 const Layer *next_layer = nullptr) override {
        // Limpiar deltas
        std::fill(deltas_.begin(), deltas_.end(), 0.0f);
        
        // Calcular deltas para esta capa
        const std::vector<float>& next_deltas = next_layer->get_deltas();
        
        for (int ic = 0; ic < input_channels_; ++ic) {
            for (int ih = 0; ih < input_height_; ++ih) {
                for (int iw = 0; iw < input_width_; ++iw) {
                    float delta = 0.0f;
                    
                    for (int oc = 0; oc < output_channels_; ++oc) {
                        for (int kh = 0; kh < kernel_size_; ++kh) {
                            for (int kw = 0; kw < kernel_size_; ++kw) {
                                int oh = (ih - kh + padding_) / stride_;
                                int ow = (iw - kw + padding_) / stride_;
                                
                                if (oh >= 0 && oh < output_height_ && ow >= 0 && ow < output_width_ &&
                                    (ih - kh + padding_) % stride_ == 0 &&
                                    (iw - kw + padding_) % stride_ == 0) {
                                    int next_delta_idx = oc * (output_height_ * output_width_) + oh * output_width_ + ow;
                                    delta += next_deltas[next_delta_idx] * weights_[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    
                    int delta_idx = ic * (input_height_ * input_width_) + ih * input_width_ + iw;
                    deltas_[delta_idx] = delta;
                }
            }
        }
        
        if (is_training_) {
            accumulate_gradients();
        }
    }

    void accumulate_gradients() override {
        // Calcular gradientes de los pesos
        for (int oc = 0; oc < output_channels_; ++oc) {
            for (int ic = 0; ic < input_channels_; ++ic) {
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        float grad = 0.0f;
                        
                        for (int oh = 0; oh < output_height_; ++oh) {
                            for (int ow = 0; ow < output_width_; ++ow) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                
                                if (ih >= 0 && ih < input_height_ && iw >= 0 && iw < input_width_) {
                                    int input_idx = ic * (input_height_ * input_width_) + ih * input_width_ + iw;
                                    int output_idx = oc * (output_height_ * output_width_) + oh * output_width_ + ow;
                                    grad += inputs_cache_[input_idx] * deltas_[output_idx];
                                }
                            }
                        }
                        
                        weights_grad_[oc][ic][kh][kw] += grad;
                    }
                }
            }
        }
        
        // Calcular gradientes de los biases
        for (int oc = 0; oc < output_channels_; ++oc) {
            float bias_grad = 0.0f;
            for (int oh = 0; oh < output_height_; ++oh) {
                for (int ow = 0; ow < output_width_; ++ow) {
                    int output_idx = oc * (output_height_ * output_width_) + oh * output_width_ + ow;
                    bias_grad += deltas_[output_idx];
                }
            }
            biases_grad_[oc] += bias_grad;
        }
    }

    void apply_gradients(float batch_size) override {
        // Normalizar gradientes por tamaño de batch
        float scale = 1.0f / batch_size;
        
        // Actualizar pesos
        for (int oc = 0; oc < output_channels_; ++oc) {
            for (int ic = 0; ic < input_channels_; ++ic) {
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        weights_[oc][ic][kh][kw] -= scale * weights_grad_[oc][ic][kh][kw];
                    }
                }
            }
        }
        
        // Actualizar biases
        for (int oc = 0; oc < output_channels_; ++oc) {
            biases_[oc] -= scale * biases_grad_[oc];
        }
    }

    void zero_grad() override {
        // Resetear gradientes de pesos
        for (auto &oc : weights_grad_)
            for (auto &ic : oc)
                for (auto &kh : ic)
                    for (auto &kw : kh)
                        kw = 0.0f;
        
        // Resetear gradientes de biases
        for (auto &bg : biases_grad_)
            bg = 0.0f;
    }

    // Getters
    const std::vector<float> &get_outputs() const override { return outputs_; }
    const std::vector<float> &get_deltas() const override { return deltas_; }
    
    const std::vector<std::vector<float>> &get_weights() const override {
        // Necesitarías adaptar esta función según tu implementación
        static std::vector<std::vector<float>> flat_weights;
        return flat_weights;
    }
    
    void set_weights(const std::vector<std::vector<float>> &new_weights) override {
        // Implementación para cargar pesos pre-entrenados
    }

    int input_size() const override { 
        return input_channels_ * input_height_ * input_width_; 
    }
    
    int output_size() const override { 
        return output_channels_ * output_height_ * output_width_; 
    }
    
    bool has_weights() const override { return true; }
    
    void set_training(bool is_training) override { is_training_ = is_training; }

    void update_weights () {}
};
