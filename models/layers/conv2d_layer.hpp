#pragma once
#include "layer.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include "../../utils/activations.hpp"

class Conv2DLayer : public Layer {
private:
    int inputChannels_;
    int outputChannels_;
    int kernelSize_;
    int stride_;
    int padding_;
    int inputHeight_;
    int inputWidth_;
    int outputHeight_;
    int outputWidth_;

    // Pesos y biases
    std::vector<std::vector<std::vector<std::vector<float>>>> weights_;
    std::vector<float> biases_;

    // Cache para forward/backward
    std::vector<float> inputsCache_;
    std::vector<float> preActivations_; // <-- Declara aquí
    std::vector<float> outputs_;
    std::vector<float> deltas_;

    // Gradientes
    std::vector<std::vector<std::vector<std::vector<float>>>> weightsGrad_;
    std::vector<float> biasesGrad_;

    bool isTraining_;
    ActivationFunction* activation_;

public:
    Conv2DLayer(int inputChannels, int outputChannels, int kernelSize,
                int inputHeight, int inputWidth,
                ActivationFunction* activation,
                int stride = 1, int padding = 0)
      : inputChannels_(inputChannels), outputChannels_(outputChannels),
        kernelSize_(kernelSize), stride_(stride), padding_(padding),
        inputHeight_(inputHeight), inputWidth_(inputWidth),
        activation_(activation), isTraining_(true)
    {
        outputHeight_ = (inputHeight_ + 2 * padding_ - kernelSize_) / stride_ + 1;
        outputWidth_  = (inputWidth_  + 2 * padding_ - kernelSize_) / stride_ + 1;

        initializeWeights();

        weightsGrad_.assign(
            outputChannels_,
            std::vector<std::vector<std::vector<float>>>(
                inputChannels_,
                std::vector<std::vector<float>>(
                    kernelSize_, std::vector<float>(kernelSize_, 0.0f))));
        biasesGrad_.assign(outputChannels_, 0.0f);

        // Reservar buffers
        inputsCache_.reserve(inputChannels_ * inputHeight_ * inputWidth_);
        outputs_.assign(outputChannels_ * outputHeight_ * outputWidth_, 0.0f);
        preActivations_.assign(outputChannels_ * outputHeight_ * outputWidth_, 0.0f);
        deltas_.assign(inputChannels_ * inputHeight_ * inputWidth_, 0.0f);
    }

    void initializeWeights() {
        float stddev = sqrtf(2.0f / (kernelSize_ * kernelSize_ * inputChannels_));
        std::normal_distribution<float> dist(0.0f, stddev);
        weights_.resize(outputChannels_);
        for (int oc = 0; oc < outputChannels_; ++oc) {
            weights_[oc].resize(inputChannels_);
            for (int ic = 0; ic < inputChannels_; ++ic) {
                weights_[oc][ic].resize(kernelSize_);
                for (int kh = 0; kh < kernelSize_; ++kh) {
                    weights_[oc][ic][kh].resize(kernelSize_);
                    for (int kw = 0; kw < kernelSize_; ++kw) {
                        weights_[oc][ic][kh][kw] = dist(gen);
                    }
                }
            }
        }
        biases_.assign(outputChannels_, 0.0f);
    }

    std::vector<float> forward(const std::vector<float> &input) override {
        inputsCache_ = input;
        // Asegura tamaño de outputs y preActivations
        int outSize = outputChannels_ * outputHeight_ * outputWidth_;
        outputs_.assign(outSize, 0.0f);
        preActivations_.assign(outSize, 0.0f);

        for (int oc = 0; oc < outputChannels_; ++oc) {
            for (int oh = 0; oh < outputHeight_; ++oh) {
                for (int ow = 0; ow < outputWidth_; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < inputChannels_; ++ic) {
                        for (int kh = 0; kh < kernelSize_; ++kh) {
                            for (int kw = 0; kw < kernelSize_; ++kw) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                if (ih >= 0 && ih < inputHeight_ && iw >= 0 && iw < inputWidth_) {
                                    int inputIdx = ic * inputHeight_ * inputWidth_ + ih * inputWidth_ + iw;
                                    sum += input[inputIdx] * weights_[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    sum += biases_[oc];
                    int outIdx = oc * outputHeight_ * outputWidth_ + oh * outputWidth_ + ow;
                    preActivations_[outIdx] = sum;
                    outputs_[outIdx] = activation_ ? activation_->activate(sum) : sum;
                }
            }
        }
        return outputs_;
    }

    void backward(const std::vector<float>* targets = nullptr,
                  const Layer* nextLayer = nullptr) override {
        // Limpia deltas
        std::fill(deltas_.begin(), deltas_.end(), 0.0f);
        const std::vector<float>& nextDeltas = nextLayer->get_deltas();

        // Propaga error hacia atrás
        for (int ic = 0; ic < inputChannels_; ++ic) {
            for (int ih = 0; ih < inputHeight_; ++ih) {
                for (int iw = 0; iw < inputWidth_; ++iw) {
                    float deltaSum = 0.0f;
                    for (int oc = 0; oc < outputChannels_; ++oc) {
                        for (int kh = 0; kh < kernelSize_; ++kh) {
                            for (int kw = 0; kw < kernelSize_; ++kw) {
                                int oh = (ih - kh + padding_) / stride_;
                                int ow = (iw - kw + padding_) / stride_;
                                if (oh >= 0 && oh < outputHeight_ &&
                                    ow >= 0 && ow < outputWidth_ &&
                                    (ih - kh + padding_) % stride_ == 0 &&
                                    (iw - kw + padding_) % stride_ == 0) {
                                    int nextIdx = oc * (outputHeight_ * outputWidth_) + oh * outputWidth_ + ow;
                                    float z = preActivations_[nextIdx];
                                    float deriv = activation_ ? activation_->derivative(z) : 1.0f;
                                    deltaSum += nextDeltas[nextIdx] * deriv * weights_[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    int deltaIdx = ic * inputHeight_ * inputWidth_ + ih * inputWidth_ + iw;
                    deltas_[deltaIdx] = deltaSum;
                }
            }
        }
        if (isTraining_) {
            accumulate_gradients();
        }
    }

    void accumulate_gradients() override {
        // Asegura tamaños
        for (int oc = 0; oc < outputChannels_; ++oc) {
            for (int ic = 0; ic < inputChannels_; ++ic) {
                for (int kh = 0; kh < kernelSize_; ++kh) {
                    for (int kw = 0; kw < kernelSize_; ++kw) {
                        float grad = 0.0f;
                        for (int oh = 0; oh < outputHeight_; ++oh) {
                            for (int ow = 0; ow < outputWidth_; ++ow) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                if (ih >= 0 && ih < inputHeight_ && iw >= 0 && iw < inputWidth_) {
                                    int inputIdx = ic * inputHeight_ * inputWidth_ + ih * inputWidth_ + iw;
                                    int outIdx = oc * outputHeight_ * outputWidth_ + oh * outputWidth_ + ow;
                                    grad += inputsCache_[inputIdx] * deltas_[outIdx];
                                }
                            }
                        }
                        weightsGrad_[oc][ic][kh][kw] += grad;
                    }
                }
            }
        }
        for (int oc = 0; oc < outputChannels_; ++oc) {
            float biasGrad = 0.0f;
            for (int oh = 0; oh < outputHeight_; ++oh) {
                for (int ow = 0; ow < outputWidth_; ++ow) {
                    int outIdx = oc * outputHeight_ * outputWidth_ + oh * outputWidth_ + ow;
                    biasGrad += deltas_[outIdx];
                }
            }
            biasesGrad_[oc] += biasGrad;
        }
    }

    void apply_gradients(float batchSize) override {
        float scale = 1.0f / batchSize;
        for (int oc = 0; oc < outputChannels_; ++oc) {
            for (int ic = 0; ic < inputChannels_; ++ic) {
                for (int kh = 0; kh < kernelSize_; ++kh) {
                    for (int kw = 0; kw < kernelSize_; ++kw) {
                        weights_[oc][ic][kh][kw] -= scale * weightsGrad_[oc][ic][kh][kw];
                    }
                }
            }
        }
        for (int oc = 0; oc < outputChannels_; ++oc) {
            biases_[oc] -= scale * biasesGrad_[oc];
        }
    }

    void zero_grad() override {
        for (auto &oc : weightsGrad_)
            for (auto &ic : oc)
                for (auto &kh : ic)
                    for (auto &kw : kh)
                        kw = 0.0f;
        for (auto &bg : biasesGrad_)
            bg = 0.0f;
    }

    const std::vector<float>& get_outputs() const override { return outputs_; }
    const std::vector<float>& get_deltas() const override { return deltas_; }
    const std::vector<std::vector<float>>& get_weights() const override {
        static std::vector<std::vector<float>> dummy;
        return dummy;
    }
    void set_weights(const std::vector<std::vector<float>>& ) override {}
    int input_size() const override { return inputChannels_ * inputHeight_ * inputWidth_; }
    int output_size() const override { return outputChannels_ * outputHeight_ * outputWidth_; }
    bool has_weights() const override { return true; }
    void set_training(bool isTraining) override { isTraining_ = isTraining; }
    void update_weights() override {} 
};
