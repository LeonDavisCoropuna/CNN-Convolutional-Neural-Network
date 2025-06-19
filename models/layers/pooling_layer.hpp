#pragma once
#include "layer.hpp"
#include <algorithm>
#include <cassert>

class PoolingLayer : public Layer {
private:
  int input_channels_, input_height_, input_width_;
  int pool_size_, stride_;
  int output_height_, output_width_;

  std::vector<float> outputs_;
  std::vector<float> deltas_;
  std::vector<int> max_indices_; // Para backward
  std::vector<float> input_cache_;
  bool is_training_;

public:
  PoolingLayer(int input_channels, int input_height, int input_width,
               int pool_size = 2, int stride = 2)
      : input_channels_(input_channels), input_height_(input_height),
        input_width_(input_width), pool_size_(pool_size), stride_(stride),
        is_training_(true) {
    output_height_ = (input_height_ - pool_size_) / stride_ + 1;
    output_width_ = (input_width_ - pool_size_) / stride_ + 1;

    outputs_.resize(input_channels_ * output_height_ * output_width_);
    deltas_.resize(input_channels_ * input_height_ * input_width_);
    max_indices_.resize(outputs_.size(), -1);
  }

  std::vector<float> forward(const std::vector<float> &input) override {
    input_cache_ = input;
    std::fill(outputs_.begin(), outputs_.end(), 0.0f);
    std::fill(max_indices_.begin(), max_indices_.end(), -1);

    for (int c = 0; c < input_channels_; ++c) {
      for (int oh = 0; oh < output_height_; ++oh) {
        for (int ow = 0; ow < output_width_; ++ow) {
          float max_val = -1e30f;
          int max_idx = -1;

          for (int ph = 0; ph < pool_size_; ++ph) {
            for (int pw = 0; pw < pool_size_; ++pw) {
              int ih = oh * stride_ + ph;
              int iw = ow * stride_ + pw;
              int input_idx = c * input_height_ * input_width_ + ih * input_width_ + iw;

              if (ih < input_height_ && iw < input_width_) {
                float val = input[input_idx];
                if (val > max_val) {
                  max_val = val;
                  max_idx = input_idx;
                }
              }
            }
          }

          int out_idx = c * output_height_ * output_width_ + oh * output_width_ + ow;
          outputs_[out_idx] = max_val;
          max_indices_[out_idx] = max_idx;
        }
      }
    }

    return outputs_;
  }

  void backward(const std::vector<float> *targets = nullptr,
                const Layer *next_layer = nullptr) override {
    std::fill(deltas_.begin(), deltas_.end(), 0.0f);
    if (!next_layer) return;

    const std::vector<float> &next_deltas = next_layer->get_deltas();

    for (size_t i = 0; i < outputs_.size(); ++i) {
      int max_idx = max_indices_[i];
      if (max_idx >= 0)
        deltas_[max_idx] += next_deltas[i];
    }
  }

  void update_weights() override {}
  void accumulate_gradients() override {}
  void apply_gradients(float) override {}
  void zero_grad() override {}

  const std::vector<float> &get_outputs() const override {
    return outputs_;
  }

  const std::vector<std::vector<float>> &get_weights() const override {
    static const std::vector<std::vector<float>> empty;
    return empty;
  }

  const std::vector<float> &get_deltas() const override {
    return deltas_;
  }

  int input_size() const override {
    return input_channels_ * input_height_ * input_width_;
  }

  int output_size() const override {
    return outputs_.size();
  }

  bool has_weights() const override {
    return false;
  }

  void set_training(bool is_training) override {
    is_training_ = is_training;
  }
};
