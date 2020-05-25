#include "WrapperSynet.hpp"
#include "Synet/Network.h"

namespace WrapperSynet {

class Tensor::impl {
public:

    impl(const Synet::Tensor<float> *ptr)
        : ptr_(ptr)
    {}

    const Synet::Tensor<float> *ptr_;
};

const float* Tensor::CpuData() const {
    return pimpl->ptr_->CpuData();
}

std::vector<size_t> Tensor::Shape() const {
    return pimpl->ptr_->Shape();
}

std::string Tensor::Name() const {
    return pimpl->ptr_->Name();
}

Tensor::Tensor() {}

class View::impl {
public:
    impl(const cv::Mat &img)
        : view_(img)
    {}

    Synet::View view_;
};

View::View(const cv::Mat &img)
    : pimpl{std::make_shared<impl>(img)}
{
}

View::~View() {
}

View::View(const View &view)
    : pimpl{view.pimpl}
{
}

class Network::impl {
public:

    impl() = default;

    Synet::Network<float> net_;
};

Network::Network()
    : pimpl{std::make_unique<impl>()}
{
}

Network::~Network()
{}

bool Network::Load(const std::string & model, const std::string & weight) {
    return pimpl->net_.Load(model, weight);
}

bool Network::Load(const char * modelData, size_t modelSize, const char * weightData, size_t weightSize) {
    return pimpl->net_.Load(modelData, modelSize, weightData, weightSize);
}

bool Network::Reshape(size_t width, size_t height, size_t batch) {
    return pimpl->net_.Reshape(width, height, batch);
}

void Network::Forward() {
    pimpl->net_.Forward();
}

const Network::TensorPtrs Network::Src() const {
    std::vector<std::shared_ptr<Tensor>> vec;
    auto ptrs = pimpl->net_.Src();
    for (const auto &p : ptrs) {
        auto tensor_shared = std::make_shared<Tensor>();
        tensor_shared->pimpl = std::make_shared<Tensor::impl>(Tensor::impl(p));
        vec.push_back(tensor_shared);
    }
    return vec;
}

const Network::TensorPtrs Network::Dst() const {
    std::vector<std::shared_ptr<Tensor>> vec;
    auto ptrs = pimpl->net_.Dst();
    for (const auto &p : ptrs) {
        auto tensor_shared = std::make_shared<Tensor>();
        tensor_shared->pimpl = std::make_shared<Tensor::impl>(Tensor::impl(p));
        vec.push_back(tensor_shared);
    }
    return vec;
}

bool Network::SetInput(const View &view, float lower, float upper) {
    return pimpl->net_.SetInput(view.pimpl->view_, lower, upper);
}

bool Network::SetInput(const View &view, const std::vector<float> &lower, const std::vector<float> &upper) {
    return pimpl->net_.SetInput(view.pimpl->view_, lower, upper);
}

bool Network::SetInputs(const std::vector<View> &views, float lower, float upper) {
    std::vector<Synet::View> v;
    for (int i = 0; i < views.size(); ++i) {
        v.push_back(views[i].pimpl->view_);
    }
    return pimpl->net_.SetInputs(v, lower, upper);
}

bool Network::SetInputs(const std::vector<View> &views, const std::vector<float> &lower, const std::vector<float> &upper) {
    std::vector<Synet::View> v;
    for (int i = 0; i < views.size(); ++i) {
        v.push_back(views[i].pimpl->view_);
    }
    return pimpl->net_.SetInputs(v, lower, upper);
}

float ToFloat(float value, float scale, float shift) {
    return value * scale + shift;
}

float ToBgrFloat(const float* src, size_t channel) {
    return src[channel];
}

void loadToSynetFloat(const float* src, size_t width, size_t height, size_t stride,
            const std::vector<float> &lower, const std::vector<float> &upper, float *dst, size_t channels) {
    float scale[3];
    const int step = 3;
    for (size_t i = 0; i < channels; ++i) {
        scale[i] = (upper[i] - lower[i]) / 255.0f;
    }

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x, src += step) {
            *dst++ = ToFloat(ToBgrFloat(src, 0), scale[0], lower[0]);
            *dst++ = ToFloat(ToBgrFloat(src, 1), scale[1], lower[1]);
            *dst++ = ToFloat(ToBgrFloat(src, 2), scale[2], lower[2]);
        }
        src += (stride - width * step);
    }
}

bool Network::SetInputs(const std::vector<cv::Mat> &views, const std::vector<float> &lower, const std::vector<float> &upper) {
    if (pimpl->net_.Src().size() != views.size() || lower.size() != upper.size()) {
        return false;
    }
    const auto& shape = pimpl->net_.NchwShape();
    if (shape.size() != 4 || shape[0] != 1) {
        return false;
    }
    if (shape[1] != 3) {
        return false;
    }
    if (lower.size() != shape[1]) {
        return false;
    }
    for (size_t i = 0; i < views.size(); ++i) {
        if (views[i].cols != shape[3] || views[i].rows != shape[2]) {
            return false;
        }
        if (views[i].type() != CV_32FC3) {
            return false;
        }
        if (!views[i].isContinuous()) {
            return false;
        }
    }
    for (size_t i = 0; i < pimpl->net_.Src().size(); ++i) {
        if (pimpl->net_.Src()[0]->Format() != Synet::TensorFormatNhwc) {
            return false;
        }
    }
    for (size_t i = 0; i < views.size(); ++i) {
        float * dst = pimpl->net_.Src()[i]->CpuData();
        loadToSynetFloat((float*)views[i].data, views[i].cols, views[i].rows, views[i].step1(),
            lower, upper, dst, 3);
    }
    return true;
}

} // namespace WrapperSynet
