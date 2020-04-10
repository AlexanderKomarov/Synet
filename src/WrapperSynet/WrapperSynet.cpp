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
    impl(cv::Mat img)
        : view_(img)
    {}

    Synet::View view_;
};

View::View(cv::Mat img)
    : pimpl{std::make_unique<impl>(img)}
{
}

View::~View()
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
} // namespace WrapperSynet
