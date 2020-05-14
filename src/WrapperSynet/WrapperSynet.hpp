#pragma once
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

namespace WrapperSynet {

class Tensor {
public:
    Tensor();

    const float* CpuData() const;
    std::vector<size_t> Shape() const;
    std::string Name() const;

private:
    class impl;
    std::shared_ptr<impl> pimpl;

    friend class Network;
};

class View {
public:

    View(const cv::Mat &img);
    View(const View &view);
    ~View();

private:
    class impl;
    std::shared_ptr<impl> pimpl;

    friend class Network;
};

class Network {
public:

    typedef std::vector<std::shared_ptr<Tensor>> TensorPtrs;

    Network();
    ~Network();

    bool Load(const std::string & model, const std::string & weight);
    bool Load(const char * modelData, size_t modelSize, const char * weightData, size_t weightSize);
    bool Reshape(size_t width, size_t height, size_t batch = 1);
    void Forward();

    bool SetInput(const View &view, float lower, float upper);
    bool SetInput(const View &view, const std::vector<float> &lower, const std::vector<float> &upper);

    bool SetInputs(const std::vector<View> &views, float lower, float upper);
    bool SetInputs(const std::vector<View> &views, const std::vector<float> &lower, const std::vector<float> &upper);

    const TensorPtrs Src() const;
    const TensorPtrs Dst() const;

private:
    class impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace WrapperSynet
