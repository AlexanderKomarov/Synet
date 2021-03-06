/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "Test/TestCompare.h"

#include "Synet/Converters/InferenceEngine.h"

#ifdef SYNET_OTHER_RUN

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <ie_blob.h>
#include <ie_plugin_dispatcher.hpp>
#include <ext_list.hpp>
#include <inference_engine.hpp>

namespace Test
{
    struct InferenceEngineNetwork : public Network
    {
        InferenceEngineNetwork()
        {
        }

        virtual ~InferenceEngineNetwork()
        {
        }

        virtual String Name() const
        {
            return "Inference Engine";
        }

        virtual size_t SrcCount() const
        {
            return _ieInput.size();
        }

        virtual Shape SrcShape(size_t index) const
        {
            Shape shape = _ieInput[index]->getTensorDesc().getDims();
            if (_batchSize > 1)
            {
                assert(shape.size() == 4);
                shape[0] = _batchSize;
            }
            return shape;
        }

        virtual size_t SrcSize(size_t index) const
        {
            Shape shape = SrcShape(index);
            size_t size = 1;
            for (size_t i = 0; i < shape.size(); ++i)
                size *= shape[i];
            return size;
        }

        virtual bool Init(const String & model, const String & weight, const Options & options, const TestParam & param)
        {
            TEST_PERF_FUNC();

            _regionThreshold = options.regionThreshold;
            try
            {
                _iePlugin = InferenceEngine::PluginDispatcher({ "" }).getPluginByDevice("CPU");
                _iePlugin.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>());

                InferenceEngine::CNNNetReader reader;
                reader.ReadNetwork(model);
                reader.ReadWeights(weight);
                _ieNetwork = reader.getNetwork();

                _inputNames.clear();
                InferenceEngine::InputsDataMap inputsInfo = _ieNetwork.getInputsInfo();
                for (InferenceEngine::InputsDataMap::iterator it = inputsInfo.begin(); it != inputsInfo.end(); ++it)
                    _inputNames.push_back(it->first);

                _outputNames.clear();
                if (param.output().size())
                {
                    for (size_t i = 0; i < param.output().size(); ++i)
                        _outputNames.push_back(param.output()[i].name());
                }
                else
                {
                    InferenceEngine::OutputsDataMap outputsInfo = _ieNetwork.getOutputsInfo();
                    for (InferenceEngine::OutputsDataMap::iterator it = outputsInfo.begin(); it != outputsInfo.end(); ++it)
                        _outputNames.push_back(it->first);
                }

                AddInterimOutput(options);

                StringMap config;
                config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(options.workThreads);
                _batchSize = 1;
                if (options.batchSize > 1)
                {
                    try
                    {
                        _ieNetwork.setBatchSize(options.batchSize);
                        config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
                        _ieExecutableNetwork = _iePlugin.LoadNetwork(_ieNetwork, config);
                        _ieInferRequest = _ieExecutableNetwork.CreateInferRequest();
                        _ieInferRequest.SetBatch(options.batchSize);
                    }
                    catch (std::exception & e)
                    {
                        std::cout << "Inference Engine init trouble: '" << e.what() << "', try to emulate batch > 1." << std::endl;
                        _batchSize = options.batchSize;
                        _ieNetwork.setBatchSize(1);
                        config.erase(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED);
                        _ieExecutableNetwork = _iePlugin.LoadNetwork(_ieNetwork, config);
                        _ieInferRequest = _ieExecutableNetwork.CreateInferRequest();
                    }
                }
                else
                {
                    _ieExecutableNetwork = _iePlugin.LoadNetwork(_ieNetwork, config);
                    _ieInferRequest = _ieExecutableNetwork.CreateInferRequest();
                }

                _ieInput.clear();
                for (size_t i = 0; i < _inputNames.size(); ++i)
                    _ieInput.push_back(_ieInferRequest.GetBlob(_inputNames[i]));

                _ieOutput.clear();
                for (size_t i = 0; i < _outputNames.size(); ++i)
                    _ieOutput.push_back(_ieInferRequest.GetBlob(_outputNames[i]));

                _ieInterim.clear();
                for (size_t i = 0; i < _interimNames.size(); ++i)
                    _ieInterim.push_back(_ieInferRequest.GetBlob(_interimNames[i]));
            }
            catch (std::exception & e)
            {
                std::cout << "Inference Engine init error: " << e.what() << std::endl;
                return false;
            }

            {
                Vectors stub(SrcCount());
                for (size_t i = 0; i < SrcCount(); ++i)
                    stub[i].resize(SrcSize(i));
                SetInput(stub, 0);
                _ieInferRequest.Infer();
            }

            if(options.debugPrint & (1 << Synet::DebugPrintLayerDst))
                _ieExecutableNetwork.GetExecGraphInfo().serialize(MakePath(options.outputDirectory, "ie_exec_outs.xml"));

            return true;
        }

        virtual const Vectors & Predict(const Vectors & src)
        {
            if (_batchSize == 1)
            {
                SetInput(src, 0);
                {
                    TEST_PERF_FUNC();
                    _ieInferRequest.Infer();
                }
                SetOutput(0);
            }
            else
            {
                TEST_PERF_BLOCK("batch emulation");
                for (size_t b = 0; b < _batchSize; ++b)
                {
                    SetInput(src, b);
                    _ieInferRequest.Infer();
                    SetOutput(b);
                }
            }
            return _output;
        }

        virtual void DebugPrint(std::ostream & os, int flag, int first, int last, int precision)
        {
            for (size_t i = 0; i < _ieInterim.size(); ++i)
                DebugPrint(os, *_ieInterim[i], _interimNames[i], flag, first, last, precision);

            for (size_t o = 0; o < _ieOutput.size(); ++o)
                DebugPrint(os, *_ieOutput[o], _outputNames[o], flag, first, last, precision);
        }

        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const
        {
            Regions regions;
            for (size_t i = 0; i < _output[0].size(); i += 7)
            {
                if (_output[0][i + 2] > threshold)
                {
                    Region region;
                    region.id = (size_t)_output[0][i + 1];
                    region.prob = _output[0][i + 2];
                    region.x = size.x*(_output[0][i + 3] + _output[0][i + 5]) / 2.0f;
                    region.y = size.y*(_output[0][i + 4] + _output[0][i + 6]) / 2.0f;
                    region.w = size.x*(_output[0][i + 5] - _output[0][i + 3]);
                    region.h = size.y*(_output[0][i + 6] - _output[0][i + 4]);
                    regions.push_back(region);
                }
            }
            return regions;
        }

    private:
        typedef InferenceEngine::SizeVector Sizes;
        typedef std::map<std::string, std::string> StringMap;

        InferenceEngine::InferencePlugin _iePlugin;
        InferenceEngine::CNNNetwork _ieNetwork;
        InferenceEngine::ExecutableNetwork _ieExecutableNetwork;
        InferenceEngine::InferRequest _ieInferRequest;
        std::vector<InferenceEngine::Blob::Ptr> _ieInput, _ieOutput, _ieInterim;
        Vectors _output;
        Strings _inputNames, _outputNames, _interimNames;
        size_t _batchSize;

        void SetInput(const Vectors & x, size_t b)
        {
            assert(_ieInput.size() == x.size() && _ieInput[0]->getTensorDesc().getLayout() == InferenceEngine::Layout::NCHW);
            for (size_t i = 0; i < x.size(); ++i)
            {
                const InferenceEngine::SizeVector & dims = _ieInput[i]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector & strides = _ieInput[i]->getTensorDesc().getBlockingDesc().getStrides();
                const float * src = x[i].data() + b * x[i].size() / _batchSize;
                float * dst = (float*)_ieInput[i]->buffer();
                SetInput(dims, strides, 0, src, dst);
            }
        }

        void SetInput(const Sizes & dims, const Sizes & strides, size_t current, const float * src, float * dst)
        {
            if (current == dims.size() - 1)
            {
                memcpy(dst, src, dims[current] * sizeof(float));
            }
            else
            {
                size_t srcStride = 1;
                for (size_t i = current + 1; i < dims.size(); ++i)
                    srcStride *= dims[i];
                size_t dstStride = strides[current];
                for (size_t i = 0; i < dims[current]; ++i)
                    SetInput(dims, strides, current + 1, src + i * srcStride, dst + i * dstStride);
            }
        }

        void SetOutput(size_t b)
        {
            _output.resize(_ieOutput.size());
            for (size_t o = 0; o < _ieOutput.size(); ++o)
            {
                const InferenceEngine::SizeVector & dims = _ieOutput[o]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector & strides = _ieOutput[o]->getTensorDesc().getBlockingDesc().getStrides();
                if (dims.size() == 4 && dims[3] == 7)
                {
                    const float * pOut = _ieOutput[o]->buffer();
                    if (b == 0)
                        _output[o].clear();
                    for (size_t j = 0; j < dims[2]; ++j, pOut += 7)
                    {
                        if (pOut[0] == -1 || pOut[2] <= _regionThreshold)
                            break;
                        size_t size = _output[o].size();
                        _output[o].resize(size + 7);
                        _output[o][size + 0] = pOut[0];
                        _output[o][size + 1] = pOut[1];
                        _output[o][size + 2] = pOut[2];
                        _output[o][size + 3] = pOut[3];
                        _output[o][size + 4] = pOut[4];
                        _output[o][size + 5] = pOut[5];
                        _output[o][size + 6] = pOut[6];
                    }
                    SortDetectionOutput(_output[o].data(), _output[o].size());
                }
                else
                {
                    size_t size = 1;
                    for (size_t i = 0; i < dims.size(); ++i)
                        size *= dims[i];
                    _output[o].resize(size*_batchSize);
                    const float * pOut = _ieOutput[o]->buffer();
                    SetOutput(dims, strides, 0, pOut, _output[o].data() + b * size);
                }
            }
        }

        template<class T> void SetOutput(const Sizes & dims, const Sizes & strides, size_t current, const T * src, T * dst)
        {
            if (current == dims.size() - 1)
                memcpy(dst, src, dims[current] * sizeof(T));
            else
            {
                size_t srcStride = strides[current];
                size_t dstStride = 1;
                for (size_t i = current + 1; i < dims.size(); ++i)
                    dstStride *= dims[i];
                for(size_t i = 0; i < dims[current]; ++i)
                    SetOutput(dims, strides, current + 1, src + i * srcStride, dst + i * dstStride);
            }
        }

        void AddInterimOutput(const Options& options)
        {
            _interimNames.clear();
            if ((options.debugPrint & (1 << Synet::DebugPrintLayerDst)) == 0)
                return;

            StringMap config, interim;
            InferenceEngine::ExecutableNetwork exec = _iePlugin.LoadNetwork(_ieNetwork, config);
            InferenceEngine::CNNNetwork net = exec.GetExecGraphInfo();
            net.serialize(MakePath(options.outputDirectory, "ie_exec_orig.xml"));
            for (InferenceEngine::details::CNNNetworkIterator it = net.begin(); it != net.end(); ++it)
            {
                const InferenceEngine::CNNLayer& layer = **it;
                StringMap::const_iterator names = layer.params.find("originalLayersNames");
                StringMap::const_iterator order = layer.params.find("execOrder");
                if (names != layer.params.end() && names->second.size() && order != layer.params.end())
                {
                    String name = Synet::Separate(names->second, ",").back();
                    bool unique = true;
                    for (size_t o = 0; unique && o < _outputNames.size(); ++o)
                        if (_outputNames[o] == name)
                            unique = false;
                    if(unique)
                        interim[order->second] = name;
                }
            }
            for (StringMap::const_iterator it = interim.begin(); it != interim.end(); ++it)
            {
                String name = it->second;
                _ieNetwork.addOutput(name);
                _interimNames.push_back(name);
            }
        }

        void DebugPrint(std::ostream& os, InferenceEngine::Blob & blob, const String & name, int flag, int first, int last, int precision)
        {
            os << "Layer: " << name << " : " << GetLayerType(name) << std::endl;
            Sizes dims = blob.getTensorDesc().getDims();
            const Sizes & strides = blob.getTensorDesc().getBlockingDesc().getStrides();
            Synet::TensorFormat format = Synet::TensorFormatUnknown;
            if (blob.getTensorDesc().getLayout() == InferenceEngine::Layout::NHWC)
                format = Synet::TensorFormatNhwc;
            dims[0] = _batchSize;
            switch (blob.getTensorDesc().getPrecision())
            {
            case InferenceEngine::Precision::FP32:
            {
                Synet::Tensor<float> tensor(dims, format);
                const float* pOut = blob.buffer();
                SetOutput(dims, strides, 0, pOut, tensor.CpuData());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case InferenceEngine::Precision::U8:
            {
                Synet::Tensor<uint8_t> tensor(dims, format);
                const uint8_t* pOut = blob.buffer();
                SetOutput(dims, strides, 0, pOut, tensor.CpuData());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case InferenceEngine::Precision::I8:
            {
                Synet::Tensor<int8_t> tensor(dims, format);
                const int8_t* pOut = blob.buffer();
                SetOutput(dims, strides, 0, pOut, tensor.CpuData());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            default:
                std::cout << "Can't debug print for layer '" << name << "' , unknown precision: " << blob.getTensorDesc().getPrecision() << std::endl;
                break;
            }
        }

        String GetLayerType(const String& name)
        {
            for (InferenceEngine::details::CNNNetworkIterator it = _ieNetwork.begin(); it != _ieNetwork.end(); ++it)
            {
                const InferenceEngine::CNNLayer& layer = **it;
                if (layer.name == name)
                    return layer.type;
            }
            return String();
        }
    };
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#else //SYNET_OTHER_RUN
namespace Test
{
    struct InferenceEngineNetwork : public Network
    {
    };
}
#endif//SYNET_OTHER_RUN

namespace Test
{
    bool ConvertTextWeightToBinary(const String & src, const String & dst)
    {
        std::ifstream ifs(src.c_str());
        if (!ifs.is_open())
        {
            std::cout << "Can't open input text file '" << src << "' with weight!" << std::endl;
            return false;
        }
        std::ofstream ofs(dst.c_str(), std::ofstream::binary);
        if (!ofs.is_open())
        {
            std::cout << "Can't open output binary file '" << dst << "' with weight!" << std::endl;
            ifs.close();
            return false;
        }
        while (!ifs.eof())
        {
            std::string str;
            ifs >> str;
            float val = std::stof(str);
            ofs.write((char*)&val, 4);
        }
        ifs.close();
        ofs.close();
        return true;
    }
}

Test::PerformanceMeasurerStorage Test::PerformanceMeasurerStorage::s_storage;

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    if (options.mode == "convert")
    {
        SYNET_PERF_FUNC();
        std::cout << "Convert network from Inference Engine to Synet : ";
        options.result = Synet::ConvertInferenceEngineToSynet(options.otherModel, options.otherWeight, options.tensorFormat == 1, options.synetModel, options.synetWeight);
        std::cout << (options.result ? "OK." : " Conversion finished with errors!") << std::endl;
    }
    else if (options.mode == "compare")
    {
        Test::Comparer<Test::InferenceEngineNetwork> comparer(options);
        options.result = comparer.Run();
    }
    else if (options.mode == "txt2bin")
    {
        std::cout << "Convert text weight to binary : ";
        options.result = Test::ConvertTextWeightToBinary(options.textWeight, options.otherWeight);
        std::cout << (options.result ? "OK." : " Conversion finished with errors!") << std::endl;
    }
    else
        std::cout << "Unknown mode : " << options.mode << std::endl;

    return options.result ? 0 : 1;
}