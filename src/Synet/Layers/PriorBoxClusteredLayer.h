/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#pragma once

#include "Synet/Common.h"
#include "Synet/Layer.h"

namespace Synet
{
    template <class T> class PriorBoxClusteredLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PriorBoxClusteredLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const PriorBoxClusteredParam & param = this->Param().priorBoxClustered();

            _heights = param.heights();
            _widths = param.widths();
            _clip = param.clip();
            _variance = param.variance();

            _numPriors = _widths.size();
            if (_variance.empty())
                _variance.push_back(0.1f);

            _trans = src[0]->Format() == TensorFormatNhwc;
            size_t layerH, layerW;
            if (_trans)
            {
                layerH = src[0]->Axis(-3);
                layerW = src[0]->Axis(-2);
                _imgH = param.imgH() ? param.imgH() : src[1]->Axis(-3);
                _imgW = param.imgW() ? param.imgW() : src[1]->Axis(-2);
            }
            else
            {
                layerH = src[0]->Axis(-2);
                layerW = src[0]->Axis(-1);
                _imgH = param.imgH() ? param.imgH() : src[1]->Axis(-2);
                _imgW = param.imgW() ? param.imgW() : src[1]->Axis(-1);
            }

            _stepH = param.stepH() == 0 ? param.step() : param.stepH();
            _stepW = param.stepW() == 0 ? param.step() : param.stepW();
            if (_stepH == 0 && _stepW == 0) 
            {
                _stepH = float(_imgH) / layerH;
                _stepW = float(_imgW) / layerW;
            }
            _offset = param.offset();

            Shape shape(3);
            shape[0] = 1;
            shape[1] = 2;
            shape[2] = layerW * layerH * _numPriors * 4;
            dst[0]->Reshape(shape);
            float * pDst0 = dst[0]->CpuData({ 0, 0, 0 });
            float * pDst1 = dst[0]->CpuData({ 0, 1, 0 });
            size_t varSize = _variance.size();

            for (size_t h = 0; h < layerH; ++h) 
            {
                for (size_t w = 0; w < layerW; ++w) 
                {
                    float centerY = (h + _offset) * _stepH;
                    float centerX = (w + _offset) * _stepW;

                    for (size_t s = 0; s < _numPriors; ++s) 
                    {
                        float boxW = _widths[s];
                        float boxH = _heights[s];

                        float xmin = (centerX - boxW / 2.0f) / _imgW;
                        float ymin = (centerY - boxH / 2.0f) / _imgH;
                        float xmax = (centerX + boxW / 2.0f) / _imgW;
                        float ymax = (centerY + boxH / 2.0f) / _imgH;
                        if (_clip) 
                        {
                            xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                            ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                            xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                            ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                        }
                        pDst0[h * layerW * _numPriors * 4 + w * _numPriors * 4 + s * 4 + 0] = xmin;
                        pDst0[h * layerW * _numPriors * 4 + w * _numPriors * 4 + s * 4 + 1] = ymin;
                        pDst0[h * layerW * _numPriors * 4 + w * _numPriors * 4 + s * 4 + 2] = xmax;
                        pDst0[h * layerW * _numPriors * 4 + w * _numPriors * 4 + s * 4 + 3] = ymax;
                        for (size_t j = 0; j < varSize; j++)
                            pDst1[h * layerW * _numPriors * varSize + w * _numPriors * varSize +  s * varSize + j] = _variance[j];
                    }
                }
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }

    private:

        Floats _heights, _widths, _variance;
        bool _clip, _trans;
        size_t _numPriors, _imgW, _imgH;
        float _stepW, _stepH, _offset;
    };
}