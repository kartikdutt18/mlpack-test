// Include all required libraries.
#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace std;

size_t inputWidth = 1;
size_t inputHeight = 1;

Sequential<>* ConvolutionBlock(const size_t inSize,
                                const size_t outSize,
                                const size_t kernelWidth,
                                const size_t kernelHeight,
                                const size_t strideWidth = 1,
                                const size_t strideHeight = 1,
                                const size_t padW = 0,
                                const size_t padH = 0)
{
    Sequential<>* convolutionBlock;
    convolutionBlock->Add<Convolution<>>(inSize, outSize, kernelWidth,
        kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
        inputHeight);
    convolutionBlock->Add<ReLULayer<>>();
    return convolutionBlock;
}
int main()
{
  Sequential<>* alexNet;
  alexNet->Add(ConvolutionBlock(1, 1, 1, 1));
  return 0;
}