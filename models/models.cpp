// Include all required libraries.
#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer_names.hpp>
#include <mlpack/core/data/split_data.hpp>
#include "../../models/Kaggle/kaggle_utils.hpp"
#include <ensmallen.hpp>

using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace std;
using namespace ens;

class AlexNet
{
public:
  //! Create the AlexNet object.
  AlexNet();

  /**
   * AlexNet constructor intializes input shape, number of classes
   * and width multiplier.
   *
   * @param inputChannels Number of input channels of the input image.
   * @param inputWidth Width of the input image.
   * @param inputHeight Height of the input image.
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param includeTop whether to include the fully-connected layer at 
   *        the top of the network.
   * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
   */
  AlexNet(const size_t inputChannel,
          const size_t inputWidth,
          const size_t inputHeight,
          const size_t numClasses = 1000,
          const bool includeTop = true,
          const std::string &weights = "None");

  /**
   * AlexNet constructor intializes input shape, number of classes
   * and width multiplier.
   *  
   * @param inputShape A three-valued tuple indicating input shape.
   *                   First value is number of Channels (Channels-First).
   *                   Second value is input height.
   *                   Third value is input width..
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param includeTop whether to include the fully-connected layer at 
   *        the top of the network.
   * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
   */
  AlexNet(const std::tuple<size_t, size_t, size_t> inputShape,
          const size_t numClasses = 1000,
          const bool includeTop = true,
          const std::string &weights = "None");

  // Custom Destructor.
  ~AlexNet()
  {
    delete alexNet;
  }

  /** 
   * Defines Model Architecture.
   * 
   * @return Sequential Pointer to the sequential AlexNet model.
   */
  Sequential<> *CompileModel();

  /**
   * Load model from a path.
   * 
   *
   * @param filePath Path to load the model from.
   * @return Sequential Pointer to a sequential model.
   */
  Sequential<> *LoadModel(const std::string &filePath);

  /**
   * Save model to a location.
   *
   * @param filePath Path to save the model to.
   */
  void SaveModel(const std::string &filePath);

  /**
   * Return output shape of model.
   * @returns outputShape of size_t type.
   */
  size_t OutputShape() { return outputShape; };

  /**
   * Returns compiled version of model.
   * If called without compiling would result in empty Sequetial
   * Pointer.
   * 
   * @return Sequential Pointer to a sequential model.
   */
  Sequential<> *GetModel() { return alexNet; };

private:
  /**
   * Returns AdaptivePooling Block.
   * 
   * @param outputlWidth Width of the output.
   * @param outputHeight Height of the output.
   */
  void AdaptivePoolingBlock(const size_t outputWidth,
                                     const size_t outputHeight)
  {
    Sequential<> *poolingBlock = new Sequential<>();
    const size_t strideWidth = std::floor(inputWidth / outputWidth);
    const size_t strideHeight = std::floor(inputHeight / outputHeight);

    const size_t kernelWidth = inputWidth - (outputWidth - 1) * strideWidth;
    const size_t kernelHeight = inputHeight - (outputHeight - 1) * strideHeight;
    poolingBlock->Add<MaxPooling<>>(kernelWidth, kernelHeight,
                                    strideWidth, strideHeight);
    // Update inputWidth and inputHeight.
    inputWidth = outputWidth;
    inputHeight = outputHeight;
    alexNet->Add(poolingBlock);
    return;
  }

  /**
   * Returns Convolution Block.
   * 
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   */
  void ConvolutionBlock(const size_t inSize,
                                 const size_t outSize,
                                 const size_t kernelWidth,
                                 const size_t kernelHeight,
                                 const size_t strideWidth = 1,
                                 const size_t strideHeight = 1,
                                 const size_t padW = 0,
                                 const size_t padH = 0)
  {
    Sequential<> *convolutionBlock = new Sequential<>();
    convolutionBlock->Add<Convolution<>>(inSize, outSize, kernelWidth,
                                         kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
                                         inputHeight);
    convolutionBlock->Add<ReLULayer<>>();

    // Update inputWidth and input Height.
    inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
    inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
    alexNet->Add(convolutionBlock);
    return;
  }

  /**
   * Returns Pooling Block.
   * 
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   */
  void PoolingBlock(const size_t kernelWidth,
                             const size_t kernelHeight,
                             const size_t strideWidth = 1,
                             const size_t strideHeight = 1)
  {
    Sequential<> *poolingBlock = new Sequential<>();
    poolingBlock->Add<MaxPooling<>>(kernelWidth, kernelHeight,
                                    strideWidth, strideHeight);
    // Update inputWidth and inputHeight.
    inputWidth = PoolOutSize(inputWidth, kernelWidth, strideWidth);
    inputHeight = PoolOutSize(inputHeight, kernelHeight, strideHeight);
    alexNet->Add(poolingBlock);
    return;
  }

  /**
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param pSideOne The size of the padding (width or height) on one side.
   * @param pSideTwo The size of the padding (width or height) on another side.
   * @return The convolution output size.
   */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t padding)
  {
    return std::floor(size + 2 * padding - k) / s + 1;
  }

  /*
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @return The convolution output size.
   */
  size_t PoolOutSize(const size_t size,
                     const size_t k,
                     const size_t s)
  {
    return std::floor(size - 1) / s + 1;
  }
  //! Locally stored AlexNet Model.
  Sequential<> *alexNet;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored include the final dense layer.
  bool includeTop;

  //! Locally stored type of pre-trained weights.
  std::string weights;

  //! Locally stored output shape of the AlexNet
  size_t outputShape;
};

AlexNet::AlexNet(const size_t inputChannel,
                 const size_t inputWidth,
                 const size_t inputHeight,
                 const size_t numClasses,
                 const bool includeTop,
                 const std::string &weights):
                 inputWidth(inputWidth),
                 inputHeight(inputHeight),
                 inputChannel(inputChannel),
                 numClasses(numClasses),
                 includeTop(includeTop),
                 weights(weights),
                 outputShape(512)
{
  alexNet = new Sequential<>();
}

AlexNet::AlexNet(const std::tuple<size_t, size_t, size_t> inputShape,
                 const size_t numClasses,
                 const bool includeTop,
                 const std::string &weights):
                 inputWidth(std::get<1>(inputShape)),
                 inputHeight(std::get<2>(inputShape)),
                 inputChannel(std::get<0>(inputShape)),
                 numClasses(numClasses),
                 includeTop(includeTop),
                 weights(weights),
                 outputShape(512)
{
  alexNet = new Sequential<>();
}

Sequential<>* AlexNet::CompileModel()
{
  // Add Convlution Block with inputChannels as input maps,
  // output maps = 64, kernel_size = (11, 11) stride = (4, 4)
  // and padding = (2, 2).
  ConvolutionBlock(inputChannel, 64, 11, 11, 4, 4, 2, 2);

  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  PoolingBlock(3, 3, 2, 2);

  // Add Convlution Block with inputChannels = 64,
  // output maps = 192, kernel_size = (5, 5) stride = (1, 1)
  // and padding = (2, 2).
  ConvolutionBlock(64, 192, 5, 5, 1, 1, 2, 2);

  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  PoolingBlock(3, 3, 2, 2);


  // Add Convlution Block with input maps = 192,
  // output maps = 384, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  ConvolutionBlock(192, 384, 3, 3, 1, 1, 1, 1);


  // Add Convlution Block with input maps = 384,
  // output maps = 256, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  ConvolutionBlock(384, 256, 3, 3, 1, 1, 1, 1);

  // Add Convlution Block with input maps = 256,
  // output maps = 256, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  ConvolutionBlock(256, 256, 3, 3, 1, 1, 1, 1);

  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  PoolingBlock(3, 3, 2, 2);

  if(includeTop)
  {
    AdaptivePoolingBlock(6, 6);
    alexNet->Add<Dropout<> >(0.2);
    alexNet->Add<Linear<> >(256 * 6 * 6, 4096);
    alexNet->Add<ReLULayer<> >();
    alexNet->Add<Dropout<> >(0.2);
    alexNet->Add<Linear<> >(4096, 4096);
    alexNet->Add<ReLULayer<> >();
    alexNet->Add<Linear<> >(4096, numClasses);
  }
  else
  {
    alexNet->Add<MaxPooling<> >(inputWidth, inputHeight, 1, 1, true);
    outputShape = 512;
  }

  return alexNet;
}

int main()
{
  AlexNet alexnet(1, 1, 28, 28);
  Sequential<>* model_layer = alexnet.CompileModel();
  // Dataset is randomly split into training
  // and validation parts with following ratio.
  constexpr double RATIO = 0.1;
  // The number of neurons in the first layer.
  constexpr int H1 = 100;
  // The number of neurons in the second layer.
  constexpr int H2 = 100;

  // The solution is done in several approaches (CYCLES), so each approach
  // uses previous results as starting point and have a different optimizer
  // options (here the step size is different).

  // Number of iteration per cycle.
  constexpr int ITERATIONS_PER_CYCLE = 10000;

  // Number of cycles.
  constexpr int CYCLES = 20;

  // Step size of an optimizer.
  constexpr double STEP_SIZE = 5e-4;

  // Number of data points in each iteration of SGD
  // Power of 2 is better for data parallelism
  constexpr int BATCH_SIZE = 64;
  cout << "Reading data ..." << endl;

  // Labeled dataset that contains data for training is loaded from CSV file,
  // rows represent features, columns represent data points.
  mat tempDataset;
  // The original file could be download from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("../../models/Kaggle/data/train.csv", tempDataset, true);
  // Originally on Kaggle dataset CSV file has header, so it's necessary to
  // get rid of the this row, in Armadillo representation it's the first column.
  mat dataset = tempDataset.submat(0, 1,
    tempDataset.n_rows - 1, tempDataset.n_cols - 1);

  // Splitting the dataset on training and validation parts.
  mat train, valid;
  data::Split(dataset, train, valid, RATIO);

  // Getting training and validating dataset with features only.
  const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
  const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);

  // According to NegativeLogLikelihood output layer of NN, labels should
  // specify class of a data point and be in the interval from 1 to
  // number of classes (in this case from 1 to 10).

  // Creating labels for training and validating dataset.
  const mat trainY = train.row(0) + 1;
  const mat validY = valid.row(0) + 1;

  FFN <NegativeLogLikelihood<>, RandomInitialization> model;
  cout << "Training ..." << endl;
  SGD <AdamUpdate> optimizer(
    // Step size of the optimizer.
    STEP_SIZE,
    // Batch size. Number of data points that are used in each iteration.
    BATCH_SIZE,
    // Max number of iterations
    ITERATIONS_PER_CYCLE,
    // Tolerance, used as a stopping condition. This small number
    // means we never stop by this condition and continue to optimize
    // up to reaching maximum of iterations.
    1e-8,
    // Shuffle. If optimizer should take random data points from the dataset at
    // each iteration.
    true,
    // Adam update policy.
    AdamUpdate(1e-8, 0.9, 0.999));
  // Cycles for monitoring the process of a solution.
  for (int i = 0; i <= CYCLES; i++)
  {

    // Train neural network. If this is the first iteration, weights are
    // random, using current values as starting point otherwise.
    model.Train(trainX, trainY, optimizer);

    // Don't reset optimizer's parameters between cycles.
    optimizer.ResetPolicy() = false;

    mat predOut;
    // Getting predictions on training data points.
    model.Predict(trainX, predOut);
    // Calculating accuracy on training data points.
    Row <size_t> predLabels = getLabels(predOut);
    double trainAccuracy = accuracy(predLabels, trainY);
    // Getting predictions on validating data points.
    model.Predict(validX, predOut);
    // Calculating accuracy on validating data points.
    predLabels = getLabels(predOut);
    double validAccuracy = accuracy(predLabels, validY);

    cout << i << " - accuracy: train = " << trainAccuracy << "%," <<
      " valid = " << validAccuracy << "%" << endl;
  }

  return 0;
}