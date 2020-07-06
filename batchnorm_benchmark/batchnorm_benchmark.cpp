#include <iostream>
#include <bits/stdc++.h>
#include <armadillo>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <chrono>
#include <ensmallen.hpp>
using namespace std;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

struct Results
{
  double LinearBatchNormTime;
  double MiniBatchNormTime;
  size_t inputCols;
};

std::vector<Results> results;



void BatchNormTrainTimeDifference(arma::mat& input, size_t inSize)
{
  FFN<> module1;
  module1.Add<IdentityLayer<>>();
  module1.Add<BatchNorm<>>(inSize, 1e-8, false);
  module1.Add<LogSoftMax<>>();
  arma::mat output(arma::size(input));
  SGD<AdamUpdate> optimizer1(
      0.1,
      5,
      0,
      1e-8,
      true,
      AdamUpdate(1e-8, 0.9, 0.999));

  output.ones();
  // Mini Batch Norm.
  auto start1 = chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 1000; i++)
  {
    module1.Train(input, output, optimizer1);
  }

  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 -
      start1).count();
  time_taken1 *= 1e-9;

  // Linear Batch Norm.
  FFN<> module2;
  module2.Add<IdentityLayer<>>();
  module2.Add<BatchNorm<>>(input.n_rows, 1e-8, false);
  module2.Add<LogSoftMax<>>();
  SGD<AdamUpdate> optimizer2(
      0.1,
      5,
      10,
      1e-8,
      true,
      AdamUpdate(1e-8, 0.9, 0.999));
  auto start2 = chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 1000; i++)
  {
    module2.Train(input, output, optimizer2);
    //module2.Forward(input, output);
  }

  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 -
      start2).count();
  time_taken2 *= 1e-9;
  results.push_back(Results{time_taken1, time_taken2, (size_t)input.n_cols});
}

int main()
{
  results.clear();
  for (size_t numCols = 10; numCols <= 10000; numCols*=10)
  {
    arma::mat input(32 * 32 * 32, numCols);
    input.zeros();
    BatchNormTrainTimeDifference(input, 32);
    std::cout << results.back().inputCols << "," <<
        results.back().MiniBatchNormTime << "," <<
        results.back().LinearBatchNormTime << std::endl;
  }

  return 0;
}