#include <iostream>
#include <bits/stdc++.h>
#include <armadillo>
#define ll long long
using namespace std;

void fillInput(arma::mat& input)
{
  input << 1 << 446 << 42 << arma::endr
      << 2 << 16 << 63 << arma::endr
      << 3 << 13 << 63 << arma::endr
      << 4 << 21 << 21 << arma::endr
      << 1 << 13 << 11 << arma::endr
      << 32 << 45 << 42 << arma::endr
      << 22 << 16 << 63 << arma::endr
      << 32 << 13 << 42 << arma::endr;
}

void Backward(arma::mat input,
              arma::mat output,
              arma::mat inputMean,
              arma::mat gamma,
              arma::mat variance,
              const size_t size = 2,
              const double eps = 1e-8)
{

  size_t inputSize = input.n_rows / size;
  const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);
  arma::mat delta;
  delta.set_size(arma::size(input));
  for (size_t channelIdx = 0; channelIdx <= size - 1; channelIdx++)
  {
    // Step 1: dl / dxhat.
    arma::mat norm = output(arma::span(channelIdx * inputSize,
          (channelIdx + 1) * inputSize - 1), arma::span()) * gamma(channelIdx);

    const arma::mat var = arma::sum(norm % inputMean(arma::span(channelIdx * inputSize,
          (channelIdx + 1) * inputSize - 1), arma::span()), 1) *
          std::pow(stdInv(channelIdx), 3) * -0.5;
    delta(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
        arma::span()) = norm * stdInv(channelIdx) +
        inputMean(arma::span(channelIdx * inputSize,
        (channelIdx + 1) * inputSize - 1), arma::span()).each_col() %
        var * 2 / input.n_cols;
    delta(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
        arma::span()).each_col() += arma::sum(norm * stdInv(channelIdx), 1);
  }

  std::cout << "TRAIN_BACKWARD " << std::endl;
  delta.print();
}

void Forward()
{
  arma::mat input(8, 3);
  fillInput(input);
  input.print();
  std::cout << "-----\n";

  size_t size = 2;
  size_t inputSize = input.n_rows / size;
  arma::mat output = input;
  arma::mat Mean(1, size);
  arma::mat Var(1, size);
  arma::mat runningMean(size, 1, arma::fill::zeros);
  arma::mat runningVar(size, 1, arma::fill::zeros);
  double momentum = 0.1;

  arma::mat inputMean;
  inputMean.set_size(arma::size(input));
  double n = 1.0 * input.n_elem / size;
  n = 1.0 * n / (1.0 * n - 1.0);
  arma::mat gamma(size, 1);
  gamma.fill(1);

  for (size_t channelIdx = 0; channelIdx <= size - 1 ; channelIdx++)
  {
    std::cout << "channel" << channelIdx << endl;
    arma::mat temp = arma::mean(input(arma::span(channelIdx * inputSize,
          (channelIdx + 1) * inputSize - 1), arma::span()), 1);
    // arma::mean(temp, 0).print();
    temp = arma::mean(temp, 0);
    Mean(channelIdx) = temp(0);
    std::cout << "-----\n";

    arma::mat t2 = input(arma::span(channelIdx * inputSize,
          (channelIdx + 1) * inputSize - 1), arma::span());
    t2 = t2 - temp(0);
    t2 = arma::pow(t2, 2);
    // arma::mean(arma::mean(t2, 1), 0).print();
    t2 = arma::mean(arma::mean(t2, 1), 0);
    Var(channelIdx) = t2(0);

    //Var(channelIdx) *= (1.0 * n / (1.0 * (n - 1)));
    output(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
          arma::span()) -= temp(0);

    inputMean(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
          arma::span()) = output(arma::span(channelIdx * inputSize,
          (channelIdx + 1) * inputSize - 1), arma::span());

    output(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
          arma::span()) /= std::sqrt(Var(channelIdx) + 1e-5);
    runningMean(channelIdx) = momentum * Mean(channelIdx) +
            (1 - momentum) * runningMean(channelIdx);
    runningVar(channelIdx) = (momentum * Var(channelIdx) * n) +
            (1 - momentum) * runningVar(channelIdx);
  }

  std::cout << "Running Mean \n";
  runningMean.print();
  std::cout << "Running Var \n";
  runningVar.print();
  std::cout << "Mean \n";
  Mean.print();
  std::cout << "Var \n";
  Var.print();
  std::cout << "Output \n";
  output.print();
  Backward(input, output, inputMean, gamma, Var);
  std::cout << "CASE ENDS\n---------\n";

  input.clear();
  output.clear();
  input = arma::mat(12, 1);
  input << 12 << 443 << arma::endr
      << 134 << 45 << arma::endr
      << 11 << 13 << arma::endr
      << 14 << 55 << arma::endr
      << 110 << 4 << arma::endr
      << 1 << 45 << arma::endr;
  output = input;
  inputSize = input.n_rows / size;
  inputMean.set_size(arma::size(input));

  n = 1.0 * input.n_elem / size;
  n = 1.0 * n / (1.0 * n - 1.0);
  for (size_t channelIdx = 0; channelIdx <= size - 1; channelIdx++)
  {
    std::cout << "channel" << channelIdx << endl;
    arma::mat temp = arma::mean(input(arma::span(channelIdx * inputSize,
        (channelIdx + 1) * inputSize - 1),
        arma::span()), 1);
    // arma::mean(temp, 0).print();
    temp = arma::mean(temp, 0);
    Mean(channelIdx) = temp(0);
    std::cout << "-----\n";

    arma::mat t2 = input(arma::span(channelIdx * inputSize,
        (channelIdx + 1) * inputSize - 1), arma::span());
    t2 = t2 - temp(0);
    t2 = arma::pow(t2, 2);
    // arma::mean(arma::mean(t2, 1), 0).print();
    t2 = arma::mean(arma::mean(t2, 1), 0);
    Var(channelIdx) = t2(0);
    //Var(channelIdx) *= (1.0 * n / (1.0 * (n - 1)));
    output(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
        arma::span()) -= temp(0);
    output(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
        arma::span()) /= std::sqrt(Var(channelIdx) + 1e-5);
    runningMean(channelIdx) = momentum * Mean(channelIdx) +
      (1 - momentum) * runningMean(channelIdx);
    runningVar(channelIdx) = (momentum * Var(channelIdx) * n) +
      (1 - momentum) * runningVar(channelIdx);
  }

  std::cout << "Running Mean \n";
  runningMean.print();
  std::cout << "Running Var \n";
  runningVar.print();
  std::cout << "Mean \n";
  Mean.print();
  std::cout << "Var \n";
  Var.print();
  std::cout << "Output \n";
  output.print();
  Backward(input, output, inputMean, gamma, Var);
  std::cout << "CASE ENDS\n---------\n";
  output = input;
  for (size_t channelIdx = 0; channelIdx <= size - 1; channelIdx++)
  {
    output(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
        arma::span()) -= runningMean(channelIdx);
    output(arma::span(channelIdx * inputSize, (channelIdx + 1) * inputSize - 1),
        arma::span()) /= std::sqrt(runningVar(channelIdx) + 1e-5);
  }
  std::cout << "CASE ENDS\n---------\n";
  std::cout << "Running Mean \n";
  runningMean.print();
  std::cout << "Running Var \n";
  runningVar.print();
  std::cout << "Mean \n";
  Mean.print();
  std::cout << "Var \n";
  Var.print();
  std::cout << "Output \n";
  output.print();
}

int main()
{
  Forward();
  return 0;
}