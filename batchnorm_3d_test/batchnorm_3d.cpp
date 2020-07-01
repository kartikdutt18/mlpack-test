#include <iostream>
#include <bits/stdc++.h>
#include <armadillo>
#define ll long long
using namespace std;

arma::mat inputMean, mean, variance, runningMean, runningVar, normalized, Gamma, beta;
std::string lineBreak(16, '-');
std::string caseBreak(16, '=');

void fillInput(arma::mat& input, int caseNum = 0)
{
  switch(caseNum)
  {
    case 0:
      input.clear();
      input.set_size(8, 3);
      input << 1 << 446 << 42 << arma::endr
            << 2 << 16 << 63 << arma::endr
            << 3 << 13 << 63 << arma::endr
            << 4 << 21 << 21 << arma::endr
            << 1 << 13 << 11 << arma::endr
            << 32 << 45 << 42 << arma::endr
            << 22 << 16 << 63 << arma::endr
            << 32 << 13 << 42 << arma::endr;
      break;
    case 1:
      input.clear();
      input.set_size(12, 1);
      input << 12 << 443 << arma::endr
            << 134 << 45 << arma::endr
            << 11 << 13 << arma::endr
            << 14 << 55 << arma::endr
            << 110 << 4 << arma::endr
            << 1 << 45 << arma::endr;
      break;
    default : std::cout << "Error!";
              break;
  }
}

template<typename eT>
void Forward(arma::mat& input,
             arma::mat& output,
             size_t size,
             bool debug = true,
             bool deterministic = false,
             double eps = 1e-8,
             double momentum = 0.1)
{
  if (!deterministic)
  {
    size_t batchSize = input.n_cols;
    arma::cube inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(),
        input.n_rows / size, size, batchSize, false, false);

    output.set_size(arma::size(input));

    if (debug)
    {
      std::cout << "Reshaped Input (As a cube)\n";
      inputTemp.print();
      std::cout << lineBreak << std::endl;
    }

    arma::mat mean = arma::mean(arma::mean(inputTemp, 2), 0);
    if (debug)
    {
      std::cout << "Input mean : \n";
      mean.print();
      std::cout << lineBreak << std::endl;
    }

    arma::mat var = arma::mean(arma::mean(arma::pow(
        inputTemp.each_slice() - arma::repmat(mean,
        input.n_rows / size, 1), 2), 2), 0);

    if (debug)
    {
      std::cout << "Input Variance : \n";
      var.print();
      std::cout << lineBreak << std::endl;
    }

    arma::cube outputTemp(const_cast<arma::Mat<eT>&>(output).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);

    outputTemp = inputTemp;
    outputTemp.each_slice() -= arma::repmat(mean, input.n_rows / size, 1);

    inputMean.clear();
    inputMean.set_size(arma::size(input));

    arma::cube inputMeanTemp(const_cast<arma::Mat<eT>&>(inputMean).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);

    inputMeanTemp = outputTemp;

    outputTemp.each_slice() /= arma::sqrt(arma::repmat(var, input.n_rows / size, 1) + eps);

    normalized.clear();
    normalized.set_size(arma::size(input));
    arma::cube normalizedTemp(const_cast<arma::Mat<eT>&>(normalized).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);
    normalizedTemp = outputTemp;

    outputTemp.each_slice() %= arma::repmat(Gamma.t(), input.n_rows / size, 1);
    outputTemp.each_slice() += arma::repmat(beta.t(), input.n_rows / size, 1);

    // The final output.
    if (debug)
    {
      std::cout << "Training Output : \n";
      outputTemp.print();
      std::cout << lineBreak << std::endl;
    }

    double nElements = (double) input.n_elem / size;
    nElements = 1.0 * nElements / (1.0 * nElements - 1.0);
    runningMean = (1 - momentum) * runningMean + momentum * mean.t();
    runningVar = (1 - momentum) * runningVar + nElements * momentum * var.t();

    if (debug)
    {
      std::cout << "Training Running Mean : \n";
      runningMean.print();
      std::cout << lineBreak << std::endl;
      std::cout << "Training Running Variance : \n";
      runningVar.print();
    }
  }
  else
  {
    output.set_size(arma::size(input));
    output = input;
    arma::cube outputTemp(const_cast<arma::Mat<eT>&>(output).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);

    outputTemp.each_slice() -= arma::repmat(runningMean.t(),
        input.n_rows / size, 1);
    outputTemp.each_slice() /= arma::sqrt(arma::repmat(runningVar.t(),
        input.n_rows / size, 1) + eps);
    outputTemp.each_slice() %= arma::repmat(Gamma.t(),
        input.n_rows / size, 1);
    outputTemp.each_slice() += arma::repmat(beta.t(),
        input.n_rows / size, 1);

    if (debug)
    {
      std::cout << "Testing Output : \n";
      outputTemp.print();
      std::cout << lineBreak << std::endl;
    }
  }
}

int main()
{
  size_t numCases = 2;
  size_t size = 2;

  Gamma = arma::mat(size, 1, arma::fill::ones);
  beta = arma::mat(size, 1, arma::fill::zeros);

  runningMean = arma::mat(size, 1, arma::fill::zeros);
  runningVar = arma::mat(size, 1, arma::fill::zeros);

  for (size_t i = 0; i < numCases; i++)
  {
    arma::mat input, output;
    fillInput(input, i);
    std::cout << "TRAINING\n";
    std::cout << caseBreak << std::endl;
    Forward<double>(input, output, size);
    std::cout << caseBreak << std::endl;
    std::cout << "TESTING\n";
    std::cout << caseBreak << std::endl;
    Forward<double>(input, output, size, true, true);
    std::cout << caseBreak << std::endl;
  }
  return 0;
}