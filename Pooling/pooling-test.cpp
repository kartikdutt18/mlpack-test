#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
using namespace mlpack;
using namespace mlpack::ann;
using namespace std;

int main()
{
  arma::mat input = arma::mat(12, 1);
  arma::mat output;
  input = arma::mat(6, 1);
  input.zeros();
  input(0) = 1;
  input(1) = 1;
  input(3) = 1;
  MaxPooling<> layer1(2, 1, 1, 1, true);
  layer1.InputHeight() = 2;
  layer1.InputWidth() = 3;
  layer1.Forward(std::move(input), std::move(output));
  input.print();
  cout << "-------" << endl;
  output.print();
}