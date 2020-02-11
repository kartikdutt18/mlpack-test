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
  input.zeros();
  input(0) = 1;
  input(1) = 2;
  input(2) = 3;
  input(3) = input(8) = 7;
  input(4) = 4;
  input(5) = 5;
  input(6) = input(7) = 6;
  input(10) = 8;
  input(11) = 9;
  MeanPooling<> layer1(2, 2, 2, 1);
  layer1.InputHeight() = 3;
  layer1.InputWidth() = 4;
  layer1.Forward(std::move(input), std::move(output));
  input.print();
  cout << "-------" << endl;
  output.print();
}