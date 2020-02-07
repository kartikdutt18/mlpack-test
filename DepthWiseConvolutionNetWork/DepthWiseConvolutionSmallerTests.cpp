#include<iostream>
#include<bits/stdc++.h>
#include<armadillo>
//#include "../../mlpack/src/mlpack/methods/ann/layer/separable_convolution.hpp"

#define ll long long
using namespace std;

void CheckReset(size_t inSize, size_t outSize, size_t kernelWidth, size_t kernelHeight, size_t numGroups = 1)
{
    arma::mat weights;
    weights.set_size((inSize * outSize * kernelWidth * kernelHeight) / numGroups
                      + outSize,1);
    weights.zeros();

    weights(1) = 1;
    weights(2) = 1;
    weights(5) = 2;
    weights(7) = 2;
    weights(8) = 3;
    weights(14) = 4;

    weights.print();
    cout<<"-------------------------"<<endl;
    cout << "-------------------------" << endl;
    cout << "-------------------------" << endl;
    arma::cube weight = arma::cube(weights.memptr(), kernelWidth, kernelHeight,
                        outSize * inSize, false, false);
    weight.print();
    cout << "-------------------------" << endl;
    cout << "-------------------------" << endl;
    cout << "-------------------------" << endl;
    arma::mat bias = arma::mat(weights.memptr() + weight.n_elem,
                     outSize, 1, false, false);
    bias.print();
    cout << "-------------------------" << endl;
    cout << "-------------------------" << endl;
    cout << "-------------------------" << endl;
}

void CheckInput()
{
    arma::mat input = arma::linspace<arma::colvec>(0, 8, 9);
    input.zeros();
    input(0) = 1.0;
    input(3) = 3.0;
    input(4) = 4.0;
    input(5) = 5.0;
    input(7) = 1.0;

    input.print();
    cout << "-------------------------" << endl;
    cout << "-------------------------" << endl;
    cout << "-------------------------" << endl;
}

void checkOutPutForwardPass()
{
    arma::mat output, input, delta;
    SeparableConvolution<> module1(1, 4, 1, 1, 1, 1, 0, 0, 3, 3, 1, "valid");

    // Test the forward function.
    input = arma::linspace<arma::colvec>(0, 8, 9);
    input.zeros();
    input(0) = 1.0;
    input(3) = 3.0;
    input(4) = 4.0;
    input(5) = 5.0;
    input(7) = 1.0;
    module1.Parameters() = arma::mat(20, 1, arma::fill::zeros);
    module1.Parameters()(1) = 1.0;
    module1.Parameters()(2) = 1.0;
    module1.Parameters()(5) = 2.0;
    module1.Parameters()(7) = 2.0;
    module1.Parameters()(8) = 3.0;
    module1.Parameters()(14) = 4.0;
    module1.Reset();
    module1.Forward(std::move(input), std::move(output));
}
int main() {
    CheckInput();
    CheckReset(1, 4, 2, 2, 1);
    return 0;
}