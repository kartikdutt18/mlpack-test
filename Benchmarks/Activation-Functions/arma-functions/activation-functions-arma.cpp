#include<armadillo>
#include<bits/stdc++.h>

template <typename InputType>
struct HardSigmoidFnArma
{
  void operator()(InputType x)
  {
    InputType y;
    y.set_size(arma::size(x));
    y = ((0.2 * x + 0.5) > 0) % (0.2 * x + 0.5);
    y = (y < 1) % (0.2 * x + 0.5) + (y >= 1);
  }
};

template <typename InputType>
struct LogisticFnArma {
  void operator()(InputType x)
  {
    InputType y;
    y = (1.0 / (1 + arma::exp(-x)));
  }
};

template <typename InputType>
struct MishFnArma
{
  void operator()(InputType x)
  {
    InputType y;
    y = x % (arma::exp(2 * x) + 2 * arma::exp(x)) /
        (2 + 2 * arma::exp(x) + arma::exp(2 * x));
  }
};

template <typename InputType>
struct SoftPlusFnArma
{
  void operator()(InputType x)
  {
    InputType y;
    y = (x > DBL_MAX) + (x < DBL_MAX) % (x > -DBL_MAX) %
                            arma::log(1 + arma::exp(x));
  }
};

template <typename InputType>
struct SoftSignFnArma
{
  void operator()(InputType x)
  {
    InputType y;
    y = (x > DBL_MAX) + (x < DBL_MAX) % (x > -DBL_MAX) %
                            arma::log(1 + arma::abs(x));
  }
};

template <typename InputType>
struct SwishFnArma
{
  void operator()(InputType x)
  {
    InputType y;
    y = x / (1.0 + arma::exp(-x));
  }
};