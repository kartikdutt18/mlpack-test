#include <armadillo>
#include <bits/stdc++.h>

template <typename InputType>
struct HardSigmoidFnSimple
{
  void operator()(InputType x)
  {
    InputType y;
    y.set_size(x.n_elem);
    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = std::min(1.0, std::max(0.0, 0.2 * x(i) + 0.5));
    }
  }
};

template <typename InputType>
struct LogisticFnSimple
{
  void operator()(InputType x)
  {
    InputType y;
    y.set_size(x.n_elem);
    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = std::min(1.0, std::max(0.0, 0.2 * x(i) + 0.5));
    }
  }
};

template <typename InputType>
struct MishFnSimple
{
  void operator()(InputType x)
  {
    InputType y;
    y.set_size(x.n_elem);
    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = x(i) * (std::exp(2 * x(i)) + 2 * std::exp(x(i))) /
            (2 + 2 * std::exp(x(i)) + std::exp(2 * x(i)));
    }
  }
};
template <typename InputType>
struct SoftPlusFnSimple
{
  void operator()(InputType x)
  {
    InputType y;
    y.set_size(x.n_elem);
    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = x(i) > DBL_MAX ? 1 : std::log(1 + std::exp(x(i)));
    }
  }
};
template <typename InputType>
struct SoftSignFnSimple
{
  void operator()(InputType x)
  {
    InputType y;
    y.set_size(x.n_elem);
    for (size_t i = 0; i < x.n_elem; i++)
    {
      y(i) = x(i) > DBL_MAX ? 1 : std::log(1 + std::abs(x(i)));
    }
  }
};
  
template <typename InputType>
struct SwishFnSimple{
  void operator()(InputType x)
    {
      InputType y;
      y.set_size(x.n_elem);
      for (size_t i = 0; i < x.n_elem; i++)
      {
        y(i) = x(i) / (1.0 + std::exp(-x(i)));
      }
    }
};