#include<bits/stdc++.h>
#include <chrono>
#include<armadillo>
using namespace std;

void LogisticTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = (1.0 / (1 + arma::exp(-x)));
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current logistic implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = 1.0 / (1 + exp(x(j)));
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken2 *= 1e-9;
  cout << "Time taken by new logistic implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void LogisticDerivativeTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = x % (1.0 - x);
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current logistic derivative implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = x(j) * (1.0 - x(j));
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken2 *= 1e-9;
  cout << "Time taken by new logistic derivative implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void MishTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = x % (arma::exp(2 * x) + 2 * arma::exp(x)) /
                          (2 + 2 * arma::exp(x) + arma::exp(2 * x));
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current mish implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = x(j) * (std::exp(2 * x(j)) + 2 * std::exp(x(j))) /
                  (2 + 2 * std::exp(x(j)) + std::exp(2 * x(j)));
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken2 *= 1e-9;
  cout << "Time taken by new mish implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void MishDerivativeTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = arma::exp(x) % (4 * (x + 1) + arma::exp(x) % (4 * x + 6) + 4 * arma::exp(2 * x) + arma::exp(3 * x)) /
                          arma::pow(arma::exp(2 * x) + 2 * arma::exp(x) + 2, 2);
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current mish derivative implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = std::exp(x(j)) * (4 * (x(j) + 1) + std::exp(x(j)) * (4 * x(j) + 6) + 4 * std::exp(2 * x(j)) + std::exp(3 * x(j))) /
                  std::pow(std::exp(2 * x(j)) + 2 * std::exp(x(j)) + 2, 2);
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken2 *= 1e-9;
  cout << "Time taken by new mish derivative implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void SoftPlusDerivativeTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = 1.0 / (1 + arma::exp(-x));
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current soft plus derivative implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = 1.0 / (1 + std::exp(-x(j)));
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken2 *= 1e-9;
  cout << "Time taken by new soft plus derivative implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void SoftSignDerivativeTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = arma::pow(1.0 - arma::abs(x), 2);
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current soft sign derivative implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = std::pow(1.0 - std::abs(x(j)), 2);
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken2 *= 1e-9;
  cout << "Time taken by new soft sign derivative implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void SwishTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = x / (1 + arma::exp(-x)) + (1 - x / (1 + arma::exp(-x))) /
                                                            (1 + arma::exp(-x));
    ;
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current swish derivative implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = x(j) / (1 + std::exp(-x(j))) + (1 - x(j) / (1 + std::exp(-x(j)))) /
                                                     (1 + std::exp(-x(j)));
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken2 *= 1e-9;
  cout << "Time taken by new swish derivative implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

int main()
{
  LogisticTimeDifference();
  LogisticDerivativeTimeDifference();
  MishTimeDifference();
  MishDerivativeTimeDifference();
  SoftPlusDerivativeTimeDifference();
  SwishTimeDifference();
  return 0;
}