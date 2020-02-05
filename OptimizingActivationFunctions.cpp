#include<bits/stdc++.h>
#include <chrono>
#include<armadillo>
using namespace std;

void SwishTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = x / (1 + arma::exp(-1*x));
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by new swish implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = x(j) / (1.0 + exp(-x(j)));
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current swish implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void SoftPlusTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 DBL_MAX -DBL_MAX 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = (x > DBL_MAX) + (x<DBL_MAX) % (x>-DBL_MAX) % (arma::log(1+arma::exp(x)));
  
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by new soft_plus implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = x(j) > -DBL_MAX ? std::log(1 + std::exp(x(j))) : 0;
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current soft_plus implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void HardSigmoidTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = arma::min(arma::ones(x.n_elem), arma::max(arma::zeros(x.n_elem), (x * 0.2 + 0.5)));
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by new hard_sigmoid implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = min(1.0, max(0.0, 0.2 * x(j) + 0.5));
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current hard_sigmoid implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}

void SoftSignTimeDifference()
{
  const arma::colvec x("-1 1 1 -1 1 -1 1 0");
  auto start1 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output = (x > DBL_MAX) + (x < DBL_MAX) % (x > -DBL_MAX) % (arma::log(1 + arma::abs(x)));
  }
  auto end1 = chrono::high_resolution_clock::now();
  double time_taken1 = chrono::duration_cast<chrono::nanoseconds>(end1 - start1).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by new soft_sign implementation is : "
       << time_taken1 << setprecision(5)
       << " sec " << endl;

  auto start2 = chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000000; i++)
  {
    arma::colvec output(x.n_elem);
    for (size_t j = 0; j < x.n_elem; j++)
    {
      output(j) = x(j) > -DBL_MAX ? x(j) / (1.0 + abs(x(j))) : -1.0;
    }
  }
  auto end2 = chrono::high_resolution_clock::now();
  double time_taken2 = chrono::duration_cast<chrono::nanoseconds>(end2 - start2).count();
  time_taken1 *= 1e-9;
  cout << "Time taken by current soft_sign implementation is : "
       << time_taken2 << setprecision(5)
       << " sec " << endl;
  cout << "---------------------------" << endl;
}
int main()
{
  SwishTimeDifference();
  HardSigmoidTimeDifference();
  SoftPlusTimeDifference();
  SoftSignTimeDifference();
  return 0;
}