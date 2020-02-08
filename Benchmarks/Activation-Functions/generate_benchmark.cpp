#include<bits/stdc++.h>
#include<chrono>
#include<armadillo>
#include "./arma-functions/activation-functions-arma.cpp"
#include "./simple-functions/activation-functions-simple.cpp"
using namespace std;

std::__1::chrono::steady_clock::time_point getTime()
{
  // returns Current Time.
  return chrono::high_resolution_clock::now();
}

template<typename CTime>
double getDt(CTime start, CTime end)
{
  // returns time difference.
  double time_taken = chrono::duration_cast<chrono::nanoseconds>(end -
      start).count();
  time_taken *= 1e-9;
  return time_taken;
}

template <typename InputType, typename Function>
void GenerateSingleBenchmark(Function F, InputType x, int insize, string inputType,
      string FunctionName, string FunctionType)
{
  // Generates benchmarks by passing given input 10k times to the function.
  auto start = getTime();
  for(int i = 0; i < 100000; i++)
  {
    F(x);
  }
  auto end = getTime();

  double timeTaken = getDt(start, end);

  cout<<FunctionName<<","<<FunctionType<<","<<insize<<","<<inputType<<","<<timeTaken<<endl;
}

template<typename InputType>
void GetBenchmarks(InputType x, int insize, string inputType)
{
  // Generates BenchMarks for all functions.
  GenerateSingleBenchmark(HardSigmoidFnArma<InputType>(),x, insize, inputType, "HardSigmoid", "Armadillo");
  GenerateSingleBenchmark(HardSigmoidFnSimple<InputType>(), x, insize, inputType, "HardSigmoid", "Simple");

  GenerateSingleBenchmark(LogisticFnArma<InputType>(), x, insize, inputType, "Logistic", "Armadillo");
  GenerateSingleBenchmark(LogisticFnSimple<InputType>(), x, insize, inputType, "Logistic", "Simple");

  GenerateSingleBenchmark(MishFnArma<InputType>(), x, insize, inputType, "Mish", "Armadillo");
  GenerateSingleBenchmark(MishFnSimple<InputType>(), x, insize, inputType, "Mish", "Simple");

  GenerateSingleBenchmark(SoftPlusFnArma<InputType>(), x, insize, inputType, "SoftPlus", "Armadillo");
  GenerateSingleBenchmark(SoftPlusFnSimple<InputType>(), x, insize, inputType, "SoftPlus", "Simple");

  GenerateSingleBenchmark(SoftSignFnArma<InputType>(), x, insize, inputType, "SoftSign", "Armadillo");
  GenerateSingleBenchmark(SoftSignFnSimple<InputType>(), x, insize, inputType, "SoftSign", "Simple");

  GenerateSingleBenchmark(SwishFnArma<InputType>(), x, insize, inputType, "Swish", "Armadillo");
  GenerateSingleBenchmark(SwishFnSimple<InputType>(), x, insize, inputType, "Swish", "Simple");
}

void GenerateBenchmarks()
{
  arma::colvec x;
  for(int i = 1; i <= 1000; i*=10)
  {
    x.randn(i);
    GetBenchmarks(x, i, "Vector");
  }
  arma::mat x2;
  for (int i = 1; i <= 1000; i *= 10)
  {
    x2.randn(i,i / 10 + 1);
    GetBenchmarks(x2, i, "Matrix");
  }
}

int main()
{
  GenerateBenchmarks();
  return 0;
}