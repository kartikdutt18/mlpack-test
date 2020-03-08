#include<iostream>
#include<bits/stdc++.h>
#include<armadillo>
#define ll long long
#define ARMA_DONT_USE_WRAPPER
using namespace std;
using namespace arma;
template <
    typename InputDataType,
    typename OutputDataType
>
vector<double> CosineDistance(InputDataType c, InputDataType d)
{
 arma::colvec a=vectorise(c),b=vectorise(d);
 int n = c.n_cols;

 arma::colvec y = a % b;
 
 vector<double> ans;
 for(int i = 0; i < c.n_rows * c.n_slices; i+=n)
 {
   
   double t = arma::accu(y(span(i,i+n-1)));
   
   double z = 1 - t / sqrt(arma::accu(arma::pow(a(span(i, i + n -1)), 2)) * arma::accu(arma::pow(b(span(i, i + n-1)), 2)));
  ans.push_back(z);
 }
 
 return ans;
}
template <
    typename InputDataType,
    typename OutputDataType
>
vector<double> CosineEmbeddings(InputDataType c, InputDataType d, OutputDataType e, double margin = 0.0)
{
  arma::colvec a = vectorise(c), b = vectorise(d), y = vectorise(e);
  arma::colvec z = a % b;
  vector<double> ans;
  int n = c.n_cols;
  for (int i = 0; i < (z.n_elem / n); i += n)
  {
    //cout<<i<<endl;
    if(y(i/n) == 1)
    {
      double t = arma::accu(y(span(i, i + n -1)));

      double z = 1 - t / sqrt(arma::accu(arma::pow(a(span(i, i + n -1)), 2)) * arma::accu(arma::pow(b(span(i, i + n - 1)), 2)));
      ans.push_back(1 - z);
    }
    else
    {
      double t = arma::accu(y(span(i, i + n -1)));

      double z = 1 - t / sqrt(arma::accu(arma::pow(a(span(i, i + n -1)), 2)) * arma::accu(arma::pow(b(span(i, i + n -1)), 2)));
      z=(z-margin)>0?z:0;
      ans.push_back(z);
    }
    
  }
  return ans;
}
void check()
{
  arma::colvec input1(3 * 2);
  arma::colvec input2(3 * 2);
  arma::colvec y(3 * 1);
  y.ones();
  y = y * -1;
  input1 = arma::colvec(3 * 2);
  input2 = arma::colvec(3 * 2);
  input1.fill(1);
  input1(4) = 2;
  input2.fill(1);
  input2(0) = 2;
  input2(1) = 2;
  input2(2) = 2;
  
}
int main()
{
  
  
  return 0;
}