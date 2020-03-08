#include<iostream>
#include<bits/stdc++.h>
#include<armadillo>
#define ll long long
#define ARMA_DONT_USE_WRAPPER
using namespace std;



int main() 
{
  arma::mat input1,input2;
  input1 = arma::mat(3, 2);
  input2 = arma::mat(3, 2);
  input1.fill(1);
  input1(4) = 2;
  input2.fill(1);
  input2(0) = 2;
  input2(1) = 2;
  input2(2) = 2;
  arma::colvec x1=arma::vectorise(input1);
  arma::colvec x2 = arma::vectorise(input2);
  x1(arma::span(0,input1.n_cols)).print();
  return 0; 
}