#include<iostream>
#include<bits/stdc++.h>
#include <armadillo>
#define ll long long

using namespace std;


void Check(arma::mat fieldVec)
{
    fieldVec.set_size(1, 1);
    fieldVec.insert_cols(0, arma::vec(5).fill(1));
}

void Check(arma::field<arma::vec> fieldVec)
{
  fieldVec(0, 0) = arma::vec(3);
}

void span5(arma::vec& a)
{
  for(size_t i = 0; i + 5<= a.n_rows; i+=5)
  {
    a.subvec(i, i + 4).print();
  }
  cout << endl;
}

int main()
{
  std::deque<arma::vec> tempVector;
  
  arma::vec a(5);
  a.fill(1);
  arma::vec b(5);
  b.fill(3);
  tempVector.push_front(a);
  tempVector.push_front(b);

  arma::vec predictions(5);
  predictions.fill(0);
  arma::vec boundingBoxes;
  boundingBoxes.insert_rows(0, predictions);
  predictions.fill(1);
  boundingBoxes.insert_rows(boundingBoxes.n_rows , predictions);
  tempVector.push_back(boundingBoxes);
  /*
  arma::field<arma::vec> fieldVec(1, tempVector.size());
  Check(fieldVec);
  Check(arma::mat());
  arma::mat matType(5, tempVector.size());
  cout << typeid(fieldVec).name() << endl;
  for (size_t i = 0; i < tempVector.size(); i++)
  {
    fieldVec(0, i) = tempVector[i];
    matType.col(i) = tempVector[i];
  }

  fieldVec.print();
  */
  for(size_t i = 0; i < tempVector.size();i++)
  {
    span5(tempVector[i]);
  }

  return 0;
}