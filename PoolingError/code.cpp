#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/core/data/split_data.hpp>
using namespace mlpack;
using namespace mlpack::ann;
using namespace std;

arma::Row<size_t> getLabels(const arma::mat& predOut)
{
  arma::Row<size_t> pred(predOut.n_cols);

  // Class of a j-th data point is chosen to be the one with maximum value
  // in j-th column plus 1 (since column's elements are numbered from 0).
  for (size_t j = 0; j < predOut.n_cols; ++j)
  { 
    pred(j) = arma::as_scalar(arma::index_max(predOut.col(j))) + 1;
  }

  return pred;
}

double get_accuracy(arma::Row<size_t> predLabels, const arma::mat& realY)
{
  // Calculating how many predicted classes are coincide with real labels.
  size_t success = 0;
  for (size_t j = 0; j < realY.n_cols; j++) {
    if (predLabels(j) == std::round(realY(j))) {
      ++success;
    }
  }

  // Calculating percentage of correctly classified data points.
  return (double)success / (double)realY.n_cols * 100.0;
}

void save(const std::string filename, std::string header,
  const arma::Row<size_t>& predLabels)
{
  ofstream out(filename);
  out << header << endl;
  for (size_t j = 0; j < predLabels.n_cols; ++j)
  {
    out << j + 1 << "," << std::round(predLabels(j)) - 1;
    if (j < predLabels.n_cols - 1)
    {
      out << std::endl;
    }
  }
  out.close();
}


int main(){

	constexpr double RATIO = 0.1;
	constexpr int ITERATIONS_PER_CYCLE = 24;
	constexpr int CYCLES = 40;
	constexpr double STEP_SIZE = 1.2e-3;
	constexpr int BATCH_SIZE = 50;
	
	cout << "Data preparation..." << endl;
	arma::mat dataset;
	data::Load("train.csv", dataset, true);
	dataset = dataset.submat(0, 1, dataset.n_rows-1, dataset.n_cols-1);
	arma::mat train, valid;
	data::Split(dataset, train, valid, RATIO);
	const arma::mat trainX = train.submat(1, 0, train.n_rows-1, train.n_cols-1);
	const arma::mat validX = valid.submat(1, 0, valid.n_rows-1, valid.n_cols-1);
	const arma::mat trainY = train.submat(0, 0, 0, train.n_cols-1) + 1;
	const arma::mat validY = valid.submat(0, 0, 0, valid.n_cols-1) + 1;

	FFN<NegativeLogLikelihood<>, RandomInitialization> model;
	model.Add<Convolution<> >(
		1,
		6,
		5,
		5,
		1,
		1,
		0,
		0,
		28,
		28
		);
	model.Add<LeakyReLU<> >();
	model.Add<MaxPooling<> >(
		2, 2, 2, 2, true
		);
	model.Add<Convolution<> >(
		6,
		16,
		5,
		5,
		1,
		1,
		0,
		0,
		12,
		12
		);
	model.Add<LeakyReLU<> >();
	model.Add<MaxPooling<> >(2, 2, 2, 2, true);
	model.Add<Linear<> >(16*4*4, 10);
	model.Add<LogSoftMax<> >();

	ens::SGD<ens::AdamUpdate> optimizer(
		STEP_SIZE,
		BATCH_SIZE,
		ITERATIONS_PER_CYCLE,
		1e-8,
		true,
		ens::AdamUpdate(1e-8, 0.9, 0.999)
		);

	cout << "Training..." << endl;

	for(int i = 0; i <= CYCLES; i++){
		model.Train(trainX, trainY, optimizer);
		arma::mat output;
		model.Predict(trainX, output);
		cout << "Tgfvd" << endl;
		arma::Row<size_t> predictions = getLabels(output);
		double acc = get_accuracy(predictions, trainY);
		model.Predict(validX, output);
		predictions = getLabels(output);
		// for(int j = 0; j < predictions.n_cols-1; j++){
		// 	cout << predictions[j] << " " << validY[j] << endl;
		// }
		double validAcc = get_accuracy(predictions, validY);

		cout << "Training acc: " << acc << "\n" << "Validation acc: " 
			<< validAcc << endl;
	}

	cout << "Testing.." << endl;
	data::Load("test.csv", dataset);
	arma::mat testX = dataset.submat(0, 1, dataset.n_rows-1, dataset.n_cols-1);
	arma::mat output;
	model.Predict(testX, output);
	arma::Row<size_t> predictions = getLabels(output);

	cout << "Saving results to results.csv" << endl;
	save("results.csv", "ID, label", predictions);
}