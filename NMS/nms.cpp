#include <armadillo>
using namespace std;

int main()
{
  arma::mat boundingBoxes;
  arma::vec bbox1(4), bbox2(4), bbox3(4);
  // Set values of each bounding box.
  bbox1 << 0.0 << 0.0 << 41.0 << 31.0;
  bbox2 << 1.0 << 1.0 << 42.0 << 22.0;
  bbox3 << 10.0 << 13.0 << 90.0 << 100.0;

  // Fill bounding box.
  boundingBoxes.insert_cols(0, bbox3);
  boundingBoxes.insert_cols(0, bbox2);
  boundingBoxes.insert_cols(0, bbox1);

  cout << "bBoxes" << endl;
  boundingBoxes.print();
  arma::mat selectedBoundingBoxes;
  arma::vec confidenceScores(3);
  confidenceScores << 0.7 << 0.6 << 0.4;

  arma::ucolvec sortedIndices = arma::sort_index(confidenceScores);
  arma::ucolvec selectedIndices;
  selectedIndices.clear();
  // Pre-Compute area of each bounding box.
  arma::mat area;
  if (false)
  {
    area = (boundingBoxes.row(2) - boundingBoxes.row(0)) %
        (boundingBoxes.row(3) - boundingBoxes.row(1));
  }
  else
  {
    area = (boundingBoxes.row(2)) % (boundingBoxes.row(3));
  }


  while (sortedIndices.n_elem > 0)
  {
    
    size_t selectedIndex = sortedIndices(sortedIndices.n_elem - 1);

    // Choose the box with the largest probability.
    selectedIndices.insert_rows(0, arma::uvec(1).fill(selectedIndex));

    // Check if there are other bounding boxes to compare with.
    if (sortedIndices.n_elem == 1)
    {
      break;
    }

    // Remove the last index.
    sortedIndices = sortedIndices(arma::span(0, sortedIndices.n_rows - 2),
        arma::span());

    // Calculate IoU of remaining boxes with the last bounding box with
    // the highest confidence score.
    arma::mat intersectionArea;
    
    intersectionArea = arma::clamp(arma::clamp(
          boundingBoxes.submat(arma::uvec(1).fill(2), sortedIndices), INT_MIN,
          boundingBoxes(2, selectedIndex)) - arma::clamp(
          boundingBoxes.submat(arma::uvec(1).fill(0), sortedIndices),
          boundingBoxes(0, selectedIndex), INT_MAX), 0.0, INT_MAX) %
          arma::clamp(arma::clamp(boundingBoxes.submat(arma::uvec(1).fill(3),
          sortedIndices), INT_MIN, boundingBoxes(3, selectedIndex)) -
          arma::clamp(boundingBoxes.submat(arma::uvec(1).fill(1),
          sortedIndices), boundingBoxes(1, selectedIndex), INT_MAX),
          0.0, INT_MAX);
    

    arma::mat calculateIoU = intersectionArea /
        (area(sortedIndices).t() - intersectionArea + area(selectedIndex));

    sortedIndices = sortedIndices(arma::find(calculateIoU <= 0.5));
  }

  selectedIndices = arma::fliplr(selectedIndices);
  cout << endl;
  selectedIndices.print();
  selectedBoundingBoxes = boundingBoxes.cols(selectedIndices);
  selectedBoundingBoxes.print();
}