#include "application/library/model.h"
#include "application/library/object_detector.h"

int main(int argc, char *argv[])
{
  // Simple Classification
  ModelParameters param;
  param.input_size_dim1 = 28;
  param.input_size_dim2 = 28;
  param.input_size_dim3 = 1;
  param.output_size = 10;

  Model my_model(param);
  my_model.loadModel(argv[1]); 
  my_model.runInferenceOnImage(argv[2]);
  return 0;
}
