#ifndef MODEL_H
#define MODEL_H
#include <string>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>

struct ModelParameters
{
  int input_size_dim1;
  int input_size_dim2;
  int input_size_dim3;
  int output_size;
};

class Model
{
public:
  Model(const ModelParameters &model_parameters) : model_parameters_(model_parameters) {}
  void displaySizes();
  void displayTFVersion();
  void loadModel(const char *model_dir_path);
  void runInferenceOnImage(const char *img_path);
  std::tuple<int, float> runInferenceOnTensor(TF_Tensor *input_tensor, bool verbose = false);
  void runObjectDetection(const char *img_path);
  void testTensorflow();
  void testOpenCV();
  void checkSessionStatus(TF_Status *status);
  ModelParameters getModelParameters();

private:
  ModelParameters model_parameters_;

  TF_Session *session_;
  TF_Graph *graph_;
};

#endif // !MODEL_H
