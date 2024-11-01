#include "model.h"
#include "utils.h"
#include <iostream>

void Model::displaySizes()
{
  std::cout << "processing input sizes: " << model_parameters_.input_size_dim1
            << ", " << model_parameters_.input_size_dim2 << ", "
            << model_parameters_.input_size_dim3;
  std::cout << "\nprocessing output size: " << model_parameters_.output_size;
}

ModelParameters Model::getModelParameters()
{
  return model_parameters_;
}

void Model::displayTFVersion()
{
  std::cout << "Tensorflow Version: " << TF_Version();
}

void Model::testTensorflow()
{
  std::cout << "Tensorflow Version:--> " << TF_Version();
}

void Model::loadModel(const char *model_dir_path)
{
  graph_ = TF_NewGraph();
  TF_Status *status = TF_NewStatus();
  TF_SessionOptions *session_options = TF_NewSessionOptions();
  TF_Buffer *run_options = nullptr;

  const char *tags[] = {"serve"}; 
  session_ =
      TF_LoadSessionFromSavedModel(session_options, run_options, model_dir_path,
                                   tags, 1, graph_, nullptr, status);

  if (TF_GetCode(status) != TF_OK)
  {
    std::cerr << "Error loading model: " << TF_Message(status) << std::endl;
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(session_options);
    TF_DeleteGraph(graph_);
    return;
  }
}

void Model::checkSessionStatus(TF_Status *status)
{
  if (TF_GetCode(status) != TF_OK)
  {
    std::cerr << "Error running session: " << TF_Message(status) << std::endl;
  }
  else
  {
    std::cerr << "Session ran succesfuly " << TF_Message(status) << std::endl;
  }
}

void Model::runInferenceOnImage(const char *img_path)
{
  TF_Tensor *input_tensor = nullptr;
  cv::Mat img_full_size = my_utils::loadImageFromFile(img_path);
  cv::Mat img = my_utils::resizeImage(img_full_size, model_parameters_.input_size_dim1, model_parameters_.input_size_dim2);
  my_utils::toTensor(img, &input_tensor, model_parameters_);

  std::tuple<int, float> output = runInferenceOnTensor(input_tensor, true);
  my_utils::visualizeClassificationOutput(img_full_size, output);
}

std::tuple<int, float> Model::runInferenceOnTensor(TF_Tensor *input_tensor, bool verbose)
{
  // Create an array of input tensors
  TF_Tensor *inputs[] = {input_tensor};
  TF_Tensor *outputs[1] = {nullptr};

  // Prepare the input and output tensors
  TF_Output input_op = {
      TF_GraphOperationByName(graph_, "serving_default_input_1"), 0};
  TF_Output output_op = {
      TF_GraphOperationByName(graph_, "StatefulPartitionedCall"), 0};

  // Run the session
  TF_Status *status = TF_NewStatus();
  TF_SessionRun(session_, nullptr, &input_op, inputs, 1, &output_op,
                outputs, 1, nullptr, 0, nullptr, status);
  // checkSessionStatus(status);

  // Process the output tensor
  float *output_data = static_cast<float *>(TF_TensorData(outputs[0]));
  int cls_id = my_utils::argmax(output_data, model_parameters_.output_size);
  float max_prob = my_utils::max_prob(output_data, model_parameters_.output_size);

  std::tuple<int, float> output(cls_id, max_prob);
  if (verbose)
  {
    std::cout << std::endl
              << "------------------- " << std::endl;
    std::cout << "Class: " << cls_id << ", prob: " << max_prob << " ";
  }
  return output;
}