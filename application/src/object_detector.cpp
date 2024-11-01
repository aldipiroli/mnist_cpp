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

  // Object Detection by Sliding Window
  ObjectDetectorParameters od_param;
  od_param.model_parameters = param;
  od_param.chunk_size_x = 150;
  od_param.chunk_size_y = 150;
  od_param.step = 5;
  od_param.iou_threshold = 0.001;
  od_param.cls_threshold = 0.5;

  ObjectDetector object_detector(od_param);
  object_detector.loadModel(argv[1]); 
  object_detector.runInference(argv[2]);

  return 0;
}
