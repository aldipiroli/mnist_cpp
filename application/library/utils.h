#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <random>
#include "model.h"
namespace my_utils
{

  class Point
  {
  public:
    Point(const int &x, const int &y) : x_(x), y_(y) {}
    int x_;
    int y_;
  };

  class Corners
  {
  public:
    Corners(const Point &a, const Point &b, const Point &c, const Point &d) : a_(a), b_(b), c_(c), d_(d) {}
    Point a_;
    Point b_;
    Point c_;
    Point d_;
  };

  struct BoundingBox
  {
    int x;
    int y;
    int width;
    int height;
  };

  struct ChunckedImage
  {
    cv::Mat full_img;
    std::vector<cv::Mat> chunks;
    std::vector<BoundingBox> boxes;
  };

  cv::Mat loadImageFromFile(const std::string &image_path);
  cv::Mat resizeImage(cv::Mat img, int new_width, int new_height);
  int argmax(float *outputs, int num_outputs);
  float max_prob(float *outputs, int num_outputs);
  ChunckedImage loadImageInChunks(const std::string &image_path, int chunk_x, int chunk_y, int step);
  void showImage(cv::Mat img);
  int truncateIndex(int idx, int size);
  void visualizeBoudingBoxes(ChunckedImage chunked_img);
  void visualizeRawOutputs(ChunckedImage chunked_img, std::vector<std::tuple<int, float>> outputs);
  void visualizeSelectedOutputs(ChunckedImage chunked_img, std::vector<std::tuple<int, float>> outputs, std::vector<int> valid_idx);
  void visualizeClassificationOutput(cv::Mat img, std::tuple<int, float> output);


  cv::Scalar getRandomColor();
  void toTensor(cv::Mat img, TF_Tensor **input_tensor, ModelParameters param);
  std::vector<int> apply_nms(ChunckedImage chunked_img, std::vector<float> outputs, float iou_th, float conf_th);
  float getIoU(BoundingBox box1, BoundingBox box2);
}
#endif // !UTILS_H
