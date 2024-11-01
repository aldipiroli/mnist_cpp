#include "utils.h"
#include <iostream>
namespace my_utils
{
  cv::Mat loadImageFromFile(const std::string &image_path)
  {
    // Load the image using OpenCV
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE); // Load as grayscale
    if (img.empty())
    {
      std::cerr << "Error loading image: " << image_path << std::endl;
      return img;
    }
    return img;
  }

  cv::Mat resizeImage(cv::Mat img, int new_width, int new_height)
  {
    cv::resize(img, img, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);
    return img;
  }

  ChunckedImage loadImageInChunks(const std::string &image_path, int chunk_x, int chunk_y, int step)
  {
    ChunckedImage chunked_imge;

    // Load the image
    chunked_imge.full_img = loadImageFromFile(image_path);

    if (chunked_imge.full_img.empty())
    {
      std::cerr << "Error loading image: " << image_path << std::endl;
      exit(0);
    }

    // Traverse image in chunks
    int width = chunked_imge.full_img.size().width;
    int height = chunked_imge.full_img.size().height;
    std::list<cv::Mat> chunked_img;
    std::list<std::tuple<int, int>> chunked_img_idx;

    for (int i = 0; i < width; i += step)
    {
      for (int j = 0; j < height; j += step)
      {
        cv::Mat curr_img(chunk_x, chunk_y, CV_32F);
        BoundingBox box;

        // get chunk
        int max_i = i + chunk_x;
        int max_j = j + chunk_y;
        if ((max_i > width) || (max_j > height))
        {
          break;
        }

        curr_img = chunked_imge.full_img.colRange(i, max_i).rowRange(j, max_j);
        float min_value_threshold = 100;
        if (cv::sum(curr_img)[0] < min_value_threshold) // Remove empty chunks
        {
          continue;
        }
        chunked_imge.chunks.push_back(curr_img);

        // get box
        box.x = i;
        box.y = j;
        box.width = chunk_x;
        box.height = chunk_y;

        chunked_imge.boxes.push_back(box);
      }
    }
    return chunked_imge;
  }

  void visualizeBoudingBoxes(ChunckedImage chunked_img)
  {
    std::cout << chunked_img.boxes.size();
    for (int i = 0; i < chunked_img.boxes.size(); i++)
    {
      BoundingBox curr_box = chunked_img.boxes[i];
      cv::Rect rect(curr_box.x, curr_box.y, curr_box.width, curr_box.height);
      cv::rectangle(chunked_img.full_img, rect, getRandomColor());
    }
    cv::imshow("Display window", chunked_img.full_img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
  }

  void visualizeRawOutputs(ChunckedImage chunked_img, std::vector<std::tuple<int, float>> outputs)
  {
    assert(chunked_img.boxes.size() == outputs.size());
    for (int i = 0; i < chunked_img.boxes.size(); i++)
    {
      // cv::Mat tmp_img = chunked_img.full_img.clone();
      BoundingBox curr_box = chunked_img.boxes[i];
      cv::Rect rect(curr_box.x, curr_box.y, curr_box.width, curr_box.height);
      cv::rectangle(chunked_img.full_img, rect, getRandomColor());
      std::string cls_id = std::to_string(std::get<0>(outputs[i]));
      float cls_prob = std::get<1>(outputs[i]);

      cv::putText(chunked_img.full_img, cls_id, cv::Point(curr_box.x, curr_box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Display window", chunked_img.full_img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
  }

  void visualizeSelectedOutputs(ChunckedImage chunked_img, std::vector<std::tuple<int, float>> outputs, std::vector<int> valid_idx)
  {
    cv::Mat color_img;
    cv::cvtColor(chunked_img.full_img, color_img, cv::COLOR_GRAY2BGR);
    assert(chunked_img.boxes.size() == outputs.size());
    for (int i : valid_idx)
    {
      // cv::Mat tmp_img = chunked_img.full_img.clone();
      BoundingBox curr_box = chunked_img.boxes[i];
      cv::Rect rect(curr_box.x, curr_box.y, curr_box.width, curr_box.height);
      cv::rectangle(color_img, rect, cv::Scalar(0, 255, 0));
      std::string cls_id_str = std::to_string(std::get<0>(outputs[i]));
      std::string prob_str = std::to_string(std::get<1>(outputs[i]));
      float cls_prob = std::get<1>(outputs[i]);

      cv::putText(color_img, "[" + cls_id_str + "]" + " " + prob_str, cv::Point(curr_box.x + 5, curr_box.y + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    // display in color
    showImage(color_img);
  }

  void visualizeClassificationOutput(cv::Mat img, std::tuple<int, float> output)
  {
    cv::Mat color_img;
    cv::cvtColor(img, color_img, cv::COLOR_GRAY2BGR);

    std::string cls_id_str = std::to_string(std::get<0>(output));
    std::string prob_str = std::to_string(std::get<1>(output));
    float cls_prob = std::get<1>(output);

    cv::putText(color_img, "[" + cls_id_str + "]" + " " + prob_str, cv::Point(25, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
      // display in color
  showImage(color_img);
  }


cv::Scalar getRandomColor()
{
  std::random_device rd;                         // obtain a random number from hardware
  std::mt19937 gen(rd());                        // seed the generator
  std::uniform_int_distribution<> distr(0, 255); // define the range
  cv::Scalar color(distr(gen), distr(gen), distr(gen), 1);
  return color;
}

void showImage(cv::Mat img)
{
  cv::imshow("Display window", img);
  int k = cv::waitKey(0);
  cv::destroyAllWindows();
}

void toTensor(cv::Mat img, TF_Tensor **input_tensor, ModelParameters param)
{
  img.convertTo(img, CV_8U, 1.0);
  int64_t dims[4] = {1, param.input_size_dim1, param.input_size_dim2, param.input_size_dim3}; // Batch size of 1
  *input_tensor = TF_AllocateTensor(TF_UINT8, dims, 4, param.input_size_dim1 * param.input_size_dim2 * param.input_size_dim3);
  uint8_t *data = static_cast<uint8_t *>(TF_TensorData(*input_tensor));
  std::memcpy(data, img.data, param.input_size_dim1 * param.input_size_dim2); // Copy image data to tensor
}

int argmax(float *outputs, int num_outputs)
{
  float max_val = -1;
  int max_id = -1;
  for (int i = 0; i < num_outputs; ++i)
  {
    if (outputs[i] > max_val)
    {
      max_val = outputs[i];
      max_id = i;
    }
  }
  return max_id;
}

float max_prob(float *outputs, int num_outputs)
{
  float max_val = -1;
  int max_id = -1;
  for (int i = 0; i < num_outputs; ++i)
  {
    if (outputs[i] > max_val & outputs[i] != -1)
    {
      max_val = outputs[i];
      max_id = i;
    }
  }
  return outputs[max_id];
}

float getIoU(BoundingBox box1, BoundingBox box2)
{
  int x_overlap = std::max(0, std::min(box1.x + box1.width, box2.x + box2.width) - std::max(box1.x, box2.x));
  int y_overlap = std::max(0, std::min(box1.y + box1.height, box2.y + box2.height) - std::max(box1.y, box2.y));

  if (x_overlap <= 0 || y_overlap <= 0)
  {
    return 0.0f;
  }
  float intersection_area = x_overlap * y_overlap;

  float area_box1 = box1.width * box1.height;
  float area_box2 = box2.width * box2.height;
  float union_area = area_box1 + area_box2 - intersection_area;

  float iou = intersection_area / union_area;
  return iou;
}
void print_array(std::vector<int> arr)
{
  std::cout << std::endl
            << "-------------------" << std::endl;

  for (int i = 0; i < arr.size(); i++)
  {

    std::cout << i << ": " << arr[i] << std::endl;
  }
  std::cout << std::endl
            << "-------------------" << std::endl;
}

int find_max_element(std::vector<float> vec, std::vector<int> valid_idx)
{
  float max_val = -1;
  int max_id = -1;
  for (int curr_idx : valid_idx)
  {
    if (vec[curr_idx] > max_val)
    {
      max_val = vec[curr_idx];
      max_id = curr_idx;
    }
  }
  return max_id;
}

std::vector<int> arange(int min, int max)
{
  std::vector<int> arange;
  for (int i = min; i < max; i++)
  {
    arange.push_back(i);
  }
  return arange;
}

std::vector<int> apply_nms(ChunckedImage chunked_img, std::vector<float> outputs, float iou_th, float conf_th)
{
  std::vector<int> keep;
  std::vector<int> all_idx = arange(0, outputs.size());
  std::vector<my_utils::BoundingBox> boxes = chunked_img.boxes;

  // remove low scores predictions
  for (int i = 0; i < outputs.size(); i++)
  {
    if (outputs[i] < conf_th)
    {
      all_idx[i] = -1;
    }
  }

  while (true)
  {
    int max_prob_idx = find_max_element(outputs, all_idx);
    if (max_prob_idx == -1)
    {
      break;
    }
    keep.push_back(max_prob_idx);
    all_idx[max_prob_idx] = -1;

    for (int i = 0; i < all_idx.size(); i++)
    {
      if (all_idx[i] == -1)
      {
        continue;
      }
      my_utils::BoundingBox box1 = boxes[max_prob_idx];
      my_utils::BoundingBox box2 = boxes[i];
      float iou = getIoU(box1, box2);
      if (iou > iou_th)
      {
        all_idx[i] = -1;
      }
    }
  }
  std::cout << std::endl
            << "Before nms: " << outputs.size() << ", After nms: " << keep.size();

  return keep;
}
}
