#include "object_detector.h"

void ObjectDetector::runInference(const char *img_path)
{
    my_utils::ChunckedImage chunked_img = my_utils::loadImageInChunks(img_path, parameters_.chunk_size_x, parameters_.chunk_size_y, parameters_.step);

    std::vector<std::tuple<int, float>> outputs;
    std::vector<float> output_scores;
    for (int i = 0; i < chunked_img.chunks.size(); i++)
    {   
        // Load chunk
        cv::Mat curr_chunk = chunked_img.chunks[i];
        curr_chunk = my_utils::resizeImage(curr_chunk, parameters_.model_parameters.input_size_dim1, parameters_.model_parameters.input_size_dim2);

        // Convert to tensor
        TF_Tensor *input_tensor = nullptr;
        my_utils::toTensor(curr_chunk, &input_tensor, getModelParameters());

        // Run inference
        std::tuple<int, float> out = runInferenceOnTensor(input_tensor);
        outputs.push_back(out);
        output_scores.push_back(std::get<1>(out));
    }
    std::vector<int> valid_boxes = my_utils::apply_nms(chunked_img, output_scores, parameters_.iou_threshold, parameters_.cls_threshold);
    // my_utils::visualizeRawOutputs(chunked_img, outputs);
    for (int i : valid_boxes)
    {
        std::cout << std::endl
                  << "[" << std::get<0>(outputs[i]) << "] " << std::get<1>(outputs[i]);
    }
    my_utils::visualizeSelectedOutputs(chunked_img, outputs, valid_boxes);
}