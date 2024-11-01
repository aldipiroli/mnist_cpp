#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H
#include "model.h"
#include "utils.h"

struct ObjectDetectorParameters
{
    ModelParameters model_parameters;
    int chunk_size_x;
    int chunk_size_y;
    int step;
    float iou_threshold = 0.01;
    float cls_threshold = 0.5;
};
class ObjectDetector : public Model
{
public:
    ObjectDetector(const ObjectDetectorParameters &parameters) : parameters_(parameters), Model(parameters.model_parameters) {}
    void runInference(const char *img_path);

private:
    ObjectDetectorParameters parameters_;
};

#endif // !OBJECT_DETECTOR
