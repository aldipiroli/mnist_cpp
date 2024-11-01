# mnist_cpp
Training a network on the mnist_dataset in tensorflow and then deploying it in C++.

## Run Object Detector 
Note: The classification network applied to a sliding window selection of the source image.
```bash
bazel run //application/src:object_detector -- /ABSOLUTE/PATH/TO/mnist_cpp/assets/ckpts /ABSOLUTE/PATH/TO/mnist_cpp/assets/imgs/multi.png
```
![image description](assets/teaser/output_object_detection.png)

## Run classification
```bash
bazel run //application/src:classification -- /ABSOLUTE/PATH/TO/mnist_cpp/assets/ckpts /ABSOLUTE/PATH/TO/mnist_cpp/assets/imgs/single.png
```
![image description](assets/teaser/output_classification.png)

## Downalod the mnist_dataset


## Run training
Download the mnist_dataset. The data should be download in `mnist_cpp/data/mnist`.
```bash
python python/dataloader/utils.py
```

Run model training:
```bash
bazel run //python/tools:train -- --path /ABSOLUTE/PATH/TO/mnist_cpp/
```

## Run inference in python
```bash
bazel run //python/tools:run_inference -- --path /ABSOLUTE/PATH/TO/mnist_cpp/
```

## Run on Docker Container 
Build the docker container 
```
cd mnist_cpp/docker
bash build_docker.sh
```
Run the docking container 
```
cd mnist_cpp/docker
bash run_docker.sh
```