cc_library(
    name = "utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h","model.h"],
    copts = [
    "-I/usr/local/include/tensorflow",
    "-I/usr/include/opencv4",
    ],
    linkopts = [
      "-ltensorflow",
      "-lopencv_core",
      "-lopencv_imgcodecs",
      "-lopencv_imgproc",
      "-lopencv_highgui"
    ],
    visibility = ["//visibility:public", ],
)


cc_library(
    name = "model",
    hdrs = ["model.h"],
    srcs = ["model.cpp"],
    copts = [
      "-I/usr/local/include",
      "-I/usr/local/include/tensorflow",
      "-I/usr/include/opencv4",
    ],
    linkopts = [
      "-L/usr/local/lib", 
      "-ltensorflow",
    ],
    visibility = ["//visibility:public"],
    deps = [":utils"]
)


cc_library(
    name = "object_detector",
    hdrs = ["object_detector.h"],
    srcs = ["object_detector.cpp"],
    copts = [
      "-I/usr/local/include",
      "-I/usr/local/include/tensorflow",
      "-I/usr/include/opencv4",
    ],
    linkopts = [
      "-L/usr/local/lib", 
      "-ltensorflow",
    ],
    visibility = ["//visibility:public"],
    deps = [":utils", ":model"]
)