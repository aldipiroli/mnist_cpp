cc_binary(
name =    "classification", 
srcs = ["classification.cpp"],
copts = [
    "-I/usr/include/opencv4",
],
deps = ["//application/library:model","//application/library:object_detector"],
)

cc_binary(
name =    "object_detector", 
srcs = ["object_detector.cpp"],
copts = [
    "-I/usr/include/opencv4",
],
deps = ["//application/library:model","//application/library:object_detector"],
)