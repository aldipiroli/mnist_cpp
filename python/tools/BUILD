py_binary(
  name="train",
  srcs = ["train.py"],
  deps = [  "//python/model:model",  "//python/utils:utils",   "//python/model:losses", "//python/model:metrics", "//python/dataloader:dataloader" ]
)

py_binary(
  name="run_inference",
  srcs = ["run_inference.py"],
  deps = [  "//python/model:model",  "//python/utils:utils" ]
)


py_binary(
    name = "convert_model",
    srcs = ["convert_model.py"],
)
