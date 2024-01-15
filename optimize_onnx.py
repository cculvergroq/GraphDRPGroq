#  This script is ran outside of the GraphDRP conda environment
#  due to this dependency issue
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# candle 0.0.1 requires protobuf==3.19, but you have protobuf 4.24.4 which is incompatible.


import onnx
import onnxruntime as ort
import onnxruntime.quantization as orq
from onnxconverter_common import float16
from pathlib import Path, PurePath

onnxfilepath = PurePath.joinpath(Path.cwd(), 'out', 'infer.onnx')
model = onnx.load(onnxfilepath)
model_fp16 = float16.convert_float_to_float16(model)
onnxfp16path = PurePath.joinpath(Path.cwd(), 'out', 'infer_fp16.onnx')
onnx.save(model_fp16, str(onnxfp16path))

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

optfilepath = PurePath.joinpath(Path.cwd(), 'out', 'infer_opt_fp16.onnx')
session_options.optimized_model_filepath = str(optfilepath)
session = ort.InferenceSession(str(onnxfp16path), session_options)