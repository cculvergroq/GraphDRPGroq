#  This script is ran outside of the GraphDRP conda environment
#  due to this dependency issue
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# candle 0.0.1 requires protobuf==3.19, but you have protobuf 4.24.4 which is incompatible.


import onnx
import onnxruntime as ort
import onnxruntime.quantization as orq
from onnxconverter_common import float16
from pathlib import Path, PurePath

def process_onnx(inpath: str, outpath: str) -> None:
    onnx.shape_inference.infer_shapes_path(inpath, outpath, strict_mode=True)

    options = ort.SessionOptions()

    # Set graph optimization level
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )

    # To enable model serialization after graph optimization set this
    options.optimized_model_filepath = outpath
    _ = ort.InferenceSession(outpath, options)
    onnx.shape_inference.infer_shapes_path(outpath, strict_mode=True)


onnxfilepath = PurePath.joinpath(Path.cwd(), 'out', 'infer.onnx')
model = onnx.load(onnxfilepath)

#session_options = ort.SessionOptions()
#session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

optfilepath = PurePath.joinpath(Path.cwd(), 'out', 'infer_opt.onnx')
process_onnx(str(onnxfilepath),str(optfilepath))

model_fp16 = float16.convert_float_to_float16_model_path(optfilepath)
onnxfp16path = PurePath.joinpath(Path.cwd(), 'out', 'infer_opt_fp16.onnx')
onnx.save(model_fp16, str(onnxfp16path))

#session_options.optimized_model_filepath = str(optfilepath)
#session = ort.InferenceSession(str(onnxfp16path), session_options)