from models.tf import TFConv, TFBN, TFBottleneck, TFConv2d
from models.common import Concat, Conv, C3, SPPF
from models.yolo import Detect
import torch
import sys
import torch.nn as nn
import tensorflow as tf
from prune_utils.TFLayers import MyTFC3, TFSPPF, MyTFModel
from models.tf import TFDetect, TFConcat, TFUpsample
from tensorflow import keras as tf_keras
from models.experimental import attempt_load
from export import export_tflite
import argparse
from pathlib import Path
import platform
import os
from utils.general import print_args, check_img_size


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def export(data=ROOT / "data/coco128.yaml",  # 'dataset.yaml path'
    weights=ROOT / "runs/train/exp/weights/best.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv5 Detect() inplace=True
    keras=False,  # use Keras
    optimize=False,  # TorchScript: optimize for mobile
    int8=False,  # CoreML/TF INT8 quantization
    per_tensor=False,  # TF per tensor quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    cache="",  # TensorRT: timing cache path
    simplify=False,  # ONNX: simplify model
    mlmodel=False,  # CoreML: Export in *.mlmodel format
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    nms=False,  # TF: add NMS to model
    agnostic_nms=False,  # TF: add agnostic NMS to model
    topk_per_class=100,  # TF.js NMS: topk per class to keep
    topk_all=100,  # TF.js NMS: topk for all classes to keep
    iou_thres=0.45,  # TF.js NMS: IoU threshold
    conf_thres=0.25,  # TF.js NMS: confidence threshold
          ):

    
    model = attempt_load(weights,device="cpu", inplace=True, fuse=True)
    yaml_data = model.yaml["backbone"]+model.yaml["head"]

    imgsz *= 2 if len(imgsz) == 1 else 1 
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]
    
    
    layers = []
    with torch.no_grad():
        
        anchors, nc = model.yaml["anchors"], model.yaml["nc"]
        
        for i,(l, (f, n, m, args)) in enumerate(zip(model.model, yaml_data)):
            
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except NameError:
                    pass
            
            if isinstance(l, Concat):
                tf_m = TFConcat(1, w=l) # This layer single paramter d, d is 1 for all Concat in yaml. TFConcat expects 1 too, and handle changes accordingly
       
                
            elif isinstance(l, Conv):
    
                # required params of TFConv: c1, c2, k=1, s=1, p=None, and w=None
                # I want stride and padding params from config, filters will be handled using our pruned torch model
                cfg_params = args
                
                # contains padding parameter 
                if len(cfg_params)==4:
                    c2, k, s, p = cfg_params
                else:
                    c2, k, s = cfg_params
                    p=None
                c1, c2 = l.conv.in_channels, l.conv.out_channels
                tf_m = TFConv(c1, c2, k, s, p, w = l)
            
        
            elif isinstance(l, C3):
                
                c1, c_, c2, n = l.cv1.conv.in_channels, l.cv1.conv.out_channels, l.cv3.conv.out_channels, len(l.m)
                tf_m = MyTFC3(c1,c_,c2,n,*args[1:],w=l) # channel in, channel out, n, shortcut(optional), layer
    
        
        
            elif isinstance(l, nn.Upsample):
                
                tf_m = TFUpsample(*args,w=l)
        
            elif isinstance(l, SPPF):
                c1, c_, c2, = l.cv1.conv.in_channels, l.cv1.conv.out_channels, l.cv2.conv.out_channels
                tf_m = TFSPPF(c1, c_, c2, *args[1:], w=l)
        
            # If Detect instance, then adjust each detecion conv layer. do not prune them.
            elif isinstance(l, Detect):
                detection_input_layers_idxs = [17, 20, 23]
                outs = []
                for detection_input_layer_idx in detection_input_layers_idxs:
                    out_channels = model.model[detection_input_layer_idx].cv3.conv.out_channels
                    outs.append(out_channels)
                
                args.append(outs)
                args.append(imgsz)
                tf_m = TFDetect(*args, w=l)
    
            tf_m.f, tf_m.i = f, i
            layers.append(tf_m)
        keras_model = tf_keras.Sequential(layers)
    export2tflite(keras_model, weights=weights, data = data, per_tensor=per_tensor, tf_nms=nms, agnostic_nms=agnostic_nms, topk_per_class=topk_per_class, topk_all=topk_all, iou_thres=iou_thres, conf_thres=conf_thres, keras=keras,dynamic=dynamic, batch_size = 1, imgsz=imgsz, int8=int8)


def export2tflite(keras_model,weights="runs/train/exp/weights/best.pt", data = "coc128.yaml", per_tensor=False, tf_nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, keras=False,dynamic=False, batch_size = 1, imgsz=(416, 416), int8=False):
    
    tf_model = MyTFModel(sequential_model = keras_model)
    
    h,w = imgsz
    im = tf.zeros((1, h, w, 3))  # BHWC order for TensorFlow
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(h,w, 3), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()


    im2 = tf.transpose(im, perm=[0, 3, 1, 2]) # format expected by export_tflite
    
    export_tflite(keras_model, im2, weights, int8, per_tensor, data=data, nms=tf_nms, agnostic_nms=agnostic_nms)



def parse_opt(known=False):
    """
    Parse command-line options for YOLOv5 model export configurations.

    Args:
        known (bool): If True, uses `argparse.ArgumentParser.parse_known_args`; otherwise, uses `argparse.ArgumentParser.parse_args`.
                      Default is False.

    Returns:
        argparse.Namespace: Object containing parsed command-line arguments.

    Example:
        ```python
        opts = parse_opt()
        print(opts.data)
        print(opts.weights)
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640, 640], help="image (h, w)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF/OpenVINO INT8 quantization")
    parser.add_argument("--per-tensor", action="store_true", help="TF per-tensor quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--cache", type=str, default="", help="TensorRT: timing cache file path")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--mlmodel", action="store_true", help="CoreML: Export in *.mlmodel format")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: add agnostic NMS to model")
    parser.add_argument("--topk-per-class", type=int, default=100, help="TF.js NMS: topk per class to keep")
    parser.add_argument("--topk-all", type=int, default=100, help="TF.js NMS: topk for all classes to keep")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Run(**vars(opt))  # Execute the run function with parsed options."""
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        export(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    
