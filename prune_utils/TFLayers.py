from models.tf import TFConv, TFBN, TFBottleneck, TFConv2d
from tensorflow import keras
import tensorflow as tf
"""
The code is adapted from the official repo for my pruned model. yolov5/models.tf file

"""
class MyTFC3(keras.layers.Layer):
    """CSP bottleneck layer with 3 convolutions for TensorFlow, supporting optional shortcuts and group convolutions."""

    def __init__(self, c1, c_, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        """
        Initializes CSP Bottleneck with 3 convolutions, supporting optional shortcuts and group convolutions.

        Inputs are ch_in, ch_out, number, shortcut, groups, expansion.
        """
        super().__init__()
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        """
        Processes input through a sequence of transformations for object detection (YOLOv5).

        See https://github.com/ultralytics/yolov5.
        """
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))



# Scenario 1: if c1=58, c_=27, c2=54, will it produce tensorflow version of C3 that's consistent with my pruned torch? yes it is
# Scenario 2: if c1=102, c_=48, c2=96, yes
# Scenatio 3: if c1=192, c_=90, c2=179, yes
# Scenario 4: if c1=358, c_=166, c2=333, yes
# scenario 5: if c1=371, c_=84, c2=169, yes
# scenario 6: if c1=358, c_=192, c2=384, yes
# Formula: c1= c3.cv1.conv.in_channels,  c_ = c3.cv1.conv.out_channels, c2 = c3.cv3.conv.out_channels
# How is this usually called in models.tf's parse_model? remember im talking about TFC3, not MyTFC3. Called as [c1,c2,n,*args[1:]]
# continued. I will call mine as [c1,c_,c2,n,*args[1:]]



class TFSPPF(keras.layers.Layer):
    """Implements a fast spatial pyramid pooling layer for TensorFlow with optimized feature extraction."""

    def __init__(self, c1, c_, c2, k=5, w=None):
        """Initializes a fast spatial pyramid pooling layer with customizable in/out channels, kernel size, and
        weights.
        """
        super().__init__()
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="SAME")

    def call(self, inputs):
        """Executes the model's forward pass, concatenating input features with three max-pooled versions before final
        convolution.
        """
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class MyTFModel:
    """Implements YOLOv5 model in TensorFlow, supporting TensorFlow, Keras, and TFLite formats for object detection."""

    def __init__(self, ch=3, nc=None, sequential_model=None):
        """Initializes TF YOLOv5 model with specified configuration, channels, classes, model instance, and input
        size.
        """
        super().__init__()
        
        self.model, self.savelist = sequential_model, [6, 4, 14, 10, 17, 20, 23]  #parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz)

    def predict(
        self,
        inputs,
        tf_nms=False,
        agnostic_nms=False,
        topk_per_class=100,
        topk_all=100,
        iou_thres=0.45,
        conf_thres=0.25,
    ):
        """Runs inference on input data, with an option for TensorFlow NMS."""
        y = []  # outputs
        x = inputs
        for m in self.model.layers:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.savelist else None)  # save output

        # Add TensorFlow NMS
        if tf_nms:
            boxes = self._xywh2xyxy(x[0][..., :4])
            probs = x[0][:, :, 4:5]
            classes = x[0][:, :, 5:]
            scores = probs * classes
            if agnostic_nms:
                nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
            else:
                boxes = tf.expand_dims(boxes, 2)
                nms = tf.image.combined_non_max_suppression(
                    boxes, scores, topk_per_class, topk_all, iou_thres, conf_thres, clip_boxes=False
                )
            return (nms,)
        return x 

    @staticmethod
    def _xywh2xyxy(xywh):
        """Converts bounding box format from [x, y, w, h] to [x1, y1, x2, y2], where xy1=top-left and xy2=bottom-
        right.
        """
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)
