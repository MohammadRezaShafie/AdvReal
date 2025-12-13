import torch
import numpy as np
from .yolov5.utils.general import non_max_suppression, scale_coords
from ...base import DetectorBase
from ultralytics import YOLO

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


class HHYolov11(DetectorBase):
    def __init__(self, name, cfg, input_tensor_size=640, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.imgsz = (input_tensor_size, input_tensor_size)
        self.stride, self.pt = None, None

    def load(self, model_weights, **args):
        yolo_wrapper = YOLO(model_weights)
        self.detector = yolo_wrapper.model.to(self.device)
        self.detector.load_state_dict(torch.load(model_weights, map_location=self.device)['model'].float().state_dict())
        self.eval()

    def __call__(self, batch_tensor, **kwargs):
        
        print(f"batch_tensor Shape: {batch_tensor.shape}")
        detections_with_grad = self.detector(batch_tensor, augment=False, visualize=False)[0]
        print(f"detections_with_grad Shape: {detections_with_grad.shape}")
        detections_with_grad_clone = detections_with_grad.clone()
        preds = non_max_suppression(detections_with_grad_clone, self.conf_thres, self.iou_thres)
        cls_max_ids = None
        bbox_array = []
        for pred in preds:
            box = scale_coords(batch_tensor.shape[-2:], pred, self.ori_size)
            box[:, [0, 2]] /= self.ori_size[1]
            box[:, [1, 3]] /= self.ori_size[0]
            bbox_array.append(box)
        obj_confs = detections_with_grad[..., 4]
        self.raw_preds = detections_with_grad
        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, 'cls_max_ids': cls_max_ids}
        return output

    def parameters(self):
        return self.detector.parameters()
