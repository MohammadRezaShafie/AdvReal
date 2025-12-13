import torch
from .yolov5.utils.general import non_max_suppression, scale_coords
from .yolov5.models_v5.experimental import attempt_load
from .yolov5.models_v5.yolo import Model
from .yolov5.models_v5.utils.general import check_yaml
from ...base import DetectorBase

class HHYolov5(DetectorBase):
    def __init__(self, name, cfg, input_tensor_size=640, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.imgsz = (input_tensor_size, input_tensor_size)
        self.stride, self.pt = None, None

    def load_(self, model_weights, **args):
        w = str(model_weights[0] if isinstance(model_weights, list) else model_weights)
        self.detector = attempt_load(model_weights if isinstance(model_weights, list) else w, map_location=self.device, inplace=False)
        self.eval()
        self.stride = max(int(self.detector.stride.max()), 32)
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names

    def load(self, model_weights, **args):
        model_config = args['model_config']
        self.detector = Model(model_config).to(self.device)
        self.detector.load_state_dict(torch.load(model_weights, map_location=self.device)['model'].float().state_dict())
        self.eval()

    def __call__(self, batch_tensor, **kwargs):
        
        detections_with_grad = self.detector(batch_tensor, augment=False, visualize=False)[0]
        detections_with_grad_clone = detections_with_grad.clone()
        preds = non_max_suppression(detections_with_grad_clone, self.conf_thres, self.iou_thres)
        print(f"yolov5 preds Shape: {preds}")
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
