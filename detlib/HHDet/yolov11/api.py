import torch
import numpy as np
# Import Ultralytics NMS as 'ul_nms' to avoid conflict with legacy YOLOv5 functions
from ultralytics.utils.ops import non_max_suppression as ul_nms
from ...base import DetectorBase

class HHYolov11(DetectorBase):
    """
    Robust Detector wrapper for YOLOv11.
    Fixes the 'Garbage Output' (413 confidence) issue by forcing correct NMS usage.
    """

    def __init__(self, name, cfg, input_tensor_size=640, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.imgsz = (input_tensor_size, input_tensor_size)
        self.device = device
        self.detector = None
        
        # Load NMS thresholds
        self.conf_thres = getattr(self, 'conf_thres', 0.25)
        self.iou_thres = getattr(self, 'iou_thres', 0.45)

    def load(self, model_weights, **args):
        try:
            from ultralytics import YOLO
            # 1. Load the YOLO wrapper
            yolo_wrapper = YOLO(model_weights)
            
            # 2. Extract the underlying PyTorch model (nn.Module)
            self.detector = yolo_wrapper.model.to(self.device)
            self.detector.eval()
            self.names = self.detector.names
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv11 model: {e}")

    def __call__(self, batch_tensor, **kwargs):
        assert self.detector is not None, "Model not loaded"
        
        batch_tensor = batch_tensor.to(self.device)
        B, _, H, W = batch_tensor.shape

        # --- 1. Forward Pass (Get Raw Output) ---
        # YOLOv11 Raw Output Shape: [B, 84, 8400] (for 80 classes)
        # Rows 0-3: x, y, w, h (in pixels)
        # Rows 4-83: Class probabilities
        preds = self.detector(batch_tensor)
        
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # --- 2. Create 'Objectness' for Attack Gradients ---
        # The Attack loop expects [B, Anchors, 84] to calculate gradients.
        # So we transpose explicitly for this part.
        preds_transposed = preds.transpose(-1, -2) # [B, 8400, 84]
        
        # Calculate max class probability to simulate "objectness"
        pred_cls_logits = preds_transposed[..., 4:]
        obj_confs = torch.sigmoid(pred_cls_logits).max(dim=-1).values

        # --- 3. NMS for Visualization (CRITICAL FIX) ---
        # We MUST pass the UN-TRANSPOSED [B, 84, 8400] tensor to Ultralytics NMS.
        # It handles the dimensions internally.
        nms_preds = ul_nms(
            preds,  # <--- Passing the raw untransposed tensor
            self.conf_thres, 
            self.iou_thres, 
            multi_label=False
        )
        
        bbox_array = []
        for det in nms_preds:
            # det contains [x1, y1, x2, y2, conf, cls]
            if det is None or len(det) == 0:
                bbox_array.append(torch.empty((0, 6), device=self.device))
                continue
            
            det_norm = det.clone()
            
            # --- 4. Normalize Correctly ---
            # det contains pixels (e.g. 413.0). We must normalize to [0,1].
            # Only divide if values are > 1.0 (Pixels).
            if det_norm[:, :4].max() > 1.0:
                det_norm[:, [0, 2]] /= W
                det_norm[:, [1, 3]] /= H
            
            bbox_array.append(det_norm)

        self.raw_preds = preds_transposed

        output = {
            'bbox_array': bbox_array, 
            'obj_confs': obj_confs,
            'cls_max_ids': None 
        }
        
        return output

    def parameters(self):
        return self.detector.parameters()