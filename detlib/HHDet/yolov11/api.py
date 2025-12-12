import torch
import types
from ...base import DetectorBase


class HHYolov11(DetectorBase):
    """
    Detector wrapper for YOLOv11 to unify loading and inference with the AdvReal pipeline.

    Responsibilities:
    - load(model_weights, model_config=...) initializes the model and sets eval mode.
    - __call__(batch_tensor) returns a dict with:
        - 'bbox_array': list of tensors per batch item, each [x1,y1,x2,y2,conf,cls]
        - 'obj_confs': raw objectness/confidence tensor from logits if available
    - parameters() exposes model parameters.

    Implementation notes:
    - Uses Ultralytics YOLO API if available. Falls back gracefully if API changes.
    - Expects inputs normalized to [0,1], shape [B,3,H,W]. Boxes returned normalized to [0,1].
    """

    def __init__(self, name, cfg, input_tensor_size=640, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.imgsz = (input_tensor_size, input_tensor_size)
        self.detector = None
        self.names = None

    def load(self, model_weights, **args):
        """
        Load YOLOv11 model weights. Prefer Ultralytics YOLO API.

        Args:
            model_weights: path to .pt weights
            model_config: optional model YAML, ignored for Ultralytics API
        """
        # Try Ultralytics (v8+/v11) unified loader
        try:
            from ultralytics import YOLO
            self.detector = YOLO(model_weights)
            # Compatibility shim: some Ultralytics versions call model.fuse(verbose=...),
            # while torch model.fuse may not accept 'verbose'. Wrap to ignore extra kwargs.
            try:
                orig_fuse = self.detector.model.fuse

                def _fuse_noverbose(this, *args, **kwargs):
                    return orig_fuse()

                self.detector.model.fuse = types.MethodType(_fuse_noverbose, self.detector.model)
            except Exception:
                pass

            # Compatibility shim: ignore unknown kwargs like 'embed' passed by AutoBackend
            try:
                orig_forward = self.detector.model.forward

                def _forward_ignore_unknown(this, *args, **kwargs):
                    # Drop kwargs not supported by some model versions
                    kwargs.pop('embed', None)
                    kwargs.pop('verbose', None)
                    return orig_forward(*args, **kwargs)

                self.detector.model.forward = types.MethodType(_forward_ignore_unknown, self.detector.model)
            except Exception:
                pass
            # Get class names if present
            try:
                self.names = self.detector.names if hasattr(self.detector, 'names') else None
            except Exception:
                self.names = None
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv11 weights via Ultralytics: {e}")

        self.eval()

    def __call__(self, batch_tensor, **kwargs):
        """
        Run inference and return normalized boxes and confidences compatible with utils_camou.
        """
        assert self.detector is not None, "YOLOv11 model not loaded"

        # Ultralytics returns a list of Results. Use stream=False for batched output.
        # Move tensor to device
        batch_tensor = batch_tensor.to(self.device)
        # 1) Get post-NMS results for bbox_array (no grad needed)
        with torch.no_grad():
            results = self.detector(batch_tensor, verbose=False)

        # 2) Get raw predictions from the underlying model to build a differentiable
        #    obj_confs tensor for optimization (keep grads).
        try:
            raw = self.detector.model(batch_tensor)
            # Some versions return (pred, aux), keep first tensor
            if isinstance(raw, (list, tuple)):
                raw_pred = raw[0]
            else:
                raw_pred = raw
            # raw_pred shape: [B, N, D]
            B, N, D = raw_pred.shape[0], raw_pred.shape[1], raw_pred.shape[2]
            # Try to infer nc (num classes) from model names if available
            nc = None
            if hasattr(self.detector, 'names') and self.detector.names:
                try:
                    nc = len(self.detector.names)
                except Exception:
                    nc = None
            # Determine layout: [x,y,w,h,obj,cls...] or [x,y,w,h,cls...]
            if nc is not None and D == nc + 5:
                obj = torch.sigmoid(raw_pred[..., 4])
                cls_logits = raw_pred[..., 5:5 + nc]
            elif nc is not None and D == nc + 4:
                # No explicit obj; approximate using max class prob
                obj = torch.sigmoid(raw_pred[..., 4:]).max(-1).values
                cls_logits = raw_pred[..., 4:4 + nc]
            else:
                # Fallback: assume last dim after first 5 are classes if D>6
                if D > 6:
                    obj = torch.sigmoid(raw_pred[..., 4])
                    cls_logits = raw_pred[..., 5:]
                else:
                    # As a last resort: treat the last dimension(s) as a single confidence
                    obj = torch.sigmoid(raw_pred[..., -1])
                    cls_logits = raw_pred[..., -1:]
            cls_ids = cls_logits.argmax(-1)
            cls_conf = torch.sigmoid(cls_logits).max(-1).values
            # Use objectness as the optimization signal; if not available, use class conf
            obj_signal = obj if obj is not None else cls_conf
            # Flatten to [B, N]
            obj_signal = obj_signal.view(B, N)
            cls_ids = cls_ids.view(B, N)
            has_raw = True
        except Exception:
            # Raw path unavailable, fall back to non-differentiable zeros (won't train)
            has_raw = False

        bbox_array = []
        confs_list = []  # per-image conf tensors (variable length)
        cls_list = []    # per-image class tensors (variable length)

        # Each result corresponds to one image in the batch
        for i, res in enumerate(results):
            # res.boxes: xyxy (x1,y1,x2,y2), conf, cls
            if not hasattr(res, 'boxes'):
                bbox_array.append(torch.empty((0, 6), device=self.device))
                confs_list.append(torch.empty((0,), device=self.device))
                cls_list.append(torch.empty((0,), dtype=torch.long, device=self.device))
                continue

            # Ultralytics Results:
            # - boxes.xyxyn: normalized [0,1] coords w.r.t original image
            # - boxes.xyxy: pixel coords w.r.t original image
            # Prefer normalized coords directly to avoid incorrect scaling
            has_norm = hasattr(res.boxes, 'xyxyn')
            boxes_xyxy = res.boxes.xyxy  # (N,4) pixels
            boxes_xyxyn = res.boxes.xyxyn if has_norm else None  # (N,4) normalized
            conf = res.boxes.conf        # (N,)
            cls = res.boxes.cls          # (N,)

            if boxes_xyxy.numel() == 0:
                bbox_array.append(torch.empty((0, 6), device=self.device))
                confs_list.append(torch.empty((0,), device=self.device))
                cls_list.append(torch.empty((0,), dtype=torch.long, device=self.device))
                continue

            # Normalize to [0,1]
            if boxes_xyxyn is not None:
                boxes_out_norm = boxes_xyxyn.clone()
            else:
                # Fall back to using Ultralytics res.orig_shape to normalize
                try:
                    orig_h, orig_w = res.orig_shape
                except Exception:
                    # As a fallback, use current batch tensor spatial dims
                    orig_h, orig_w = batch_tensor.shape[-2], batch_tensor.shape[-1]
                boxes_out_norm = boxes_xyxy.clone()
                boxes_out_norm[:, [0, 2]] /= float(orig_w)
                boxes_out_norm[:, [1, 3]] /= float(orig_h)

            # Compose [x1,y1,x2,y2,conf,cls]
            boxes_out = torch.cat([boxes_out_norm, conf.unsqueeze(-1), cls.unsqueeze(-1)], dim=-1)
            bbox_array.append(boxes_out)
            confs_list.append(conf if not has_raw else obj_signal[i, :conf.numel()])
            # prefer raw cls_ids when available to keep consistency length-wise
            if has_raw:
                cls_list.append(cls_ids[i, :conf.numel()].to(torch.long))
            else:
                cls_list.append(cls.to(torch.long))

        # Pad variable-length confs/classes preserving gradients using pad_sequence
        if len(confs_list) == 0:
            obj_confs = torch.empty((0, 0), device=self.device)
            cls_max_ids = torch.empty((0, 0), dtype=torch.long, device=self.device)
        else:
            from torch.nn.utils.rnn import pad_sequence
            obj_confs = pad_sequence(confs_list, batch_first=True, padding_value=0.0)
            # pad classes separately (no grad required)
            max_len = max(cl.numel() for cl in cls_list)
            B = len(cls_list)
            cls_max_ids = torch.full((B, max_len), -1, dtype=torch.long, device=self.device)
            for i, cl in enumerate(cls_list):
                n = cl.numel()
                if n > 0:
                    cls_max_ids[i, :n] = cl

        self.raw_preds = None  # Ultralytics abstracts raw logits; keep None for now
        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, 'cls_max_ids': cls_max_ids}
        return output

    def parameters(self):
        # Expose parameters if underlying model exists; else empty list (frozen by default)
        try:
            return self.detector.model.parameters()
        except Exception:
            return []
