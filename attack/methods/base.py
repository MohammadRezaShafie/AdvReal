import torch
from abc import ABC, abstractmethod
from torch.optim.optimizer import Optimizer


class BaseAttacker(Optimizer):
    """An Attack Base Class
    کلاس پایه حمله.
    
    This class implements the core adversarial attack logic:
    این کلاس منطق اصلی حمله متخاصم را پیاده‌سازی می‌کند:
    - Loss function computation / محاسبه تابع هزینه
    - Gradient-based optimization / بهینه‌سازی مبتنی بر گرادیان
    - Perturbation bounds enforcement / اعمال محدودیت‌های اغتشاش
    """

    def __init__(self, loss_func, norm: str, cfg, device: torch.device, detector_attacker):
        """
        Initialize base attacker with loss function and perturbation constraints.
        مقداردهی اولیه مهاجم پایه با تابع هزینه و محدودیت‌های اغتشاش.

        :param loss_func: a loss function to calculate the loss / تابع هزینه
        :param norm: the attack norm [L0, L1, L2, L_infty] / نرم حمله
        :param cfg: configuration object / شی پیکربندی
        :param device: 'cpu' or 'cuda' / دستگاه پردازش
        :param detector_attacker: this attacker should have attributes vlogger / مهاجم تشخیص‌دهنده
        """
        defaults = dict(lr=cfg.STEP_LR)
        params = [detector_attacker.patch_obj.patch]
        super().__init__(params, defaults)

        self.loss_fn = loss_func
        self.cfg = cfg
        self.detector_attacker = detector_attacker
        self.device = device
        self.norm = norm
        # Epsilon bounds for perturbation magnitude
        # محدوده‌های اپسیلون برای بزرگی اغتشاش
        self.min_epsilon = 0.
        self.max_epsilon = cfg.EPSILON / 255.
        self.max_iters = cfg.MAX_EPOCH
        self.iter_step = cfg.ITER_STEP
        self.attack_class = cfg.ATTACK_CLASS


    def logger(self, detector, adv_tensor_batch, bboxes, loss_dict):
        vlogger = self.detector_attacker.vlogger
        # TODO: this is a manually appointed logger iter num 77(for INRIA Train)
        if vlogger:
            vlogger.note_loss(loss_dict['loss'], loss_dict['det_loss'], loss_dict['tv_loss'])
            if vlogger.iter % 77 == 0:
                filter_box = self.detector_attacker.filter_bbox
                vlogger.write_tensor(self.detector_attacker.universal_patch[0], 'adv patch')
                plotted = self.detector_attacker.plot_boxes(adv_tensor_batch[0], filter_box(bboxes[0]))
                vlogger.write_cv2(plotted, f'{detector.name}')

    def non_targeted_attack(self, ori_tensor_batch, detector):
        """Execute non-targeted adversarial attack.
        اجرای حمله متخاصم غیرهدفمند.
        
        This method:
        این روش:
        1. Applies adversarial patch to images / وصله متخاصم را به تصاویر اعمال می‌کند
        2. Runs detection on adversarial images / تشخیص را روی تصاویر متخاصم اجرا می‌کند
        3. Computes attack loss to minimize detection confidence / هزینه حمله را برای کمینه کردن اطمینان تشخیص محاسبه می‌کند
        
        :param ori_tensor_batch: Original image batch / دسته تصویر اصلی
        :param detector: Object detection model / مدل تشخیص شیء
        :return: loss, tv_loss, obj_loss / هزینه، هزینه TV، هزینه شیء
        """
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            # Apply the universal adversarial patch
            # اعمال وصله متخاصم جهانی
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)

            adv_tensor_batch = adv_tensor_batch.to(detector.device)
            # Run detection on adversarial images
            # اجرای تشخیص روی تصاویر متخاصم
            # Be explicit about keys to avoid dict ordering issues across detectors
            det_out = detector(adv_tensor_batch)
            bboxes = det_out.get('bbox_array')
            confs = det_out.get('obj_confs')
            cls_array = det_out.get('cls_max_ids')

            # Robustness: if any come back as lists, convert/pad to tensors
            if isinstance(confs, list):
                # Pad variable-length per-image confs to a tensor [B, max_len]
                max_len = max((c.numel() if isinstance(c, torch.Tensor) else 0) for c in confs) if len(confs) else 0
                B = len(confs)
                confs_tensor = torch.zeros((B, max_len), device=self.device)
                for i, c in enumerate(confs):
                    if isinstance(c, torch.Tensor) and c.numel() > 0:
                        n = c.numel()
                        confs_tensor[i, :n] = c.to(self.device)
                confs = confs_tensor
            if isinstance(cls_array, list):
                max_len = max((c.numel() if isinstance(c, torch.Tensor) else 0) for c in cls_array) if len(cls_array) else 0
                B = len(cls_array)
                cls_tensor = torch.full((B, max_len), -1, dtype=torch.long, device=self.device)
                for i, c in enumerate(cls_array):
                    if isinstance(c, torch.Tensor) and c.numel() > 0:
                        n = c.numel()
                        cls_tensor[i, :n] = c.to(self.device).long()
                cls_array = cls_tensor

            if hasattr(self.cfg, 'class_specify'):
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                if not isinstance(confs, torch.Tensor) or confs.numel() == 0:
                    print("Error: confs tensor is empty!")
                    # Create a tiny epsilon tensor to keep gradients flowing
                    confs = torch.full((adv_tensor_batch.shape[0], 1), 1e-10, device=self.device)
                else:
                    confs = confs.max(dim=-1, keepdim=True)[0]
            loss,tv_loss,obj_loss = self.attack_loss(confs=confs)
        return loss,tv_loss,obj_loss

    @abstractmethod
    def patch_update(self, **kwargs):
        pass

    @property
    def patch_obj(self):
        return self.detector_attacker.patch_obj

    def attack_loss(self, confs):
        obj_loss = self.loss_fn(confs=confs)
        tv_loss = self.detector_attacker.patch_obj.total_variation()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.cuda.FloatTensor([0.1]))
        loss = obj_loss + tv_loss.to(obj_loss.device)
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out

    def begin_attack(self):
        """
        to tell attackers: now, i'm begin attacking!
        """
        pass

    def end_attack(self):
        """
        to tell attackers: now, i'm stop attacking!
        """
        pass
