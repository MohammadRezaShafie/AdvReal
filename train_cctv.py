"""
CCTV Fine-Tuning Script for Adversarial Patch Training.

3D-only training — adapts a pre-trained adversarial patch to CCTV camera
perspectives. No 2D INRIAPerson branch (patch already trained on 2D).

Loss = det_loss (3D CCTV composites) + tv_loss (patch smoothness)

Usage:
    python3 train_cctv.py                                           # defaults
    python3 train_cctv.py --nepoch 700 --checkpoints 500            # 200 epochs of fine-tuning
    python3 train_cctv.py -p ../advreal_data/patch_epoch_499.png    # custom patch path
"""
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.functional")

import sys
import os
import time
import argparse
import gc
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from datetime import datetime

from load_data import TotalVariation, YOLOv11MaxProbExtractor
from utils.parser import ConfigParser
from cctv_renderer import CCTVImageRenderer
import utils_camou

sys.path.append(os.path.abspath(''))


class CCTVPatchTrainer:
    """Fine-tune an adversarial patch for CCTV camera perspectives.

    3D-only training — renders person meshes with adversarial textures
    at steep CCTV camera angles onto real CCTV backgrounds, then
    optimizes the patch texture to minimize person detection confidence.
    """

    def __init__(self, args):
        self.args = args
        self.img_size = 416  # Standard YOLO input size

        # Device setup
        if args.device:
            device = torch.device(args.device)
            torch.cuda.set_device(device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # ── Load frozen YOLOv11 detector ──
        print("Loading YOLOv11 detector...")
        from detlib.HHDet.yolov11.api import HHYolov11
        cfg = ConfigParser(args.cfg)
        detector_cfg = cfg.DETECTOR
        self.model = HHYolov11(
            name="YOLOV11",
            cfg=detector_cfg,
            input_tensor_size=self.img_size,
            device=device,
        )
        model_weights = 'detlib/HHDet/yolov11/weights/yolo11s.pt'
        self.model.load(model_weights)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("✅ YOLOv11 loaded and frozen.")

        # ── Probability extractor ──
        self.prob_extractor = YOLOv11MaxProbExtractor(0, 80, self.model, self.img_size).to(device)

        # ── Total Variation loss ──
        self.tv_loss_fn = TotalVariation().to(device)

        # ── CCTV renderer ──
        print("Initializing CCTV renderer...")
        self.cctv_renderer = CCTVImageRenderer(
            device=device,
            nrsm_range=args.nrsm_range,
            target_size=self.img_size,
        )
        print("✅ CCTV renderer ready.")

    def _load_patch(self, patch_path):
        """Load a pre-trained adversarial patch from an image file."""
        patch_img = Image.open(patch_path).convert('RGB')
        patch = transforms.ToTensor()(patch_img).to(self.device)
        print(f"✅ Loaded patch from {patch_path}, shape: {patch.shape}")
        return patch

    def _init_tensorboard(self):
        """Create TensorBoard writer in the save directory."""
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        fname = self.args.save_path.split('/')[-1]
        log_dir = os.path.join(self.args.save_path, f'tb_{TIMESTAMP}_{fname}')
        return SummaryWriter(log_dir)

    def train(self):
        """Main fine-tuning loop — 3D CCTV branch only."""
        args = self.args
        os.makedirs(args.save_path, exist_ok=True)

        # Load pre-trained patch
        patch = self._load_patch(args.patch)
        patch.requires_grad_(True)

        # Optimizer (same as original train.py)
        optimizer = optim.Adam([patch], lr=args.lr, amsgrad=True)
        writer = self._init_tensorboard()

        print(f"\n{'=' * 60}")
        print(f"  CCTV Fine-Tuning: epochs {args.checkpoints} → {args.nepoch}")
        print(f"  Patch: {args.patch}")
        print(f"  Save:  {args.save_path}")
        print(f"  LR:    {args.lr}, TV weight: {args.tv_eta}")
        print(f"  Iters/epoch: {args.iters_per_epoch}")
        print(f"{'=' * 60}\n")

        for epoch in tqdm(range(args.checkpoints, args.nepoch), desc="CCTV Fine-Tune"):
            et0 = time.time()

            # LR schedule (same as original train.py — decay every 90th in each 100-block)
            if epoch % 100 == 90:
                optimizer.param_groups[0]['lr'] /= args.lr_decay
                print(f"  ↓ LR updated: {optimizer.param_groups[0]['lr']:.6f}")

            ep_det_loss = 0.0
            ep_tv_loss = 0.0
            ep_total_loss = 0.0
            n_batches = 0

            # Multiple iterations per epoch to cover both scenes with jitter variation
            for i_batch in range(args.iters_per_epoch):
                optimizer.zero_grad()

                # 1. Set current patch as texture on 3D meshes
                patch_c = patch.clone()
                patch_c = patch_c.clamp(0, 1)
                self.cctv_renderer.set_adv_patch_texture(patch_c)

                # 2. Generate CCTV composite batch (random scene, jittered placements)
                composite_images, gts, scene_name = self.cctv_renderer.generate_cctv_composite_batch()

                if len(composite_images) == 0:
                    continue

                # 3. Stack into batch tensors
                p_img_batch = torch.stack(composite_images).to(self.device)
                p_img_batch = p_img_batch[:, :3, :, :].float().clamp(0.0, 1.0)
                gts_batch = torch.stack(gts).to(self.device)

                # 4. Save 2 composite samples from each epoch (first batch only)
                if i_batch == 0:
                    comp_dir = os.path.join(args.save_path, 'composite')
                    os.makedirs(comp_dir, exist_ok=True)
                    for idx, img_tensor in enumerate(p_img_batch[:2]):
                        filename = f"epoch_{epoch:04d}_{scene_name}_idx_{idx}.jpg"
                        save_image(img_tensor, os.path.join(comp_dir, filename))

                # 5. Run frozen detector
                output = self.model(p_img_batch)

                # 6. Compute losses
                det_loss, max_prob_list = self.prob_extractor(
                    output, gts_batch,
                    loss_type=args.loss_type,
                    iou_thresh=args.train_iou,
                )

                tv_loss = self.tv_loss_fn(patch_c) * args.tv_eta
                
                # Weight the detection loss higher for 3D fine-tuning
                weighted_det_loss = det_loss * args.det_eta
                loss = weighted_det_loss + tv_loss

                # 7. Backprop and update patch
                patch_before = patch.detach().clone()
                loss.backward()
                grad_mean = patch.grad.abs().mean().item() if patch.grad is not None else 0.0
                optimizer.step()
                patch.data.clamp_(0, 1)
                delta_mean = (patch.detach() - patch_before).abs().mean().item()

                ep_det_loss += det_loss.item()
                ep_tv_loss += tv_loss.item()
                ep_total_loss += loss.item()
                n_batches += 1

                if i_batch % 5 == 0:
                    print(f"  [E{epoch} B{i_batch} {scene_name}] "
                          f"det={det_loss.item():.4f} tv={tv_loss.item():.4f} "
                          f"total={loss.item():.4f} grad={grad_mean:.6f} "
                          f"delta={delta_mean:.6f}")

                # Batch-level TensorBoard
                global_step = epoch * args.iters_per_epoch + i_batch
                writer.add_scalar('batch/det_loss', det_loss.item(), global_step)
                writer.add_scalar('batch/tv_loss', tv_loss.item(), global_step)
                writer.add_scalar('batch/total_loss', loss.item(), global_step)

                del composite_images, gts, p_img_batch, gts_batch, output
                del det_loss, tv_loss, loss
                gc.collect()

            # ── Epoch summary ──
            if n_batches > 0:
                ep_det_loss /= n_batches
                ep_tv_loss /= n_batches
                ep_total_loss /= n_batches

            et1 = time.time()
            print(f"\n  EPOCH {epoch}")
            print(f"  ├── DET LOSS:   {ep_det_loss:.4f}")
            print(f"  ├── TV LOSS:    {ep_tv_loss:.4f}")
            print(f"  ├── TOTAL LOSS: {ep_total_loss:.4f}")
            print(f"  └── TIME:       {et1 - et0:.1f}s\n")

            writer.add_scalar('epoch/det_loss', ep_det_loss, epoch)
            writer.add_scalar('epoch/tv_loss', ep_tv_loss, epoch)
            writer.add_scalar('epoch/total_loss', ep_total_loss, epoch)

            # ── Save patch snapshot ──
            if epoch % 10 == 0 or epoch == args.nepoch - 1:
                patch_save = patch.detach().clamp(0, 1).cpu()
                save_image(patch_save, os.path.join(args.save_path, f"patch_epoch_{epoch}.png"))

            torch.cuda.empty_cache()

        writer.close()
        print(f"\n✅ Fine-tuning complete. Patches saved to {args.save_path}/")


if __name__ == '__main__':
    print("AdvReal CCTV Fine-Tuning v1.0")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    parser = argparse.ArgumentParser(description='CCTV Adversarial Patch Fine-Tuning (3D-only)')

    # Paths
    parser.add_argument('-p', '--patch', type=str,
                        default='../advreal_data/patch_epoch_499.png',
                        help='Path to pre-trained patch to fine-tune from')
    parser.add_argument('-cfg', '--cfg', type=str,
                        default='configs/baseline/v11.yaml',
                        help='YOLOv11 config YAML path')
    parser.add_argument('--save_path', default='results/cctv_yolov11',
                        help='Output directory for patches and logs')

    # Training
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--nepoch', type=int, default=700,
                        help='Total number of epochs (fine-tuning continues from checkpoints)')
    parser.add_argument('--checkpoints', type=int, default=500,
                        help='Starting epoch index (patch was trained up to this point)')
    parser.add_argument('--iters_per_epoch', type=int, default=10,
                        help='Number of batch iterations per epoch (scenes randomly sampled)')
    parser.add_argument('--lr_decay', type=float, default=1.1)

    # Loss
    parser.add_argument('--det_eta', type=float, default=5.0, help='Detection loss weight (increase for 3D fine-tuning)')
    parser.add_argument('--tv_eta', type=float, default=2.5, help='TV loss weight')
    parser.add_argument('--loss_type', default='max_iou',
                        help='Loss type: max_iou, max_conf')
    parser.add_argument('--train_iou', type=float, default=0.45,
                        help='IoU threshold for matching GT boxes to detections')

    # NRSM
    parser.add_argument('--nrsm_range', type=float, default=0.02,
                        help='NRSM deformation magnitude')

    args = parser.parse_args()

    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Config:", args)
    trainer = CCTVPatchTrainer(args)
    trainer.train()
