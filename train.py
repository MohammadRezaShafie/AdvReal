"""
Training code for Adversarial patch training
کد آموزش برای تولید وصله‌های متخاصم (Adversarial Patch)

This module implements physical adversarial patch generation for attacking object detection models.
این ماژول تولید وصله‌های متخاصم فیزیکی را برای حمله به مدل‌های تشخیص اشیاء پیاده‌سازی می‌کند.
"""
import ssl
import certifi
# SSL configuration for certificate verification
# پیکربندی SSL برای تایید گواهینامه
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.nn\.functional",
)
import sys
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import gc
from load_data import *
from transformers import DeformableDetrForObjectDetection
import torch
from torchvision import transforms
import torchvision
from tensorboardX import SummaryWriter
import pytorch3d as p3d
from pytorch3d.io import load_objs_as_meshes
from utils.parser import ConfigParser
from attack.attacker import UniversalAttacker
from utils.loader import dataLoader
from utils.parser import logger
from utils.plot import VisualBoard
from torch.utils.data import DataLoader
from utils.utils import *
import utils_camou
sys.path.append(os.path.abspath(''))
from arch.yolov3_models import YOLOv3Darknet
from yolo2.darknet import Darknet
from color_util import *
from render import ImageRenderer

def init(detector_attacker: UniversalAttacker, cfg: ConfigParser, data_root: str, args: object =None, log: bool =True):
    """Initialize the training environment including data loader, attacker, and logger.
    مقداردهی اولیه محیط آموزش شامل بارگذار داده، مهاجم و ثبت‌کننده.
    
    Args:
        detector_attacker: Universal attacker instance / نمونه مهاجم جهانی
        cfg: Configuration parser / تحلیل‌گر پیکربندی
        data_root: Root directory of training data / دایرکتوری اصلی داده‌های آموزش
        args: Command line arguments / آرگومان‌های خط فرمان
        log: Enable logging / فعال‌سازی ثبت رویدادها
    """
    if log: logger(cfg, args)

    data_sampler = None
    # Initialize data loader for person detection images
    # مقداردهی اولیه بارگذار داده برای تصاویر تشخیص افراد
    person_detection_loader = dataLoader(data_root,
                             input_size=cfg.DETECTOR.INPUT_SIZE, is_augment=cfg.DATA.AUGMENT,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler, shuffle=True,
                             num_workers=(args.num_workers if args and hasattr(args, 'num_workers') else 4))

    # Initialize the universal adversarial patch
    # مقداردهی اولیه وصله متخاصم جهانی
    detector_attacker.init_universal_patch(args.patch)
    detector_attacker.init_attaker()

    vlogger = None
    if log and args and not args.debugging:
        vlogger = VisualBoard(name=args.board_name, new_process=args.new_process,
                              optimizer=detector_attacker.attacker)
        detector_attacker.vlogger = vlogger

    return person_detection_loader, vlogger

def collate_fn(batch):
    return batch
def get_nuscenes_loader(img_dir, batch_size=4, shuffle=True, num_workers=2, transform=None):
    dataset = NuScenesDataset(img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return loader
# add path for demo utils functions 


class PatchTrainer(object):
    """Main trainer class for adversarial patch generation using 3D rendering.
    کلاس آموزش‌دهنده اصلی برای تولید وصله متخاصم با استفاده از رندر سه‌بعدی.
    
    This class combines 2D and 3D adversarial attacks by:
    این کلاس حملات متخاصم دوبعدی و سه‌بعدی را ترکیب می‌کند:
    1. Applying adversarial patches to 2D images / اعمال وصله‌های متخاصم به تصاویر دوبعدی
    2. Rendering 3D meshes with adversarial textures / رندر مش‌های سه‌بعدی با بافت‌های متخاصم
    3. Optimizing the patch to fool object detectors / بهینه‌سازی وصله برای فریب تشخیص‌دهنده‌های اشیاء
    """
    def __init__(self, args):
        self.args = args
        # Initialize 3D renderer for creating realistic adversarial examples
        # مقداردهی اولیه رندرر سه‌بعدی برای ایجاد نمونه‌های متخاصم واقع‌گرایانه
        self.renderer_v3 = ImageRenderer(args) 
        if args.device is not None:
            device = torch.device(args.device)
            torch.cuda.set_device(device)
        else:
            device = None
        self.device = device
        self.img_size = 416  # Standard YOLO input size / اندازه استاندارد ورودی YOLO
        self.DATA_DIR = "./data"

        # Load the target detection model based on architecture argument
        # بارگذاری مدل تشخیص هدف بر اساس آرگومان معماری
        if args.arch == "rcnn":
            # Faster R-CNN with ResNet-50 backbone / Faster R-CNN با پشتیبان ResNet-50
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        elif args.arch == "yolov3":
            # YOLOv3 Darknet architecture / معماری YOLOv3 Darknet
            self.model = YOLOv3Darknet().eval().to(device)
            self.model.load_darknet_weights('arch/weights/yolov3.weights')
        elif args.arch == "detr":
            self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).eval().to(
                device)
        elif args.arch == "deformable-detr":
            self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").eval().to(device)
        elif args.arch == "yolov2":
            self.model = Darknet('yolo2/cfg/yolov2.cfg').eval().to(device)
            self.model.load_weights('yolo2/yolov2.weights')
        elif args.arch == "yolov5":
            from detlib.HHDet.yolov5.api import HHYolov5
            cfg = ConfigParser("configs/baseline/v5.yaml")
            detector_cfg = cfg.DETECTOR 
            input_size = self.img_size
            self.model = HHYolov5(name="YOLOV5", 
                                cfg=detector_cfg,  
                                input_tensor_size=input_size,
                                device=device)
            model_weights = 'detlib/HHDet/yolov5/yolov5/weight/yolov5s.pt'
            model_config = 'detlib/HHDet/yolov5/yolov5/models_v5/yolov5s.yaml'
            self.model.load(model_weights, model_config=model_config)
            self.model.eval()
            self.model_parameters = self.model.parameters()
        elif args.arch == "yolov11":
            from detlib.HHDet.yolov11.api import HHYolov11
            cfg = ConfigParser("configs/baseline/v11.yaml")
            detector_cfg = cfg.DETECTOR
            input_size = self.img_size
            self.model = HHYolov11(
                name="YOLOV11",
                cfg=detector_cfg,
                input_tensor_size=input_size,
                device=device,
            )
            # default weights path
            model_weights = 'detlib/HHDet/yolov11/weights/yolo11s.pt'
            self.model.load(model_weights)
            self.model.eval()
            self.model_parameters = self.model.parameters()
        elif args.arch == "mask_rcnn":
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(device)
        else:
            raise NotImplementedError

        # Freeze detector model parameters - we only optimize the adversarial patch
        # انجماد پارامترهای مدل تشخیص‌دهنده - فقط وصله متخاصم بهینه می‌شود
        for p in self.model.parameters():
            p.requires_grad = False

        self.batch_size = args.batch_size

        # Initialize patch transformer for applying patches to images
        # مقداردهی اولیه تبدیل‌کننده وصله برای اعمال به تصاویر
        self.patch_transformer = PatchTransformer().to(device)
        # Initialize probability extractors specific to each detector architecture
        # مقداردهی اولیه استخراج‌کننده‌های احتمال مختص هر معماری تشخیص‌دهنده
        if args.arch == "rcnn":
            self.prob_extractor = MaxProbExtractor(0, 80).to(device)
        elif args.arch == "yolov2":
            self.prob_extractor = YOLOv2MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        elif args.arch == "yolov3":
            self.prob_extractor = YOLOv3MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        elif args.arch == "deformable-detr":
            self.prob_extractor = DeformableDetrProbExtractor(0,80,self.img_size).to(device)
        elif args.arch == "yolov5":
            self.prob_extractor = YOLOv5MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        elif args.arch == "yolov11":
            self.prob_extractor = YOLOv11MaxProbExtractor(0, 80, self.model, self.img_size).to(device)
        # Total Variation loss for patch smoothness regularization
        # تابع هزینه Total Variation برای منظم‌سازی نرمی وصله
        self.tv_loss = TotalVariation()

        # Load background images for compositing 3D rendered objects
        # بارگذاری تصاویر پس‌زمینه برای ترکیب اشیاء رندر شده سه‌بعدی
        self.background_loader = get_nuscenes_loader(
            img_dir='data/background_trans/background_train_resize',  # Modify according to your directory structure
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            transform=transforms.ToTensor()
        )

        self.epoch_length = len(self.background_loader)


        # Color transformation utility for realistic colorization
        # ابزار تبدیل رنگ برای رنگ‌آمیزی واقع‌گرایانه
        color_transform = ColorTransform('color_transform_dim6.npz')
        self.color_transform = color_transform.to(device)

        self.fig_size_H = 340
        self.fig_size_W = 864

        self.fig_size_H_t = 484
        self.fig_size_W_t = 700

        resolution = 4
        h, w, h_t, w_t = int(self.fig_size_H / resolution), int(self.fig_size_W / resolution), int(self.fig_size_H_t / resolution), int(self.fig_size_W_t / resolution)
        self.h, self.w, self.h_t, self.w_t = h, w, h_t, w_t

        # Load 3D mesh models (human, t-shirt, trouser) for adversarial texture application
        # بارگذاری مدل‌های مش سه‌بعدی (انسان، تی‌شرت، شلوار) برای اعمال بافت متخاصم
        # Set paths
        obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
        obj_filename_trouser = os.path.join(self.DATA_DIR, "Archive/trouser_join/trouser.obj")

        self.coordinates = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).to(device)
        self.coordinates_t = torch.stack(torch.meshgrid(torch.arange(h_t), torch.arange(w_t)), -1).to(device)

        self.colors = torch.load("data/camouflage4.pth").float().to(device)
        self.mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
        self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)
        self.mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=device)

        self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
        self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0]

        self.faces_trouser = self.mesh_trouser.textures.faces_uvs_padded()
        self.verts_uv_trouser = self.mesh_trouser.textures.verts_uvs_padded()
        self.faces_uvs_trouser = self.mesh_trouser.textures.faces_uvs_list()[0]

        # Class names for plotting detections (COCO80)
        try:
            self.class_names = utils_camou.load_class_names('configs/namefiles/coco80.names')
        except Exception:
            self.class_names = None


    def get_loader(self, img_dir, shuffle=True):
        loader = torch.utils.data.DataLoader(
            InriaDataset(img_dir, self.img_size, shuffle=shuffle),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        return loader

    def init_tensorboard(self, name=None):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        print(time_str)
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        fname = self.args.save_path.split('/')[-1]
        return SummaryWriter(f'{self.args.patch_save_dir}/{TIMESTAMP}_{fname}')


    def train(self):
        """
        Main training loop for optimizing adversarial patches.
        حلقه آموزش اصلی برای بهینه‌سازی وصله‌های متخاصم.
        
        This function combines 2D and 3D adversarial examples:
        این تابع نمونه‌های متخاصم دوبعدی و سه‌بعدی را ترکیب می‌کند:
        1. Patches on 2D person images / وصله‌ها روی تصاویر دوبعدی افراد
        2. 3D rendered meshes with adversarial textures / مش‌های سه‌بعدی رندر شده با بافت‌های متخاصم
        :return: Nothing
        """
        self.writer = self.init_tensorboard()
        # Ensure results directory exists for saving patches
        # اطمینان از وجود دایرکتوری نتایج برای ذخیره وصله‌ها
        os.makedirs(self.args.save_path, exist_ok=True)
        args = self.args
        
        et0 = time.time()
        checkpoints = args.checkpoints
        cfg = ConfigParser(args.cfg)
        # Initialize the universal attacker with the detector
        # مقداردهی اولیه مهاجم جهانی با تشخیص‌دهنده
        detector_attacker = UniversalAttacker(cfg, self.device)
        data_root = cfg.DATA.TRAIN.IMG_DIR
        person_detection_loader, vlogger = init(detector_attacker, cfg, args=args, data_root=data_root)
        # Get the adversarial patch and enable gradient computation
        # دریافت وصله متخاصم و فعال‌سازی محاسبه گرادیان
        patch = detector_attacker.universal_patch
        patch.requires_grad_(True)
        # Adam optimizer for patch parameters only
        # بهینه‌ساز Adam فقط برای پارامترهای وصله
        optimizer = optim.Adam([patch], lr=args.lr, amsgrad=True)

        self.writer = self.init_tensorboard()
        for epoch in tqdm(range(checkpoints, args.nepoch)):
            if epoch % 100 == 90:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / args.lr_decay
                print("Updated learning rate:", optimizer.param_groups[0]['lr'])
            ep_3d_det_loss = 0
            ep_patch_loss = 0
            ep_patch_det_loss = 0
            ep_patch_tv_loss = 0
            ep_loss = 0
            eff_count = 0
            eff_count_patch = 0  
            person_detection_iter = iter(person_detection_loader)
            for i_batch, bg_batch in enumerate(self.background_loader):
                optimizer.zero_grad()
                t0 = time.time()
                try:
                    
                    person_img_batch = next(person_detection_iter)
                    
                except StopIteration:
                    person_detection_iter = iter(person_detection_loader)
                    person_img_batch = next(person_detection_iter)

                detector_attacker.universal_patch.to(self.device)

                person_img_batch = person_img_batch.to(detector_attacker.device, non_blocking=True)

                all_preds = detector_attacker.detect_bbox(person_img_batch,self.args.save_path,)

                target_nums = detector_attacker.get_patch_pos_batch(all_preds)

                if sum(target_nums) == 0: continue
                patch_loss, patch_tv_loss, patch_det_loss = detector_attacker.attack(person_img_batch, mode='optim')
                eff_count_patch += 1

                patch_c = patch.clone()
                patch_c = patch_c.clamp(0, 1)
                self.renderer_v3.set_adv_patch_texture(patch_c)
                all_composite_images = []
                all_gts = []
                
                for bg_idx, bg_image_tensor in enumerate(bg_batch):
                    composite_images, gts = self.renderer_v3.generate_composite_image_tensor(bg_image_tensor)  # composite_images: List of tensors
                    all_composite_images.extend(composite_images) 
                    all_gts.extend(gts)
                p_img_batch = torch.stack(all_composite_images).to(self.device) 
                p_img_batch = p_img_batch[:, :3, :, :]
                gts_batch = torch.stack(all_gts).to(self.device)
                t1 = time.time()
                # Ensure inputs are normalized to [0,1] for detectors like YOLOv11
                p_img_batch = p_img_batch.float()
                # If composite images are in 0-255 range, bring to 0-1
                if torch.isfinite(p_img_batch).all() and p_img_batch.max() > 1.0:
                    p_img_batch = p_img_batch / 255.0
                p_img_batch = p_img_batch.clamp(0.0, 1.0)
                if epoch % 20 == 0:
                    if i_batch % 100 == 0: 
                        try:
                            from torchvision.utils import save_image
                            
                            # Create 'composite' directory inside the save_path
                            comp_dir = os.path.join(self.args.save_path, 'composite')
                            os.makedirs(comp_dir, exist_ok=True)
                            
                            # Save each image in the current batch
                            for img_idx, img_tensor in enumerate(p_img_batch):
                                if img_idx % 4 == 0:
                                    # Construct a unique filename: epoch_batch_index.png
                                    filename = f"epoch_{epoch:03d}_batch_{i_batch:04d}_idx_{img_idx}.jpg"
                                    save_path = os.path.join(comp_dir, filename)
                                    save_image(img_tensor, save_path)
                                
                        except Exception as e:
                            print(f"Warning: Failed to save composite image: {e}")
                            
                normalize = True
                if self.args.arch == "deformable-detr" and normalize:
                    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                    p_img_batch = normalize(p_img_batch)
                output = self.model(p_img_batch)
                t2 = time.time()

                det_loss, max_prob_list = self.prob_extractor(
                    output,
                    gts_batch,
                    loss_type=args.loss_type,
                    iou_thresh=args.train_iou
                )
                eff_count += 1
                loss = 0

                # if epoch < 100:
                #     loss = patch_det_loss + patch_tv_loss
                # else:
                #     loss = det_loss + patch_det_loss + patch_tv_loss

                loss += det_loss
                loss += patch_det_loss
                loss += patch_tv_loss


                ep_patch_loss += patch_loss.item()
                ep_patch_det_loss += patch_det_loss.item()
                ep_patch_tv_loss += patch_tv_loss.item()
                ep_3d_det_loss += det_loss.item()
                ep_loss += loss.item()
                
                # Backprop and update patch; monitor gradient and delta for debugging
                patch_before = patch.detach().clone()
                loss.backward()
                grad_mean = (patch.grad.abs().mean().item() if patch.grad is not None else 0.0)
                optimizer.step()
                patch.clamp(0, 1)
                delta_mean = (patch.detach() - patch_before).abs().mean().item()
                if i_batch % 10 == 0:
                    print(f"[Debug] grad|patch: mean={grad_mean:.6f}, delta_mean={delta_mean:.6f}")
                    
                if i_batch % 10 == 0:
                    global_step = epoch * len(self.background_loader) + i_batch
                    self.writer.add_scalar('batch/3D_DET_loss', det_loss.item(), global_step)
                    self.writer.add_scalar('batch/Total_loss', loss.item(), global_step)
                    self.writer.add_scalar('batch/2D_DET_loss', patch_det_loss.item(), global_step)
                    self.writer.add_scalar('batch/2D_TV_loss', patch_tv_loss.item(), global_step)
                del patch_loss, patch_det_loss, patch_tv_loss, det_loss, bg_batch, person_img_batch, all_composite_images, all_gts, p_img_batch, gts_batch, gts
                gc.collect()
                
            del patch_c, composite_images
            gc.collect()

            et1 = time.time()
            ep_patch_loss = ep_patch_loss / eff_count_patch
            ep_patch_det_loss = ep_patch_det_loss / eff_count_patch
            ep_patch_tv_loss = ep_patch_tv_loss / eff_count_patch
            ep_3d_det_loss = ep_3d_det_loss / eff_count
            ep_loss = ep_loss / eff_count
            
            print(' EPOCH: ', epoch),
            print("##################### AdvReal_2D #####################")
            print('2D DET LOSS: ', ep_patch_det_loss) 
            print(' 2D TV LOSS: ', ep_patch_tv_loss) 
            print("##################### AdvReal_3D #####################")
            print(' 3D DET LOSS: ', ep_3d_det_loss)
            print("#####################   AdvReal  #####################")
            print('  EPOCH TIME: ', et1 - et0)
            print('  EPOCH LOSS: ', ep_loss)
            # Save current adversarial patch snapshot each epoch
            
            if epoch % 10 == 0 or epoch == args.nepoch - 1:
                try:
                    patch_to_save = detector_attacker.universal_patch.detach().clamp(0,1).cpu()
                    from torchvision.utils import save_image
                    save_image(patch_to_save, os.path.join(self.args.save_path, f"patch_epoch_{epoch}.png"))
                except Exception as e:
                    print(f"Warning: failed to save patch image: {e}")
            self.writer.add_scalar('epoch/3D_DET_loss', ep_3d_det_loss, epoch)
            self.writer.add_scalar('epoch/2D_DET_loss', ep_patch_det_loss, epoch)
            self.writer.add_scalar('epoch/2D_TV_loss', ep_patch_tv_loss, epoch)
            self.writer.add_scalar('epoch/Total_loss', ep_loss, epoch)
            et0 = time.time()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    print('Version 2.0')
    print(torch.__version__)
    print(torch.version.cuda)
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--device', default='cuda:0', help='')
    parser.add_argument('--lr', type=float, default=0.03, help='')
    parser.add_argument('--lr_seed', type=float, default=0.01, help='')
    parser.add_argument('--nepoch', type=int, default=800, help='')
    parser.add_argument('--checkpoints', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers (set <= number of CPU cores)')
    parser.add_argument('--save_path', default='results/demo', help='')
    parser.add_argument("--tv_loss", type=float, default=1, help='tv loss weight')
    parser.add_argument("--real_loss", type=float, default=0.5, help='real loss weight')
    parser.add_argument("--patch_loss", type=float, default=0.5, help='patch loss weight')
    parser.add_argument("--lr_decay", type=float, default=1.1, help='')
    parser.add_argument("--lr_decay_seed", type=float, default=2, help='')
    parser.add_argument("--arch", type=str, default="yolov2", help='deformable-detr')
    parser.add_argument("--seed_type", default='fixed', help='')
    parser.add_argument("--clamp_shift", type=float, default=0, help='')
    parser.add_argument("--resample_type", default=None, help='')
    parser.add_argument("--tps2d_range_t", type=float, default=50.0, help='')
    parser.add_argument("--tps2d_range_r", type=float, default=0.1, help='')
    parser.add_argument("--tps3d_range", type=float, default=0.15, help='')
    parser.add_argument("--disable_tps2d", default=False, action='store_true', help='')
    parser.add_argument("--disable_tps3d", default=False, action='store_true', help='')
    parser.add_argument("--loss_type", default='max_iou', help='max_iou, max_conf, softplus_max, softplus_sum')
    parser.add_argument("--train_iou", type=float, default=0.45, help='')
    parser.add_argument("--mode", default='paper_obj', help='Patterns generated in adyolo')
    parser.add_argument("--patch_save_dir", default='demo', help='The generation path of the patch in adyolo')
    parser.add_argument('-cfg', '--cfg', type=str, default=os.path.join(os.getcwd(), 'configs/baseline/v2.yaml'), help="A relative path of the .yaml proj config file.")
    parser.add_argument('-p', '--patch', type=str, default='texture/heart.png', help="Start training with a given patch instead of random init. (for training from a break-point or for fine-tune)")
    parser.add_argument('-d', '--debugging', action='store_true', help="Will not start tensorboard process if debugging=True.")
    parser.add_argument('-sp', '--save_process', action='store_true', default=True, help="Save patches from intermediate epoches.")
    parser.add_argument('-n', '--board_name', type=str, default=None, help="Name of the Tensorboard as well as the patch name.")
    parser.add_argument('-np', '--new_process', action='store_true', default=False, help="Start new TensorBoard server process.")


    args = parser.parse_args()
    assert args.seed_type in ['fixed', 'random', 'variable', 'langevin']

    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Train info:", args)
    trainer = PatchTrainer(args)
    trainer.train()
