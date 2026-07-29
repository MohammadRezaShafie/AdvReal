"""
CCTV Image Renderer for adversarial patch fine-tuning.

Renders 3D person meshes with adversarial textures at CCTV camera perspectives
and composites them onto CCTV background images.
"""
import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)
import pytorch3d_modify as p3dmd
import mesh_utils as MU
from NRSM import PrecomputedTPSDeformer


class CCTVImageRenderer:
    """
    3D person renderer for CCTV-perspective adversarial patch training.

    Renders person meshes with adversarial textures from steep top-down
    camera angles and composites them onto fixed CCTV background images
    with configurable placement jitter, occlusion masking, and NRSM deformation.
    """

    # ────────── CCTV scene definitions ──────────
    # Placement configs from the user-tuned render_cctv.py
    PLACEMENTS_D01 = [
        {"pos": (1060, 0), "azim": 45.0, "scale": 2.2, "dist": 2.8, "crop_top": 0.0, "crop_bottom": 0.0},
        {"pos": (600, 400), "azim": -120.0, "scale": 2.0, "dist": 2.2, "crop_top": 0.0, "crop_bottom": 0.0},
        {"pos": (1850, 300), "azim": -55.0, "scale": 2.0, "dist": 2.4, "crop_top": 0.0, "crop_bottom": 0.0},
    ]

    PLACEMENTS_D04 = [
        {"pos": (900, 120), "azim": 45.0, "scale": 2.2, "dist": 2.7, "crop_top": 0.0, "crop_bottom": 0.0},
        {"pos": (300, 350), "azim": -120.0, "scale": 2.0, "dist": 2.6, "crop_top": 0.0, "crop_bottom": 0.0},
        {"pos": (650, 100), "azim": -220.0, "scale": 2.0, "dist": 3.4, "crop_top": 0.0, "crop_bottom": 0.0},
        {"pos": (1300, 700), "azim": 140.0, "scale": 2.0, "dist": 2.0, "crop_top": 0.0, "crop_bottom": 0.0},
        {"pos": (1700, 0), "azim": 40.0, "scale": 2.0, "dist": 3.0, "crop_top": 0.0, "crop_bottom": 0.45},
    ]

    BG_PATH_D01 = "cctv_backgrounds/D01_20241010232332.jpg"
    BG_PATH_D04 = "cctv_backgrounds/D04_20241012110754.jpg"

    BASE_ELEV = 55.0  # Base CCTV camera elevation angle

    # Jitter ranges
    POS_JITTER = 30       # ±30 pixels
    AZIM_JITTER = 15.0    # ±15 degrees
    ELEV_JITTER = 5.0     # ±5 degrees

    # UV texture map dimensions (must match mesh UV layout from render.py)
    TSHIRT_H, TSHIRT_W = 340, 864
    TROUSER_H, TROUSER_W = 484, 700

    def __init__(self, device, nrsm_range=0.02, target_size=416):
        """
        Args:
            device: torch device (cuda/cpu)
            nrsm_range: NRSM deformation scale (default 0.02 from render.py)
            target_size: detector input size (default 416 for YOLO)
        """
        self.device = device
        self.nrsm_range = nrsm_range
        self.target_size = target_size
        self.DATA_DIR = "./data"

        # Load 3D meshes
        self._load_meshes()

        # Vertex index bookkeeping for NRSM deformation
        self.n_man = self.mesh_man.verts_packed().shape[0]
        self.n_tshirt = self.mesh_tshirt.verts_packed().shape[0]
        self.n_trouser = self.mesh_trouser.verts_packed().shape[0]
        self.idx_man = torch.arange(self.n_man, device=device)
        self.idx_tshirt = torch.arange(self.n_tshirt, device=device) + self.n_man
        self.idx_trouser = (
            torch.arange(self.n_trouser, device=device) + self.n_man + self.n_tshirt
        )

        # NRSM deformer (lazy init)
        self.deformer = None
        self.source_control_points = None
        self.original_verts = None

        # Lights (initial)
        self.lights = AmbientLights(device=device)

        # Load CCTV backgrounds
        self._load_backgrounds()

        # Build scene configs
        self.scenes = [
            {"name": "D01", "bg": self.bg_D01, "placements": self.PLACEMENTS_D01},
            {"name": "D04", "bg": self.bg_D04, "placements": self.PLACEMENTS_D04},
        ]

    # ────────── Mesh Loading ──────────

    def _load_meshes(self):
        """Load the 3D person, tshirt, and trouser meshes + UV metadata."""
        obj_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
        obj_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
        obj_trouser = os.path.join(self.DATA_DIR, "Archive/trouser_join/trouser.obj")

        self.mesh_man = load_objs_as_meshes([obj_man], device=self.device)
        self.mesh_tshirt = load_objs_as_meshes([obj_tshirt], device=self.device)
        self.mesh_trouser = load_objs_as_meshes([obj_trouser], device=self.device)

        # UV face/vertex indices for texture mapping (same var names as render.py)
        self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
        self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
        self.faces_trouser = self.mesh_trouser.textures.faces_uvs_padded()
        self.verts_uv_trouser = self.mesh_trouser.textures.verts_uvs_padded()

    def _load_backgrounds(self):
        """Load CCTV background images as tensors."""
        to_tensor = transforms.ToTensor()
        bg1 = Image.open(self.BG_PATH_D01).convert("RGB")
        bg4 = Image.open(self.BG_PATH_D04).convert("RGB")
        self.bg_D01 = to_tensor(bg1).to(self.device)
        self.bg_D04 = to_tensor(bg4).to(self.device)

    # ────────── Patch → UV Texture ──────────

    def _process_patch_four_positions(self, adv_patch_tensor, target_height, target_width, scale_factor=0.8):
        """Place adversarial patch at 4 tshirt UV positions.
        Exact replica of render.py._process_patch_four_positions.
        """
        positions = [(117, 198), (367, 198), (581, 207), (774, 207)]
        C, H_patch, W_patch = adv_patch_tensor.shape
        new_H_patch = max(1, int(H_patch * scale_factor))
        new_W_patch = max(1, int(W_patch * scale_factor))
        adv_patch_resized = F.interpolate(
            adv_patch_tensor.unsqueeze(0), size=(new_H_patch, new_W_patch),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        background = torch.ones(
            (C, target_height, target_width),
            device=adv_patch_tensor.device, dtype=adv_patch_tensor.dtype
        )
        for (center_x, center_y) in positions:
            start_y = center_y - new_H_patch // 2
            start_x = center_x - new_W_patch // 2
            background[:, start_y:start_y + new_H_patch, start_x:start_x + new_W_patch] = adv_patch_resized
        return background

    def _process_patch_six_positions(self, adv_patch_tensor, target_height, target_width, scale_factor=0.8):
        """Place adversarial patch at 6 trouser UV positions.
        Exact replica of render.py._process_patch_six_positions with bounds checking.
        """
        positions = [(54, 277), (219, 277), (414, 287), (627, 287), (133, 64), (515, 64)]
        C, H_patch, W_patch = adv_patch_tensor.shape
        new_H_patch = max(1, int(H_patch * scale_factor))
        new_W_patch = max(1, int(W_patch * scale_factor))
        patch_resized = F.interpolate(
            adv_patch_tensor.unsqueeze(0), size=(new_H_patch, new_W_patch),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        background = torch.ones(
            (C, target_height, target_width),
            dtype=adv_patch_tensor.dtype, device=adv_patch_tensor.device
        )
        for (center_x, center_y) in positions:
            start_x = center_x - (new_W_patch // 2)
            end_x = start_x + new_W_patch
            start_y = center_y - (new_H_patch // 2)
            end_y = start_y + new_H_patch
            # Clamp to canvas bounds (same logic as render.py)
            patch_start_x = 0
            patch_end_x = new_W_patch
            patch_start_y = 0
            patch_end_y = new_H_patch
            if start_x < 0:
                patch_start_x = -start_x
                start_x = 0
            if end_x > target_width:
                patch_end_x = new_W_patch - (end_x - target_width)
                end_x = target_width
            if start_y < 0:
                patch_start_y = -start_y
                start_y = 0
            if end_y > target_height:
                patch_end_y = new_H_patch - (end_y - target_height)
                end_y = target_height
            if patch_end_x > patch_start_x and patch_end_y > patch_start_y:
                background[:, start_y:end_y, start_x:end_x] = patch_resized[:, patch_start_y:patch_end_y, patch_start_x:patch_end_x]
        return background

    def set_adv_patch_texture(self, adv_patch_tensor):
        """Map the adversarial patch onto tshirt and trouser UV textures.
        Maintains gradient flow for backprop.
        Same logic as render.py set_adv_patch_texture (line 424-432).
        """
        if adv_patch_tensor.dim() == 4:
            adv_patch_tensor = adv_patch_tensor.squeeze(0)

        # Tshirt: 4 positions, scale=0.8 (same as render.py line 425)
        centered_patch_tshirt = self._process_patch_four_positions(
            adv_patch_tensor, self.TSHIRT_H, self.TSHIRT_W, scale_factor=0.8
        )
        # Trouser: 6 positions, scale=0.8 (same as render.py line 426)
        centered_patch_trouser = self._process_patch_six_positions(
            adv_patch_tensor, self.TROUSER_H, self.TROUSER_W, scale_factor=0.8
        )

        # Convert to TexturesUV format: (1, H, W, C)
        tex = centered_patch_tshirt.unsqueeze(0).permute(0, 2, 3, 1)
        tex_trouser = centered_patch_trouser.unsqueeze(0).permute(0, 2, 3, 1)

        self.mesh_tshirt.textures = TexturesUV(
            maps=tex, faces_uvs=self.faces, verts_uvs=self.verts_uv
        )
        self.mesh_trouser.textures = TexturesUV(
            maps=tex_trouser, faces_uvs=self.faces_trouser, verts_uvs=self.verts_uv_trouser
        )
        return tex, tex_trouser

    # ────────── NRSM Deformation ──────────

    def initialize_deformer(self, num_control_points=200):
        """Initialize TPS deformer for NRSM cloth deformation.
        Same logic as render.py initialize_deformer (lines 299-308).
        """
        print("🚀 Initializing NRSM deformer for CCTV renderer...")
        combined_full = MU.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])
        self.original_verts = combined_full.verts_packed().clone().to(self.device)

        combined_clothes = MU.join_meshes([self.mesh_tshirt, self.mesh_trouser])
        verts_np = combined_clothes.verts_packed().cpu().numpy()
        faces_np = combined_clothes.faces_packed().cpu().numpy()

        self.deformer = PrecomputedTPSDeformer(verts_np, faces_np)
        self.deformer.select_and_prepare(num_points=num_control_points)
        self.source_control_points = (
            torch.from_numpy(self.deformer.control_points_coords).float().to(self.device)
        )
        print("✅ NRSM deformer ready.")

    def _get_source_coordinates(self, use_nrsm=True):
        """Get vertex coordinates, optionally with NRSM deformation.
        Same deformation logic as render.py synthesis_image_person (lines 435-447).
        """
        if use_nrsm:
            if self.deformer is None:
                self.initialize_deformer()
            # Sample random displacements (same as render.py sample_nrsm_displacements)
            num_pts = self.source_control_points.shape[0]
            disp = torch.empty((num_pts, 3), device=self.device).uniform_(
                -self.nrsm_range, self.nrsm_range
            )
            # Apply deformation (same as render.py apply_deformation lines 311-322)
            deformed_clothes_np = self.deformer.deform(disp.detach().cpu().numpy())
            deformed_clothes = torch.from_numpy(deformed_clothes_np).float().to(self.device)
            n_t = self.n_tshirt
            deformed_tshirt = deformed_clothes[:n_t, :]
            deformed_trouser = deformed_clothes[n_t:, :]
            full = self.original_verts.clone()
            full[self.idx_tshirt] = deformed_tshirt
            full[self.idx_trouser] = deformed_trouser
            return full
        else:
            combined = MU.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])
            return combined.verts_packed().to(self.device)

    # ────────── Lighting ──────────

    def _sample_lights(self):
        """Randomly sample lighting conditions.
        Same distribution as render.py sample_lights (lines 244-262).
        """
        r = np.random.rand()
        theta = np.random.rand() * 2 * math.pi
        if r < 0.33:
            self.lights = AmbientLights(device=self.device)
        elif r < 0.67:
            ambient_intensity = np.random.uniform(0.6, 0.7)
            diffuse_intensity = np.random.uniform(0.4, 0.5)
            specular_intensity = np.random.uniform(0.3, 0.4)
            self.lights = DirectionalLights(
                device=self.device,
                direction=[[
                    np.sin(np.random.uniform(-np.pi, np.pi)),
                    np.sin(np.random.uniform(-np.pi / 2, np.pi / 2)),
                    np.cos(np.random.uniform(-np.pi, np.pi)),
                ]],
                ambient_color=((ambient_intensity, ambient_intensity, ambient_intensity),),
                diffuse_color=((diffuse_intensity, diffuse_intensity, diffuse_intensity),),
                specular_color=((specular_intensity, specular_intensity, specular_intensity),),
            )
        else:
            self.lights = PointLights(
                device=self.device,
                location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]],
            )

    # ────────── 3D Person Rendering ──────────

    def _render_person(self, azim, dist, elev):
        """Render a single 3D person with NRSM deformation at given camera params.
        
        Returns:
            (4, H, W) RGBA tensor with transparent borders cropped, or None if empty.
        """
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)

        source_coordinate = self._get_source_coordinates(use_nrsm=True)

        # Render using modified PyTorch3D (same call as render.py line 448-453)
        images_predicted = p3dmd.view_mesh_wrapped(
            [self.mesh_man, self.mesh_tshirt, self.mesh_trouser],
            [None, None, None],
            [None, None, None],
            source_coordinate,
            cameras=cameras,
            lights=self.lights,
            image_size=512,
            fov=45,
            max_faces_per_bin=30000,
            faces_per_pixel=3,
        )

        rendered = images_predicted[0]  # (512, 512, 4) RGBA

        # Crop transparent borders (same logic as render.py lines 455-464)
        alpha = rendered[..., 3]
        non_transparent = (alpha > 0).float()
        if non_transparent.sum() == 0:
            return None

        min_y = torch.argmax(torch.any(non_transparent, dim=1).float())
        max_y = torch.argmax(torch.any(non_transparent.flip(dims=[0]), dim=1).float())
        max_y = rendered.shape[0] - max_y
        min_x = torch.argmax(torch.any(non_transparent, dim=0).float())
        max_x = torch.argmax(torch.any(non_transparent.flip(dims=[1]), dim=0).float())
        max_x = rendered.shape[1] - max_x

        cropped = rendered[min_y:max_y + 1, min_x:max_x + 1, :]
        return cropped.permute(2, 0, 1)  # (4, H, W)

    # ────────── Compositing ──────────

    def _composite_person(self, rendered_rgba, bg_tensor, pos, scale, crop_top, crop_bottom):
        """Composite a rendered person onto a background image.
        
        Returns:
            (composite_image, gt_bbox) or (None, None) if out of bounds.
            composite_image: (3, bg_H, bg_W) tensor
            gt_bbox: (4,) long tensor [x1, y1, x2, y2] in full-resolution coords
        """
        rgb = rendered_rgba[:3, :, :]
        alpha = rendered_rgba[3:4, :, :]

        H_p, W_p = rgb.shape[1], rgb.shape[2]
        new_H = int(H_p * scale)
        new_W = int(W_p * scale)
        if new_H == 0 or new_W == 0:
            return None, None

        rgb = F.interpolate(
            rgb.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False
        ).squeeze(0)
        alpha = F.interpolate(
            alpha.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False
        ).squeeze(0)

        mask = (alpha > 0.5).float()

        # Apply occlusion cropping (counter/overhead masking)
        crop_top_px = int(new_H * crop_top) if crop_top > 0 else 0
        crop_bot_px = int(new_H * crop_bottom) if crop_bottom > 0 else 0
        if crop_top_px > 0:
            mask[0, :crop_top_px, :] = 0.0
        if crop_bot_px > 0:
            mask[0, (new_H - crop_bot_px):, :] = 0.0

        x1, y1 = pos
        x2 = x1 + new_W
        y2 = y1 + new_H

        bg_H, bg_W = bg_tensor.shape[1], bg_tensor.shape[2]
        if x2 > bg_W or y2 > bg_H or x1 < 0 or y1 < 0:
            return None, None

        # Composite: person over background (same pattern as render.py process_image_tensor)
        composite = bg_tensor.clone()
        bg_crop = composite[:, y1:y2, x1:x2]
        composite[:, y1:y2, x1:x2] = mask * rgb + (1 - mask) * bg_crop

        # GT bounding box adjusted for occlusion cropping
        gt_x1 = x1
        gt_y1 = y1 + crop_top_px
        gt_x2 = x2
        gt_y2 = y2 - crop_bot_px

        return composite, torch.tensor(
            [gt_x1, gt_y1, gt_x2, gt_y2], dtype=torch.long, device=self.device
        )

    # ────────── Main Generation API ──────────

    def generate_cctv_composite_batch(self):
        """Generate a batch of CCTV composite images with adversarial textures.

        Randomly selects a CCTV scene (D01 or D04), applies jitter to
        placement positions/azimuths/elevation, renders persons with NRSM
        deformation, and composites onto the background.

        Returns:
            composite_images: list of (3, target_size, target_size) tensors
            gts: list of (4,) long tensors [x1, y1, x2, y2] in target_size coords
            scene_name: str, name of selected scene ("D01" or "D04")
        """
        # Random scene selection
        scene = random.choice(self.scenes)
        bg_tensor = scene["bg"]
        placements = scene["placements"]
        bg_H, bg_W = bg_tensor.shape[1], bg_tensor.shape[2]

        composite_images = []
        gts = []

        # Sample lights for this batch
        self._sample_lights()

        for p in placements:
            # Apply jitter to prevent overfitting to exact pixel positions
            jx = random.randint(-self.POS_JITTER, self.POS_JITTER)
            jy = random.randint(-self.POS_JITTER, self.POS_JITTER)
            pos = (p["pos"][0] + jx, p["pos"][1] + jy)
            azim = p["azim"] + random.uniform(-self.AZIM_JITTER, self.AZIM_JITTER)
            elev = self.BASE_ELEV + random.uniform(-self.ELEV_JITTER, self.ELEV_JITTER)
            dist = p["dist"]
            scale = p["scale"]
            crop_top = p.get("crop_top", 0.0)
            crop_bottom = p.get("crop_bottom", 0.0)

            # Render 3D person with NRSM deformation
            rendered = self._render_person(azim, dist, elev)
            if rendered is None:
                continue

            # Composite onto background
            composite, gt = self._composite_person(
                rendered, bg_tensor, pos, scale, crop_top, crop_bottom
            )
            if composite is None:
                continue

            # Resize composite to detector input size (416x416)
            composite_resized = F.interpolate(
                composite.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            # Scale GT boxes to target_size coordinates
            sx = self.target_size / bg_W
            sy = self.target_size / bg_H
            gt_scaled = torch.tensor(
                [
                    int(gt[0].item() * sx),
                    int(gt[1].item() * sy),
                    int(gt[2].item() * sx),
                    int(gt[3].item() * sy),
                ],
                dtype=torch.long,
                device=self.device,
            )

            composite_images.append(composite_resized)
            gts.append(gt_scaled)

        return composite_images, gts, scene["name"]
