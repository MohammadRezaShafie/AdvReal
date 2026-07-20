import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import pytorch3d
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
    TexturesUV
)
import pytorch3d_modify as p3dmd
import mesh_utils as MU
import os
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # 1. Load background
    bg_path = 'cctv_backgrounds/D01_20241010232332.jpg'
    bg_image = Image.open(bg_path).convert('RGB')
    transform = transforms.ToTensor()
    bg_tensor = transform(bg_image).to(device)

    # 2. Load meshes
    DATA_DIR = "./data"
    obj_filename_man = os.path.join(DATA_DIR, "Archive/Man_join/man.obj")
    obj_filename_tshirt = os.path.join(DATA_DIR, "Archive/tshirt_join/tshirt.obj")
    obj_filename_trouser = os.path.join(DATA_DIR, "Archive/trouser_join/trouser.obj")
    
    print("Loading meshes...")
    mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
    mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)
    mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=device)

    # 3. Setup Camera (Steep elevation for CCTV)
    dist = 2.5 # distance
    elev = 65.0 # Top-down CCTV angle (adjustable to match exact camera angle)
    
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    lights = AmbientLights(device=device)

    # 4. Render
    print("Rendering meshes with top-down CCTV perspective...")
    combined_full = MU.join_meshes([mesh_man, mesh_tshirt, mesh_trouser])
    source_coordinate = combined_full.verts_packed().to(device)
    
    composite_image = bg_tensor.clone()
    
    # We will render a few persons at different azimuths and locations
    placements = [
        {"pos": (600, 300), "azim": 45.0, "scale": 0.4},
        {"pos": (400, 500), "azim": -120.0, "scale": 0.6},
        {"pos": (900, 450), "azim": 15.0, "scale": 0.5}
    ]

    for p in placements:
        x1, y1 = p["pos"]
        azim = p["azim"]
        scale_factor = p["scale"]
        
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=45)
        
        images_predicted = p3dmd.view_mesh_wrapped(
            [mesh_man, mesh_tshirt, mesh_trouser],
            [None, None, None],
            [None, None, None], 
            source_coordinate,
            cameras=cameras, lights=lights, image_size=512, fov=45, max_faces_per_bin=30000, faces_per_pixel=3
        )
        
        rendered_person = images_predicted[0] # (512, 512, 4)
        
        # Crop transparent borders
        alpha_channel = rendered_person[..., 3]
        non_transparent = (alpha_channel > 0).float()
        
        if non_transparent.sum() == 0:
            continue
            
        min_y = torch.argmax(torch.any(non_transparent, dim=1).float())
        max_y = rendered_person.shape[0] - 1 - torch.argmax(torch.any(non_transparent.flip(dims=[0]), dim=1).float())
        min_x = torch.argmax(torch.any(non_transparent, dim=0).float())
        max_x = rendered_person.shape[1] - 1 - torch.argmax(torch.any(non_transparent.flip(dims=[1]), dim=0).float())
        
        cropped_rendered_person = rendered_person[min_y:max_y+1, min_x:max_x+1, :]
        cropped_rendered_person = cropped_rendered_person.permute(2, 0, 1) # (4, H, W)
        
        rendered_rgb = cropped_rendered_person[:3, :, :]
        rendered_alpha = cropped_rendered_person[3, :, :].unsqueeze(0)
        
        H_p, W_p = rendered_rgb.shape[1], rendered_rgb.shape[2]
        new_H, new_W = int(H_p * scale_factor), int(W_p * scale_factor)
        
        if new_H == 0 or new_W == 0:
            continue
            
        rendered_rgb = F.interpolate(rendered_rgb.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
        rendered_alpha = F.interpolate(rendered_alpha.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
        
        mask = (rendered_alpha > 0.5).float()
        
        x2 = x1 + new_W
        y2 = y1 + new_H
        
        if x2 > composite_image.shape[2] or y2 > composite_image.shape[1]:
            continue
            
        bg_crop = composite_image[:, y1:y2, x1:x2]
        composite_crop = mask * rendered_rgb + (1 - mask) * bg_crop
        composite_image[:, y1:y2, x1:x2] = composite_crop

    out_path = 'cctv_test_render.jpg'
    save_image(composite_image, out_path)
    print(f"✅ Saved composite test image to {out_path}")

if __name__ == '__main__':
    main()
