import argparse
import os
import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import trange
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, SoftSilhouetteShader, FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, look_at_view_transform, PointLights,
    HardPhongShader, TexturesVertex, SoftPhongShader
)

def transform_verts(verts, scale):
    center = verts.mean(0)
    verts = verts - center
    if scale is None:
        scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    verts = verts @ Rx_90

    return verts, scale

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', nargs='+', default=list(range(0, 665)),
                        help='Index for an example in the multiview dataset')
    parser.add_argument('--modeldir', type=str, default='../data/output/multiview_demo/interp_models', help='path to saved models')
    parser.add_argument('--outdir', type=str, default='../data/output/multiview_demo/seq2video',
                        help='path of output')
    parser.add_argument('--mesh', type=str, default='goldfish_design_small.json', help='file of template fish')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Rx_90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).to(device)

    R, T = look_at_view_transform(2.7, 0, 110)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    R, T = look_at_view_transform(2.7, -90, 0)
    cameras2 = FoVPerspectiveCameras(device=device, R=R, T=T)

    image_size = 512
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[2.0, 2.0, 3.0]])

    renderer_1 = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    renderer_2 = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras2,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    out = cv2.VideoWriter(os.path.join(args.outdir, 'smooth_model_seq_{}-{}_(1).mp4'.format(args.index[0], args.index[-1])), cv2.VideoWriter_fourcc(*'mp4v'), 20, (image_size * 2, image_size))

    print('start rendering video')
    scale = None

    pbar = trange(len(args.index), desc="creating video")
    for i in args.index:
        model_name = str(i) + '_out_model_A.obj'
        verts, faces, aux = load_obj(os.path.join(args.modeldir, model_name))
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)

        verts, scale = transform_verts(verts, scale)

        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        fish_mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

        images_top = renderer_1(fish_mesh)
        images_side = renderer_2(fish_mesh)
        images = torch.cat([images_top, images_side], dim=2)
        out.write((images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8))
        #out.write(images[0, ..., :3].cpu().numpy())
        #plt.figure(figsize=(10, 10))
        #plt.imsave('../data/output/multiview_demo/video_render_demo.png', images[0, ..., :3].cpu().numpy())
        # img = cv2.imread('../data/output/multiview_demo/video_render_demo.png')
        #plt.imshow()
        #plt.axis("off");
        pbar.update(1)

    out.release()
    print('done')

