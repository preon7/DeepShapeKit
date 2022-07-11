import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    SoftSilhouetteShader, FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, PerspectiveCameras
)

from util.multiview_utils import get_cam


class Silhouette_Renderer():
    def __init__(self, size=(1040, 2048), deviation=torch.tensor([[0, 0, 0]]), device='cpu'):
        self.device = device
        # new renderer
        self.bottomR, self.bottomT = look_at_view_transform(1.95, 0, 0)  # bottom
        self.frontR, self.frontT = look_at_view_transform(1.8, 90, 180)  # front
        self.bottomR = self.bottomR.to(device)
        self.frontR = self.frontR.to(device)
        self.bottomT = self.bottomT.to(device) + torch.tensor([0.38,0.084,-0.], device=device)
        self.frontT = self.frontT.to(device) + torch.tensor([0.42, -0.05, 0], device=device) # front

        blend_params = BlendParams(sigma=1e-2, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=size,
            blur_radius=0.,  # np.log(1./1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=1)  # 00)

        # camera = FoVPerspectiveCameras(device=device, R=R, T=T)
        proj_m_set, focal, center, Rs, Ts, distortion = get_cam(device)
        camera = PerspectiveCameras(focal_length=focal[0], principal_point=((center[1, 0], center[1, 1]),), R=Rs[[1]],
                                    T=Ts[[1]], in_ndc=False, image_size=(size,), device=device)

        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

    def __call__(self, vertices, faces, t, dir):
        assert vertices.size(2) == 3 and faces.size(2) == 3, "shape of vertices or faces is not (N, v/f, 3)"
        if vertices.device != faces.device:
            faces = faces.to(vertices.device)

        vertices = vertices + t
        Rx_90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).to(self.device)
        Ry_90 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]).to(self.device)
        Rz_90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).to(self.device)
        # vertices = torch.einsum('bij,bkj->bki', Rx_90, vertices)
        vertices = vertices @ Ry_90 @ Ry_90 @ Rz_90

        fish_mesh = Meshes(verts=vertices,
                           faces=faces)

        if dir == 'front':
            silhouette = self.silhouette_renderer(meshes_world=fish_mesh.clone(), R=self.frontR, T=self.frontT)  # , R=R, T=T)
        elif dir == 'bottom':
            silhouette = self.silhouette_renderer(meshes_world=fish_mesh.clone(), R=self.bottomR, T=self.bottomT)

        return silhouette[..., 3] * 1.99