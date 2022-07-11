import torch
#import matplotlib.pyplot as plt

from animal_model import fish_model
from .losses import camera_fitting_loss, body_fitting_loss, mask_fitting_loss  #, kpts_fitting_loss
#from util import multiview_utils as mutil

class OptimizeMV():

    def __init__(self, lim_weight=1, prior_weight=1, bone_weight=1, mask_weight=1, smooth_weights=None,
                 step_size=1e-2,
                 num_iters=100,
                 device=torch.device('cpu'), mesh='carp.json'):

        # Store optimization hyperparameters
        if smooth_weights is None:
            smooth_weights = [1., 1., 1.]
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.lim_weight = lim_weight
        self.prior_weight = prior_weight
        self.bone_weight = bone_weight
        self.mask_weight = mask_weight
        self.smooth_weights = smooth_weights

        self.fish = fish_model.fish_model(device=device, mesh=mesh)
        self.faces = self.fish.faces

    def __call__(self, init_pose, init_bone, init_t, scale, proj_m, keypoints, masks, silhouette_renderer, has_prev=False, img_filenames=None, index=None, bboxs=None):
        """Perform multiview reconstruction
        Input:
            model:
            init_pose: (1, 12*3) initial pose parameters
            init_bone: (1, 11) initial bone length
            init_t: (1, 3) initial translation
            scale: (1,) initial scale

            multiview:
            proj_m: (n_view, 3, 4) multiview camera projection matrix
            keypoints: (n_view, 6, 3) keypoints with confidence

        """

        # Number of views
        batch_size = proj_m.shape[0]

        # Unbind keypoint location and confidence
        keypoints_2d = keypoints[:, :, :2]
        keypoints_conf = keypoints[:, :, -1]
        kpts_conf = torch.ones_like(keypoints_conf)

        # disable some keypoints
        kpts_conf[0,-3] = 0
        kpts_conf[1,-1] = 0

        # Copy all initialization
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        global_t = init_t.detach().clone()
        bone_length = init_bone.detach().clone()
        scale = scale.detach().clone()

        # pose vertices to render silhouette
        # Rx_90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]).to(self.device)
        # Ry_90 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]).to(self.device)
        # Rz_90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).to(self.device)

        # Step 1: Optimize global orientation, translation and scale
        body_pose.requires_grad = False
        bone_length.requires_grad = False
        global_orient.requires_grad = True
        global_t.requires_grad = True
        if has_prev:
            scale.requires_grad = True
        else:
            scale.requires_grad = True

        silhouette_t = torch.zeros((2, 3), device=self.device)
        silhouette_t.requires_grad = True

        gloabl_opt_params = [global_orient, global_t, scale]
        gloabl_optimizer = torch.optim.Adam(gloabl_opt_params, lr=self.step_size, betas=(0.9, 0.999))


        for i in range(self.num_iters):
            fish_output = self.fish(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length,
                                    scale=scale)
            model_keypoints = fish_output['keypoints'] + global_t.repeat(1, 1, 1)
            model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

            loss = camera_fitting_loss(model_keypoints, proj_m,
                                       keypoints_2d, kpts_conf) + \
                   (global_t - init_t).abs().sum() * self.prior_weight

            # silhouette_front = silhouette_renderer(
            #     fish_output['vertices'] @ Ry_90 @ Rz_90 + silhouette_t[0, :],
            #     self.fish.faces.unsqueeze(0), torch.tensor([0,0,0]))  # torch.tensor([-0.015,0,0]))
            # silhouette_bottom = silhouette_renderer(
            #     fish_output['vertices'] @ Ry_90 @ Ry_90 @ Rz_90 + silhouette_t[1, :],
            #     self.fish.faces.unsqueeze(0), torch.tensor([0,0,0]))  # torch.tensor([-0.025,0,0]))
            silhouette_front = silhouette_renderer(fish_output['vertices'], self.fish.faces.unsqueeze(0), global_t,
                                                   'front')
            silhouette_bottom = silhouette_renderer(fish_output['vertices'], self.fish.faces.unsqueeze(0), global_t,
                                                   'bottom')

            mask_loss = mask_fitting_loss(torch.cat([silhouette_front, silhouette_bottom], 0), masks.float(),
                                          0.1 * self.mask_weight)
            # silhouette_bottom = silhouette_renderer(
            #     fish_output['vertices'] @ Ry_90 @ Ry_90 @ Rz_90,  # for goldfish2.json
            #     self.fish.faces.unsqueeze(0), torch.tensor([-0.025,0,0])) # silhouette_t[1, :])
            # mask_loss = mask_fitting_loss(silhouette_bottom, masks[[1]].float(), self.mask_weight)

            loss += mask_loss

            gloabl_optimizer.zero_grad()
            loss.backward()
            gloabl_optimizer.step()

        # visualize step results
        # if img_filenames is not None:
        #     for i in range(len(img_filenames)):
        #         frame = [i]
        #         img = plt.imread(img_filenames[i]) * 255
        #         img_pose, img_model = mutil.render_vertex_on_frame(img, [fish_output['vertices'].cpu()+ global_t.cpu()], self.fish, frame, [keypoints.cpu()],
        #                                                            bboxs[i])
        #         # img_pose, img_model = mutil.render_vertex_on_frame(img, [debug_vert], fish, frame, [keypoints], bboxs[i])
        #
        #         plt.imsave('data/output/multiview_demo' + '/step_images/{}_view_{}_step1.png'.format(index, i), img_pose)

        # Step 2: Optimize all parameters
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        global_t.requires_grad = True
        if has_prev:
            scale.requires_grad = True
            bone_length.requires_grad = True
        else:
            scale.requires_grad = True
            bone_length.requires_grad = True

        body_opt_params = [body_pose, bone_length, global_orient, global_t, scale]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        ### fit without tail
        kpts_conf = torch.ones_like(keypoints_conf)
        # adjust keypoint weights
        kpts_conf[0, :] = torch.ones_like(kpts_conf[1, :]) * 0.8
        kpts_conf[1, :] = torch.ones_like(kpts_conf[1, :]) * 0.8
        kpts_conf[1, 0] = 1
        kpts_conf[:, -3] = 0
        # disable some keypoints
        #kpts_conf[0, -3] = 0
        kpts_conf[1, -1] = 0

        # pose_init = body_pose.detach().clone()
        # bone_init = bone_length.detach().clone()

        for i in range(self.num_iters):
            fish_output = self.fish(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length,
                                    scale=scale)
            model_keypoints = fish_output['keypoints'] + global_t.repeat(1, 1, 1)
            model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

            loss = body_fitting_loss(model_keypoints, proj_m,
                                     keypoints_2d, kpts_conf, body_pose, bone_length,
                                     lim_weight=self.lim_weight, prior_weight=self.prior_weight,
                                     bone_weight=self.bone_weight,)
                                     #pose_init=pose_init, bone_init=bone_init)

            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # body_optm_loss = loss.item()
        # body_kpt_loss = kpts_fitting_loss(model_keypoints, proj_m,
        #                                   keypoints_2d, kpts_conf, body_pose, bone_length)

        # visualize step results
        # if img_filenames is not None:
        #     for i in range(len(img_filenames)):
        #         frame = [i]
        #         img = plt.imread(img_filenames[i]) * 255
        #         img_pose, img_model = mutil.render_vertex_on_frame(img, [fish_output['vertices'].cpu()+ global_t.cpu()], self.fish, frame,
        #                                                            [keypoints.cpu()],
        #                                                            bboxs[i])
        #         # img_pose, img_model = mutil.render_vertex_on_frame(img, [debug_vert], fish, frame, [keypoints], bboxs[i])
        #
        #         plt.imsave('data/output/multiview_demo' + '/step_images/{}_view_{}_step2.png'.format(index, i), img_pose)

        # train with tail
        kpts_conf = torch.ones_like(keypoints_conf)

        kpts_conf[0,:] = torch.ones_like(kpts_conf[1,:]) * 0.8
        kpts_conf[1, :] = torch.ones_like(kpts_conf[1, :]) * 0.8
        kpts_conf[1, 0] = 1
        # disable some keypoints
        kpts_conf[0, -3] = 0.2
        kpts_conf[1, -1] = 0

        body_pose.requires_grad = True
        global_orient.requires_grad = True
        global_t.requires_grad = True
        if has_prev:
            scale.requires_grad = True
            bone_length.requires_grad = True
        else:
            scale.requires_grad = True
            bone_length.requires_grad = True

        silhouette_t.requires_grad = True

        body_opt_params = [body_pose, bone_length, global_orient, global_t, scale, silhouette_t]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        pose_init = body_pose.detach().clone()
        bone_init = bone_length.detach().clone()



        for i in range(self.num_iters):
            fish_output = self.fish(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length,
                                    scale=scale)
            model_keypoints = fish_output['keypoints'] + global_t.repeat(1, 1, 1)
            model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

            # posed_fish = Meshes(fish_output['vertices'], self.faces.unsqueeze(0).to(self.device))
            # deformed = posed_fish.offset_verts(deform_verts)

            loss = body_fitting_loss(model_keypoints, proj_m, #cam_rot, cam_t, focal_length, camera_center,
                                     keypoints_2d, kpts_conf, body_pose, bone_length, sigma=100,
                                     lim_weight=self.lim_weight, prior_weight=self.prior_weight,
                                     bone_weight=self.bone_weight, #distortion=distortion,
                                     pose_init=pose_init, bone_init=bone_init)

            # silhouette_front = silhouette_renderer(fish_output['vertices'] @ Ry_90 @ Rz_90,
            #                                        self.fish.faces.unsqueeze(0), torch.tensor([0,0,0]))  # torch.tensor([-0.015,0,0]))# silhouette_t[0,:])
            # silhouette_bottom = silhouette_renderer(fish_output['vertices'] @ Ry_90 @ Ry_90 @ Rz_90,
            #                                        self.fish.faces.unsqueeze(0), torch.tensor([0,0,0]))  # torch.tensor([-0.025,0,0])) #silhouette_t[1,:])
            #mask_loss = mask_fitting_loss(silhouette_bottom, masks[[1]].float(), self.mask_weight)

            silhouette_front = silhouette_renderer(fish_output['vertices'], self.fish.faces.unsqueeze(0), global_t,
                                                   'front')
            silhouette_bottom = silhouette_renderer(fish_output['vertices'], self.fish.faces.unsqueeze(0), global_t,
                                                    'bottom')
            mask_loss = mask_fitting_loss(torch.cat([silhouette_front, silhouette_bottom], 0), masks.float(),
                                          self.mask_weight)
            loss = loss + mask_loss

            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # tail_optm_loss = loss.item()
        # tail_kpt_loss = kpts_fitting_loss(model_keypoints, proj_m,
        #                                   keypoints_2d, kpts_conf, body_pose, bone_length)

        # optimize global position after posing
        body_pose.requires_grad = False
        bone_length.requires_grad = False
        global_orient.requires_grad = True
        global_t.requires_grad = True
        scale.requires_grad = True

        gloabl_opt_params = [global_orient, global_t, scale]
        gloabl_optimizer = torch.optim.Adam(gloabl_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(100):
            fish_output = self.fish(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length,
                                    scale=scale)
            model_keypoints = fish_output['keypoints'] + global_t.repeat(1, 1, 1)
            model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

            loss = camera_fitting_loss(model_keypoints, proj_m,
                                       keypoints_2d, kpts_conf) + \
                   (global_t - init_t).abs().sum() * self.prior_weight

            gloabl_optimizer.zero_grad()
            loss.backward()
            gloabl_optimizer.step()

        # plt.imsave('data/output/multiview_demo/mask_bottom_debug.png', masks[1].detach().cpu().numpy())
        # plt.imsave('data/output/multiview_demo/silhouette_bottom_debug.png', silhouette_bottom[0].detach().cpu().numpy())
        # plt.imsave('data/output/multiview_demo/mask_front_debug.png', masks[0].detach().cpu().numpy())
        # plt.imsave('data/output/multiview_demo/silhouette_front_debug.png',
        #            silhouette_front[0].detach().cpu().numpy())

        #print("body optimization loss: {}; tail optimization loss: {}".format(body_optm_loss, tail_optm_loss))

        # Output
        #vertices = deformed.verts_packed().unsqueeze(0).detach().cpu()
        vertices = fish_output['vertices'].detach().cpu()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach().cpu()
        bone = bone_length.detach().cpu()
        scale = scale.detach().cpu()
        global_t = global_t.detach().cpu()

        return vertices, pose, bone, scale, global_t, (0,0)
