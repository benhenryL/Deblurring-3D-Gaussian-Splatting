#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch.nn.functional as F

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, deblur=0, use_pos=False,
           lambda_s=0.01, lambda_p=0.01, max_clamp=1.1 ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else: 
        if not deblur:  # testing time
            scales = pc.get_scaling 
            rotations = pc.get_rotation 

            shs = None
            colors_precomp = None
            shs = pc.get_features
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
            
            return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

        else:
            scales = pc.get_scaling 
            rotations = pc.get_rotation
            shs = pc.get_features 
            colors_precomp = None
            M = pc.GTnet.num_moments

            _pos = means3D.detach()
            _scales = scales.detach()
            _rotations = rotations.detach()
            _viewdirs = viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)

            scales_delta, rotations_delta, pos_delta = pc.GTnet(_pos, _scales, _rotations, _viewdirs)
            scales_delta = torch.clamp(lambda_s * scales_delta + (1-lambda_s), min=1.0, max=max_clamp)
            rotations_delta = torch.clamp(lambda_s * rotations_delta + (1-lambda_s), min=1.0, max=max_clamp)

            if not use_pos:    # defocus 
                transformed_scales = scales * scales_delta
                transformed_rotations = rotations * rotations_delta

                rendered_image, radii = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = transformed_scales,
                    rotations = transformed_rotations,
                    cov3D_precomp = cov3D_precomp)
                
                return {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "visibility_filter" : radii > 0,
                        "radii": radii}
            
            elif use_pos:   # motion
                pos_delta = lambda_p * pos_delta
                pos_delta = pos_delta.view(-1, 3, M)
                scales_delta = scales_delta.view(-1, 3, M+1)
                rotations_delta = rotations_delta.view(-1, 4, M+1)

                pos = means3D
                transformed_scales = scales * scales_delta[...,-1]
                transformed_rotations = rotations * rotations_delta[...,-1]

                rendered_image, _radii = rasterizer(
                    means3D = pos,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = transformed_scales,
                    rotations = transformed_rotations,
                    cov3D_precomp = cov3D_precomp)

                # renders = [torch.clamp(rendered_image, min=0.0, max=1.0)]
                renders = [rendered_image]
                viewspace_points = [screenspace_points]
                visibility_filter = [_radii > 0]
                radii = [_radii]

                for i in range(M):
                    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
                    try:
                        screenspace_points.retain_grad()
                    except:
                        pass
                    means2D = screenspace_points
                    transformed_pos = means3D + pos_delta[...,i]
                    transformed_scales = scales * scales_delta[...,i]
                    transformed_rotations = rotations * rotations_delta[...,i]

                    rendered_image, _radii = rasterizer(
                        means3D = transformed_pos,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = colors_precomp,
                        opacities = opacity,
                        scales = transformed_scales,
                        rotations = transformed_rotations,
                        cov3D_precomp = cov3D_precomp)

                    # renders.append(torch.clamp(rendered_image, min=0.0, max=1.0))
                    renders.append(rendered_image)
                    viewspace_points.append(screenspace_points)
                    visibility_filter.append(_radii > 0)
                    radii.append(_radii)

                render = sum(renders) / len(renders)

                return {"render": render,
                    "viewspace_points": viewspace_points,
                    "visibility_filter" : visibility_filter,
                    "radii": radii}


