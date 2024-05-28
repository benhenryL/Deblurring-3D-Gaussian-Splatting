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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
import numpy as np
from utils.image_utils import psnr
from metrics import compute_img_metric


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    metric_psnr = []
    metric_ssim = []
    metric_lpips = []

    for idx, view in enumerate(tqdm(views, desc="Rendering and Evaluation progress")):
        # rendering
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        # evaluation
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt_image = torch.clamp(view.original_image, 0.0, 1.0)
        metric_psnr.append(psnr(rendering, gt_image).mean().double().item())
        rendering = rendering.permute(1,2,0)
        gt_image = gt_image.permute(1,2,0)
        metric_ssim.append(compute_img_metric(rendering, gt_image, 'ssim'))
        metric_lpips.append(compute_img_metric(rendering, gt_image, 'lpips').item())
    
    print(f"========={name} dataset=========")
    print("PSNR: ", sum(metric_psnr) / len(metric_psnr))
    print("SSIM: ", sum(metric_ssim) / len(metric_ssim))
    print("LPIPS: ", sum(metric_lpips) / len(metric_lpips))
    print(f"================================")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, 0)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=20000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
