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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import configargparse
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import imageio
import numpy as np
from metrics import compute_img_metric
import torch.nn.functional as F

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, deblur=0):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, deblur)
    scene = Scene(dataset, gaussians)
    bbox = gaussians._xyz.amax(0) - gaussians._xyz.amin(0)

    gaussians.create_GTnet(hidden=opt.hidden, width=opt.width, pos_delta=opt.use_pos, num_moments=opt.num_moments)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if first_iter == opt.iterations:
            first_iter -= 1
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    viewpoint_stack = scene.getTrainCameras().copy()

    pts_max = gaussians._xyz.amax(0)
    pts_min = gaussians._xyz.amin(0)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        img_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(img_idx)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, deblur=deblur, use_pos=opt.use_pos, 
                            lambda_s=opt.lambda_s, lambda_p=opt.lambda_p, max_clamp=opt.max_clamp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        denom = 1 / len(visibility_filter) if type(radii) == list else 1.0
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                Ll2 = l2_loss(image, gt_image)
                psnr = (-10.0 * np.log(Ll2.cpu()) / np.log(10.0)).item()
                progress_bar.set_postfix({"PSNR": f"{psnr:.{2}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.model_path)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if type(visibility_filter) == list:
                    gaussians.max_radii2D[visibility_filter[0]] = torch.max(gaussians.max_radii2D[visibility_filter[0]], radii[0][visibility_filter[0]])
                else:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, denom)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_prune_threshold, scene.cameras_extent, size_threshold, opt.densify_with_depth, opt.prune_range)

                # Point addition
                if iteration == opt.pts_iter:
                    bbox = pts_max - pts_min
                    volume = bbox[0] * bbox[1] * bbox[2]
                    if opt.pts_rate > 0.0:
                        pts_N_pts = int(min(volume / (opt.pts_rate ** 3), 200000))
                    else:
                        pts_N_pts = opt.pts_N_pts
                    print(f"Allocate {pts_N_pts} points\n")

                    gaussians.add_points(training_args=opt, dist=opt.pts_dist, N=opt.pts_N_intpl, num_pts=pts_N_pts, bound=opt.pts_add_bound)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        tag = args.expname if args.expname != None else unique_str[0:10]
        args.model_path = os.path.join("./output/", tag)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    os.makedirs(args.model_path+"/TEST", exist_ok = True)
    os.makedirs(args.model_path+"/TRAIN", exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, savedir):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            _type = config["name"].upper()
            if _type == "TEST":
                with open(f"{savedir}/psnr.txt", "a") as f:
                    f.write("[ITER {}] NUM GAUSSIAN: {} \n".format(iteration, scene.gaussians.get_xyz.shape[0]))
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    image_metric = image.permute(1,2,0)
                    gt_image_metic = gt_image.permute(1,2,0)
                    ssim_test += compute_img_metric(image_metric, gt_image_metic, 'ssim')
                    lpips = compute_img_metric(image_metric, gt_image_metic, 'lpips')
                    if isinstance(lpips, torch.Tensor):
                        lpips = lpips.item()
                    lpips_test += lpips
                        
                    imageio.imwrite(f"{savedir}/{_type}/img_{iteration}_{idx:03d}.png", (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    if iteration == testing_iterations[0]:
                        imageio.imwrite(f"{savedir}/{_type}/GT_{idx:03d}.png", (gt_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])     
                lpips_test /= len(config['cameras'])    

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print("[ITER {}] Evaluating {}: SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], ssim_test, lpips_test))
                with open(f"{savedir}/psnr.txt", "a") as f:
                    f.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))
                    f.write("[ITER {}] Evaluating {}: SSIM {:.4f} LPIPS {:.4f}\n".format(iteration, config['name'], ssim_test, lpips_test))
                    
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[20_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--deblur', type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.deblur)

    # All done
    print("\nTraining complete.")



