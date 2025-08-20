import argparse
import random
from omegaconf import OmegaConf

from dataset_interface import Dataset_Objaverse_captions
from src.mvdream.ldm.util import instantiate_from_config
from src.mvdream.ldm.models.diffusion.ddim import DDIMSampler
from src.mvdream.model_zoo import build_model
from src.mvdream.camera_utils import get_camera, get_camera_objaverse

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import DDPMScheduler
from diffusers.training_utils import compute_snr


#####################################################################################################
#####################################################################################################
def parsing(mode="args"):
    parser = argparse.ArgumentParser()

    ### Dataset details
    # parser.add_argument('--dataset_name', type=str, default="Thingi10k", help='Dataset path')
    # parser.add_argument('--output_s3_bucket', type=str, default="build3d-training-output", help='bucket path')
    # parser.add_argument('--dataset_path', nargs='+' ,  default=["/data/dataset/Shapenet_D2", "/data/dataset/BuildingNet_v", "/data/dataset/ModelNet40_v",
    #  "/data/dataset/Thingi10k_v", "/data/dataset/obj_data"], help='Dataset path')
    # parser.add_argument('--dataset_files', nargs='+', default=[], help='Dataset path')
    parser.add_argument(
        "--dataset_folders",
        nargs="+",
        default=["Objaverse_wavelet_latents_fixed"],
        help="Dataset path",
    )
    # parser.add_argument('--test_dataset_folders', nargs='+', default=None, help='Dataset path')
    # parser.add_argument('--use_dataset_folder', help="use dataset_folder", action="store_true")
    # parser.add_argument('--pre_defined_scale', nargs='+', default=None, type=float, help='pre-defined scale')
    # parser.add_argument('--pre_defined_avg', nargs='+', default=None, type=float, help='pre-defined avg')
    # parser.add_argument('--use_sample_training', help="use sample_training", action="store_true")
    # parser.add_argument('--use_sample_threshold', help="use sample_threshold", action="store_true")
    # parser.add_argument('--sample_threshold_ratio', type=float, default=0.0625, help='point_num')
    # parser.add_argument('--use_batched_threshold', help="use use_batched_threshold", action="store_true")
    # parser.add_argument('--update_stage_every', type=int, default=50000, help='update_stage_every')
    # parser.add_argument('--use_s3_data', help="use s3_data", action="store_true")
    # parser.add_argument('--use_ray', help="use ray", action="store_true")
    # parser.add_argument('--max_failures', type=int, default=0, help='update_stage_every')
    # parser.add_argument('--restore_s3_exp', type=str, default=None, help="restore_s3_exp to load")
    # parser.add_argument('--matual_precision', type=str, default=None, help="matual_precision to load")
    # parser.add_argument('--strategy', type=str, default=None, help="strategy for training")
    # parser.add_argument('--sanity_test', help="use ray", action="store_true")
    parser.add_argument(
        "--use_local_storage", type=bool, default=True, help="use use_local_storage"
    )
    # parser.add_argument('--exp_prefix', type=str, help="experiment prefix")
    # parser.add_argument('--data_sanity_test', help="use data_sanity_test", action="store_true")
    # parser.add_argument('--use_h5', help="use h5", action="store_true")
    parser.add_argument(
        "--use_compact_indices", type=bool, default=True, help="use use_compact_indices"
    )
    # parser.add_argument('--exception_return_none', help="use exception_return_none", action="store_true")
    # parser.add_argument('--use_s3_dataset', help="use use_s3_dataset", action="store_true")
    # parser.add_argument('--max_concurrency', type=int, default=200, help='maximum number of concurrent request')
    # parser.add_argument('--multipart_size', type=int, default=8388608, help='maximum part size')
    # parser.add_argument('--use_efs_storage', help="use_efs_storage", action="store_true")
    # parser.add_argument('--check_image_exist', help="check_image_exist", action="store_true")
    # parser.add_argument('--use_separate_bucket', help="use_separate_bucket", action="store_true")
    # parser.add_argument('--s3_img_separated_bucket', help="s3_img_separated_bucket", action="store_true")
    # parser.add_argument('--low_avg', type=float, default=2.20, help='low_avg')
    # parser.add_argument('--use_normalize_std', help="use_normalize_std", action="store_true")
    # parser.add_argument('--std', nargs='+', default=None, type=float, help='std')
    # parser.add_argument('--use_camera_index', help="use_camera_index", action="store_true")
    # parser.add_argument('--use_all_views', help="use_all_views", action="store_true")
    # parser.add_argument('--input_view_cnt', type=int, default=40, help='number of input views')
    # parser.add_argument('--use_multithread_downloading', help="use_multithread_downloading", action="store_true")
    # parser.add_argument('--use_image_features_only', help="use_image_features_only", action="store_true")
    # parser.add_argument('--use_multiple_views_inferences', help='use_multiple_views_inferences', action='store_true')
    # parser.add_argument('--use_multiple_views_grids', help='use_multiple_views_grids', action='store_true')
    # parser.add_argument('--use_eps_dp', help='use_eps_dp', action='store_true')
    parser.add_argument("--fill_blank_img", help="fill_blank_img", action="store_true")
    # parser.add_argument('--test_split_type', type=str, default='test', help='test_split_type')
    # parser.add_argument('--use_test_shuffle', help='use_test_shuffle', action='store_false')
    # parser.add_argument('--testing_cnt', default=None, type=int, help='testing_cnt')
    # parser.add_argument('--test_exp_name', default=None, type=str, help='test_exp_name')
    # parser.add_argument('--save_top_k', default=1, type=int, help='save_top_k')
    # parser.add_argument('--testing_views',  nargs='+', default=None, type=int, help='testing_views')
    parser.add_argument("--val_cnt", default=None, type=int, help="val_cnt")
    parser.add_argument("--use_even_val", help="use_even_val", action="store_true")
    parser.add_argument(
        "--testing_img_bucket", default=None, type=str, help="testing_img_bucket"
    )
    # parser.add_argument('--use_mask_inference', help='use_mask_inference', action='store_true')
    # parser.add_argument('--gradient_clip_val', default=0.0, type=float, help='gradient_clip_val')
    # parser.add_argument('--use_ema', help='use_ema', action='store_true')
    # parser.add_argument('--ema_decay', default=0.9999, type=float, help='ema_decay')
    # parser.add_argument('--use_ema_weights', help='use_ema_weights', action='store_false')
    # parser.add_argument('--use_loss_resample', help='use_loss_resample', action='store_true')
    # parser.add_argument('--debug_base_folder', type=str, default='dump', help='debug_base_folder')
    # parser.add_argument('--use_dummpy_wavelet', help='use_dummpy_wavelet', action='store_true')
    parser.add_argument(
        "--camera_path",
        type=str,
        default="/home/ubuntu/workspace/MVDream/data",
        help="path to camera files",
    )

    # ## use point conditions
    # parser.add_argument('--use_pointcloud_conditions', help="use use_pointcloud_conditions", action="store_true")
    # parser.add_argument('--pc_point_num', type=int, default=25000, help='pc_dims')
    # parser.add_argument('--use_pc_samples', help="use_pc_samples", action="store_true")
    # parser.add_argument('--sample_num', type=int, default=2048, help='sample_num')
    # parser.add_argument('--pc_dims', type=int, default=1024, help='pc_dims')
    # parser.add_argument('--num_inds', type=int, default=256, help='num_inds')
    # parser.add_argument('--pc_output_dim', type=int, default=128, help='pc_output_dim')
    # parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
    # parser.add_argument('--use_pointvoxel_encoder', help="use_pointvoxel_encoder", action="store_true")

    # ## use voxel conditions
    # parser.add_argument('--use_voxel_conditions', help="use use_voxel_conditions", action="store_true")
    # parser.add_argument('--voxel_context_dim', type=int, default=128, help='context voxel dim')
    # parser.add_argument('--voxel_dim', type=int, default=8, help='context voxel dim')
    # parser.add_argument('--voxel_resolution', type=int, default=64, help='context voxel dim')

    # ## use image_conditionals
    # parser.add_argument('--use_image_conditions', help="use use_image_conditions", action="store_true")
    # parser.add_argument('--s3_bucket', type=str, default="large-dataset", help='s3 images_base')
    parser.add_argument(
        "--s3_wavelet_bucket",
        type=str,
        default="new-filtered-wavelet-data",
        help="s3 images_base",
    )
    # parser.add_argument('--s3_voxels_bucket', type=str, default="voxels-data", help='s3 images_base')
    # parser.add_argument('--s3_clip_bucket', type=str, default="image-clips-features", help='s3 clip_bucket')
    # parser.add_argument('--s3_pointclouds_bucket', type=str, default="pointclouds-data", help='s3 clip_bucket')
    # parser.add_argument('--s3_prefix', type=str, default="dataset", help='s3 prefix')
    parser.add_argument(
        "--render_resolution", type=int, default=384, help="rendering resolution"
    )
    parser.add_argument(
        "--max_images_num", type=int, default=55, help="maximum number of images"
    )
    parser.add_argument("--n_px", type=int, default=256)

    # ## other setting for Image conditions
    # parser.add_argument('--num_mapping_layers', type=int, default=0, help='rendering resolution')
    # parser.add_argument('--cond_mapping_type', type=str, default=None, help="with_layernorm or not")
    # parser.add_argument("--clip_model_type", type=str, default='L-14', metavar='N', help='what model to use')
    # parser.add_argument('--use_noised_clip_features', help="use noised_clip_features", action="store_true")
    # parser.add_argument("--noise_cond_para", default=1.0, type=float, help="hyperparameter for noise_cond_para")

    # ## Optimizer setting
    # parser.add_argument("--lr", default=5e-5, type=float, help="hyperparameter for lr")
    # parser.add_argument("--beta1", default=0.9, type=float, help="hyperparameter for div beta1 (ADAM)")
    # parser.add_argument("--beta2", default=0.999, type=float, help="hyperparameter for beta1 (ADAM)")
    # parser.add_argument("--precision", type=str, default='16-mixed', help='what precision')

    # ### training details
    # parser.add_argument('--train_mode', type=str, default="train", help='train or test')
    # # parser.add_argument('--seed', type=int, default=1, help='Seed')
    # parser.add_argument('--epochs', type=int, default=300, help="Total epochs")
    # parser.add_argument('--checkpoint', type=str, default=None, help="Checkpoint to load")
    # parser.add_argument('--restore_path', type=str, default=None, help="Restore Path to load")
    # parser.add_argument('--use_timestamp', action='store_true', help='Whether to use timestamp in dump files')
    # parser.add_argument('--num_iterations', type=int, default=300000, help='How long the training shoulf go on')
    # parser.add_argument('--gpu', nargs='+', default="0", help='GPU list')
    # parser.add_argument('--optimizer', type=str, choices=('SGD', 'Adam'), default='Adam')
    # parser.add_argument('--batch_size', type=int, default=64, help='Dimension of embedding')
    # parser.add_argument('--test_batch_size', type=int, default=32, help='Dimension of embedding')
    # parser.add_argument('--threshold', type=float, default=0.45, help='Max Norm value')
    # parser.add_argument('--sampling_type', type=str, default=None, help='what sampling type: None--> Uniform')
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    # parser.add_argument('--cpu_per_worker', type=int, default=1, help='Number of workers')
    # parser.add_argument('--gpu_workers', type=int, default=8, help='Number of gpu workers')
    # parser.add_argument('--prefetch_factor', type=int, default=None, help='prefetch_factor')
    # parser.add_argument('--use_wandb', help='use_wandb', action='store_true')
    # parser.add_argument('--wandb_key', type=str, default=None, help='key for wandb')
    # parser.add_argument('--every_n_epochs', type=int, default=50, help='Number of epoch to save checkpoint')

    # ### diffusion setting
    # parser.add_argument('--diffusion_beta_schedule', type=str, default='linear', help='diffusion_beta_schedule')
    # parser.add_argument('--diffusion_step', type=int, default=1000, help='diffusion_step')
    # parser.add_argument('--diffusion_rescale_timestep', type=int, default=1000, help='diffusion_rescale_timestep')
    # parser.add_argument('--diffusion_scale_ratio', type=float, default=1.0, help='diffusion_scale_ratio')
    # parser.add_argument('--diffusion_model_var_type', type=str, default='FIXED_SMALL', help='diffusion_model_var_type')
    # parser.add_argument('--diffusion_model_mean_type', type=str, default='EPSILON', help='diffusion_model_mean_type')
    # parser.add_argument('--diffusion_loss_type', type=str, default='MSE', help='diffusion_loss_type')
    # parser.add_argument('--diffusion_sampler', type=str, default='second-order', help='diffusion_sampler')

    # ### diffusion unet setting
    # parser.add_argument('--unet_model_channels', type=int, default=64, help='unet_model_channels')
    # parser.add_argument('--with_self_att', help='with_self_att', action='store_true')
    # # parser.add_argument('--unet_num_res_blocks', type=int, default=3, help='unet_num_res_blocks')
    # parser.add_argument('--unet_num_res_blocks', nargs='+', default=[3], type=int, help='unet_num_res_blocks')
    # parser.add_argument('--unet_channel_mult', nargs='+', default=[1, 2, 2, 4], type=int, help='unet_channel_mult_low')
    # parser.add_argument('--attention_resolutions', nargs='+', default=[], type=int, help='unet_channel_mult_low')
    # parser.add_argument("--dp_cond", type=float, default=None, help='should we dropout condition or not')
    # parser.add_argument("--scale", type=float, default=3, help='scale for dp stuff')
    # parser.add_argument("--guidance_type", type=str, default=None, help='None or probs or logprobs')
    # parser.add_argument("--dp_cond_type", type=str, default=None, help='none or learnable')

    # ### UVIT
    # parser.add_argument('--network_type', type=str, default='UNET', help='network_type')
    # parser.add_argument('--num_transformer_blocks', type=int, default=8, help='Number of transformer in middle')
    # parser.add_argument('--add_num_register', type=int, default=0, help='Number of register in middle')
    # parser.add_argument('--learnable_skip_r', type=int, default=None, help='Number of learnable_skip in middle')
    # parser.add_argument('--add_condition_time_ch', type=int, default=None, help='Number of condition_time_ch in middle')
    # parser.add_argument('--add_condition_input_ch', type=int, default=None, help='Number of condition_input_ch in middle')

    # ### wavelet setting
    parser.add_argument("--resolution", type=int, default=256, help="resolution")
    parser.add_argument("--max_depth", type=int, default=3, help="max_depth")
    parser.add_argument("--max_training_level", type=int, default=2, help="max_depth")
    parser.add_argument("--point_num", type=int, default=16384, help="point_num")
    # parser.add_argument('--keep_level', type=int, default=2, help='keep_level')
    # parser.add_argument('--data_keep_level', type=int, default=2, help='data_keep_level')
    parser.add_argument("--wavelet", type=str, default="bior6.8", help="wavelet")
    parser.add_argument(
        "--padding_mode", type=str, default="constant", help="padding_mode"
    )
    # parser.add_argument('--use_normalization', help="use min max normalization", action="store_true")
    # parser.add_argument('--use_shift_mean', help="use shift_mean", action="store_true")
    # parser.add_argument('--start_stage', type=int, default=0, help='start_stage')
    # parser.add_argument('--use_adaptive_stage_update', help="use adaptive_stage_update", action="store_true")
    # parser.add_argument('--no_rebalance_loss', help="use no_rebalance_loss", action="store_true")

    # ### Logging details
    # parser.add_argument('--print_every', type=int, default=50, help='Printing the loss every')
    # parser.add_argument('--save_every', type=int, default=50, help='Saving the model every')
    # parser.add_argument('--validation_every', type=int, default=500, help='validation set every')
    # parser.add_argument('--visualization_every', type=int, default=500, help='visualization of the results every')
    # parser.add_argument("--log-level", type=str, choices=('info', 'warn', 'error'), default='info')
    # parser.add_argument('--experiment_type', type=str, default="max", help='experiment type')
    # parser.add_argument('--experiment_every', type=int, default=5, help='experiment every ')
    # # parser.add_argument('--fast_dev_run', type=helper.bool_flag, default=False, help='fast_dev_run mode or not ')
    # parser.add_argument('--limit_val_batches', type=float, default=1.0, help='limit_val_batches')
    # parser.add_argument('--compile_model', help="use compilation of model", action="store_true")
    # parser.add_argument('--superres_checkpoint', type=str, default=None, help='superres_checkpoint')
    # parser.add_argument('--superres_type', type=str, choices=('D1', 'Latent'), default='D1', help='superres_type')
    # parser.add_argument('--testing_samples', nargs='+', default=None, type=str, help='pre-defined avg')

    ### LOADING MODEL details
    parser.add_argument(
        "--model_name",
        type=str,
        default="sd-v2.1-base-4view",
        help="load pre-trained model from hugginface",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="load model from local config (override model_name)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="path to local checkpoint"
    )
    parser.add_argument(
        "--num_frames", type=int, default=1, help="num of frames (views) to generate"
    )
    parser.add_argument("--fp16", action="store_true")

    ### TEST details
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=50)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--size", type=int, default=256)

    ### SETTING
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", type=str, default="cuda")

    ### FINETUNING details
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )  # 16
    # parser.add_argument(
    #     "--dataloader_num_workers",
    #     type=int,
    #     default=0,
    #     help=(
    #         "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    #     ),
    # )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0.1,
        help="The scale of input perturbation. Recommended 0.1.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )

    if mode == "args":
        args = parser.parse_args()
        return args
    else:
        return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_conditioning(model, camera, prompt, uc, device, dtype, num_frames=1):
    assert type(prompt) == list
    with torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c}
        uc_ = {"context": uc}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

    return c_, uc_


def t2i(
    model,
    image_size,
    prompt,
    uc,
    sampler,
    step=20,
    scale=7.5,
    batch_size=8,
    ddim_eta=0.0,
    dtype=torch.float32,
    device="cuda",
    camera=None,
    num_frames=1,
):
    if type(prompt) != list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None,
        )
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


def main():
    args = parsing()

    # Settings:
    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = max(4, args.num_frames)
    set_seed(args.seed)

    # Load pre-trained model:
    print("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
    model.device = device
    model.to(device)

    sampler = DDIMSampler(model)
    uc = model.get_learned_conditioning([""]).to(device)
    print("load t2i model done . ")

    #### test forward pass
    # camera = get_camera(args.num_frames, elevation=args.camera_elev, azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
    # camera = camera.repeat(batch_size//args.num_frames,1).to(device)

    # t = args.text + args.suffix

    # set_seed(args.seed)
    # images = []
    # for j in range(3):
    #     img = t2i(model, args.size, t, uc, sampler, step=50, scale=10, batch_size=batch_size, ddim_eta=0.0,
    #             dtype=dtype, device=device, camera=camera, num_frames=args.num_frames)
    #     img = np.concatenate(img, 1)
    #     images.append(img)
    # images = np.concatenate(images, 0)
    # Image.fromarray(images).save(f"test.png")
    ###

    noise_scheduler = DDPMScheduler()

    vae = model.first_stage_model
    text_encoder = model.cond_stage_model

    unet = model.model

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    weight_dtype = torch.float32
    vae.to(weight_dtype)
    text_encoder.to(weight_dtype)
    unet.train()

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset creation:
    train_dataset = Dataset_Objaverse_captions(args)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    # Pre-compute camera matrices:
    all_cameras = get_camera_objaverse(args.camera_path).to(device)
    # all_cameras = all_cameras.repeat(batch_size//args.num_frames,1).to(device)
    # all_cameras[data['img_idx']]

    for epoch in range(args.num_train_epochs):
        # train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            # Convert images to latent space
            latents = vae.encode(batch["images"].to(device, weight_dtype)).sample()
            # latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )
            if args.input_perturbation:
                new_noise = noise + args.input_perturbation * torch.randn_like(noise)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if args.input_perturbation:
                noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            # encoder_hidden_states = text_encoder(batch["caption"])[0]
            camera = all_cameras[batch["img_idx"]]
            prompt = batch["caption"]
            cond, _ = get_conditioning(model, camera, prompt, uc, device, dtype)

            # Get the target for loss depending on the prediction type
            if args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Predict the noise residual and compute loss
            # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            model_pred = model.apply_model(noisy_latents, timesteps, cond)

            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (
                    torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    / snr
                )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # # Gather the losses across all processes for logging (if we use distributed training).
            # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            # train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # # Backpropagate
            # accelerator.backward(loss)
            loss.backward()
            # if accelerator.sync_gradients:
            #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            print(epoch, step, loss.detach().item())

    #### Test dataset
    # for data in dataset:
    #     x_sample = torch.clamp((data['images'] + 1.0) / 2.0, min=0.0, max=1.0)
    #     x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
    #     images = list(x_sample.astype(np.uint8))

    #     images = np.concatenate(images, 0)
    #     Image.fromarray(images).save(f"data_test.png")

    #     print(data['caption'])

    #     breakpoint()


if __name__ == "__main__":
    main()
