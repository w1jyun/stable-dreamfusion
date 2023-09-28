# suppress partial model loading warning
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd 
# ControlNet
from cldm.cldm import ControlledUnetModel, ControlLDM
from ldm.models.autoencoder import AutoencoderKL
# from diffusers import DDIMScheduler
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from dreamtime import Timestep
from safetensors.torch import load_file

from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from .perpneg_utils import weighted_perpendicular_aggregator
from pathlib import Path

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype) 

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class ControlNet(nn.Module):
    def __init__(self, device, iters, sd_version='2.1', hf_key=None):
        super().__init__()
        self.device = device
        torch.cuda.empty_cache()
        model = create_model('./models/cldm_v15.yaml').cpu()
        model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
        model = model.cuda()
        self.model = model
        self.vae = AutoencoderKL.init_from_ckpt('./models/control_sd15_scribble.pth')
        self.ddim_sampler = DDIMSampler(model)
        self.num_train_timesteps = 1000
        self.a_prompt = None
        self.iteration = 0
        self.index = 0
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.ddim_sampler.make_schedule(ddim_num_steps=self.num_train_timesteps, ddim_discretize="uniform", ddim_eta=0., verbose=True)
        self.Timestep = Timestep(num_of_timestep = self.num_train_timesteps - 1, num_of_iters = iters)
        
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning([prompt]).detach()
        return embeddings
    
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        # random = torch.rand(x.size(0), device=x.device)
        # prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        # input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
        # null_prompt = self.get_learned_conditioning([""])

        # # z.shape: [8, 4, 64, 64]; c.shape: [8, 1, 768]
        # # print('=========== xc shape ===========', xc.shape)
        # with torch.enable_grad():
        #     clip_emb = self.get_learned_conditioning(xc).detach()
        #     cond["c_crossattn"] = [self.cc_projection(torch.cat([torch.where(prompt_mask, null_prompt, clip_emb), T[:, None, :]], dim=-1))]
        # cond["c_concat"] = [input_mask * self.encode_first_stage((xc.to(self.device))).mode().detach()]
        # out = [z, cond]
        # if return_first_stage_outputs:
        #     xrec = self.decode_first_stage(z)
        #     out.extend([x, xrec])
        # if return_original_cond:
        #     out.append(xc)
        # return out

    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    # text_z, uncond, pred_rgb, control, guidance_scale=self.opt.guidance_scale, save_guidance_path=save_guidance_path)
    def train_step(self, text_z, uncond, pred_rgb, control, epoch, guidance_scale=100, save_guidance_path:Path=None):
        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.tensor([self.Timestep.timestep((epoch - 1) * 100 + self.index % 100)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.ddim_sampler.add_noise(latents, noise, t)
                # model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
                cond = {"c_concat": [control], "c_crossattn": [text_z]}
                un_cond = {"c_concat": [control], "c_crossattn": [uncond]}
                x_in = torch.cat([latents_noisy])
                noise_pred = self.ddim_sampler.model.apply_model(x_in, t, cond)
                e_t_uncond = self.ddim_sampler.model.apply_model(x_in, t, un_cond)
                noise_pred = e_t_uncond + guidance_scale * (noise_pred - e_t_uncond)

        w = (1 - self.ddim_sampler.ddim_alphas[t])
        grad = w * (noise_pred - noise)

        grad = torch.nan_to_num(grad)
        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        if (save_guidance_path):
            with torch.no_grad():
                pred_minus_noise = self.decode_latents(noise_pred - noise)
                arr = [pred_rgb_512, control, pred_minus_noise]
                viz_images = torch.cat(arr,dim=-1)
                save_image(viz_images, save_guidance_path)
        self.index += 1
        loss = SpecifyGradient.apply(latents, grad) 
        return loss 

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, control, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):
        # interp to 512x512 to be fed into vae.
        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts       

        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512)
        t = torch.tensor([self.Timestep.timestep((epoch - 1) * 100 + self.index % 100)], dtype=torch.long, device=self.device)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.ddim_sampler.add_noise(latents, noise, t)
                cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning(prompt + ', ' + part)]}
                un_cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning(n_prompt)]}
                x_in = torch.cat([latents_noisy])
                noise_pred = self.ddim_sampler.model.apply_model(x_in, t, cond)
                e_t_uncond = self.ddim_sampler.model.apply_model(x_in, t, un_cond)
                noise_pred = e_t_uncond + guidance_scale * (noise_pred - e_t_uncond)
                noise_pred = e_t_uncond + guidance_scale * weighted_perpendicular_aggregator(noise_pred - e_t_uncond, weights, B)
        w = (1 - self.ddim_sampler.ddim_alphas[t])
        grad = w * (noise_pred - noise)

        grad = torch.nan_to_num(grad)
        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        if (save_guidance_path):
            with torch.no_grad():
                pred_minus_noise = self.decode_latents(noise_pred - noise)
                arr = [pred_rgb_512, control, pred_minus_noise]
                viz_images = torch.cat(arr,dim=-1)
                save_image(viz_images, save_guidance_path)
        self.index += 1
        loss = SpecifyGradient.apply(latents, grad) 
        return loss 

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            text_embeddings.shape[0] * 1,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 8. Denoising loop
        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]


                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents
        

    def decode_latents(self, latents):
        # zs: [B, 4, 32, 32] Latent space image
        # with self.model.ema_scope():
        imgs = self.ddim_sampler.model.decode_first_stage(latents)  
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs # [B, 3, 256, 256] RGB space image

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256] RGB space image
        # with self.model.ema_scope():
        imgs = imgs * 2 - 1
        latents = self.model.first_stage_model.encode(imgs).sample() * 0.18215

        return latents # [B, 4, 32, 32] Latent space image
    
    def encode_imgs_by_sd(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents
    
    def prompt_to_img(self, prompts, images, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        print('prompt_to_img')
        return 0
if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    cn = ControlNet(device, opt.sd_version, opt.hf_key)
    import os
    from PIL import Image
    file_list = os.listdir('./pose2')
    for file_name in file_list:
        input_image = Image.open('./pose2' + '/' + file_name)
        imgs = cn.prompt_to_img(opt.prompt, images, opt.negative, opt.H, opt.W, opt.steps)
        # visualize image
        plt.imshow(imgs[0])
        plt.show()




