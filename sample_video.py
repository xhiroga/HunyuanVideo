import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import torch

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from PIL import Image


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Prepare initial latent if init_image is provided
    init_latents = None
    if args.init_image is not None:
        init_image = Image.open(args.init_image).convert('RGB')
        init_image = init_image.resize(args.video_size[::-1])

        init_image_tensor = hunyuan_video_sampler.pipeline.image_processor.preprocess(init_image).to(hunyuan_video_sampler.device).to(torch.float16)
        init_image_tensor = init_image_tensor.unsqueeze(2)
        with torch.no_grad():
            init_latents = hunyuan_video_sampler.pipeline.vae.encode(init_image_tensor).latent_dist.sample()
            init_latents = hunyuan_video_sampler.pipeline.vae.config.scaling_factor * init_latents

    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        init_latents=init_latents,
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()
