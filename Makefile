run:
	.venv/bin/python sample_video.py \
    --dit-weight ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt \
    --video-size 848 480 \
    --video-length 73 \
    --infer-steps 20 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --use-fp8 \
    --save-path ./results \
	--init-image ./assets/Persian_cat.jpeg


install: .venv ckpts/hunyuan-video-t2v-720p ckpts/text_encoder ckpts/text_encoder_2
	.venv/bin/pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install ninja
	.venv/bin/pip install wheel
	.venv/bin/pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
	.venv/bin/pip install xfuser==0.4.0
	.venv/bin/pip install "huggingface_hub[cli]"

.venv:
	python3.10 -m venv .venv

ckpts/hunyuan-video-t2v-720p:
	.venv/bin/huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts

ckpts/text_encoder:
	.venv/bin/huggingface-cli download Kijai/llava-llama-3-8b-text-encoder-tokenizer --local-dir $@

ckpts/text_encoder_2:
	.venv/bin/huggingface-cli download openai/clip-vit-large-patch14 --local-dir $@
