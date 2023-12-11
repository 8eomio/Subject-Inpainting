python scripts/inference_base.py \
--plms --outdir results/base \
--config configs/v1_base.yaml \
--ckpt ./models/Subject-Paint/base/2023-10-16T13-02-44_v1_base/checkpoints/last.ckpt \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path dataset/teddybear/images/00.jpg \
--seed 321 \
--scale 0.1