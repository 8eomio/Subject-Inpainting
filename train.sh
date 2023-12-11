python -u main.py \
--logdir models/Subject-Paint/barn \
--pretrained_model ./checkpoints/model.ckpt \
--base configs/v1_subject.yaml \
--scale_lr False \
--no-test True
