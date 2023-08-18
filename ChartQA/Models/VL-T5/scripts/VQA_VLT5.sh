# experiment 1
name=VLT5
# original_batch_size = 32
# original_valid_batch_size = 64
output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
/home/spandey8/anaconda3/envs/vlt5/bin/python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vqa.py \
        --distributed --multiGPU \
        --train train \
        --valid validation \
        --test test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 10 \
        --num_workers 16 \
        --backbone 't5-base' \
        --output '/home/spandey8/ChartQA/Models/VL-T5/output/' \
        --load "/home/spandey8/ChartQA/Models/VL-T5/checkpoint/originalvlt5" \
        --num_beams 5 \
        --batch_size 12 \
        --valid_batch_size 64 \
        --src_folder "/home/spandey8/ChartQA/Models/VL-T5/data_chartqa/" \
        --raw_label \
        --fp16 \