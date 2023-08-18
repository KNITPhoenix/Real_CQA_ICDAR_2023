# checkpoint evaluation
name=VLT5

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
/home/feixu/anaconda3/envs/vlt5/bin/python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vqa_inference.py \
        --distributed --multiGPU \
        --test test \
        --num_workers 8 \
        --backbone 't5-base' \
        --output "/home/feixu/ChartQA/Models/VL-T5/output/predictions/" \
        --load "/home/feixu/ChartQA/Models/VL-T5/checkpoint/ftvlt5_pmc" \
        --num_beams 5 \
        --valid_batch_size 64 \
        --src_folder "/home/feixu/ChartQA/Models/VL-T5/data_chartqa/" \
        --raw_label \
        --fp16 \
        --use_vis_order_embedding True \
