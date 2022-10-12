CUDA_VISIBLE_DEVICES=0,1,2 \
    python -m torch.distributed.launch --nproc_per_node=3 \
    train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --data /home/stack/data_sdc/cityscapes \
    --save-dir ./checkpoints \
    --log-dir ./logdirs \
    --max-iterations 80000 \
    --batch-size 8 \
    --lr 0.00375 \
    --teacher-pretrained ./data/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base ./data/resnet18-imagenet.pth
