CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 48551 save_embeddings.py \
    --model psp_mobile \
    --backbone MobileNetV2 \
    --dataset citys \
    --data /home/inspur/data/cityscapes \
    --save-dir ./ \
    --pretrained ../data/kd_deeplab_mobile_mobilenetv2_citys_best_model.pth
    
# /home/inspur/project/DIST_KD/segmentation/work_dirs/dist_dv3-r101_dv3_r18/kd_deeplabv3_resnet18_citys_best_model.pth
    
python tsne_visual.py
