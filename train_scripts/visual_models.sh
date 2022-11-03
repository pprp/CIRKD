#!/bin/bash 

# python test.py --model deeplabv3 --backbone resnet18 \
#     --method ours \
#     --save-dir ./save_testresults/ours_autokd_all_test_set \
#     --pretrained /home/inspur/project/DIST_KD/segmentation/work_dirs/dist_dv3-r101_dv3_r18/kd_deeplabv3_resnet18_citys_best_model.pth 


# python test.py --model deeplabv3 --backbone resnet101 \
#     --method tea_r101 \
#     --save-dir ./save_testresults/teacher_kd_all_test_set \
#     --pretrained /home/inspur/project/CIRKD/data/deeplabv3_resnet101_citys_best_model.pth

python test.py --model deeplab_mobile --backbone mobilenetv2 \
    --method mbv2 \
    --save-dir ./save_testresults/mobilev2_kd_all_test_set \
    --pretrained /home/inspur/project/CIRKD/data/kd_deeplab_mobile_mobilenetv2_citys_best_model.pth

# python test.py --model psp_mobile --backbone resnet18 \
#     --method psp_r18 \
#     --save-dir ./save_testresults/psp_kd_all_test_set \
#     --pretrained /home/inspur/project/CIRKD/data/kd_psp_resnet18_citys_best_model.pth