#!/bin/bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
./tools/dist_train.sh './configs/serialgfl/serialgfl_r50_fpn_1x_coco_[DGQP-c64].py' 8 --seed 100 --deterministic

#  --resume-from './work_dirs/serialgfl_r50_fpn_1x_coco_[DGQP-c64]/epoch_9.pth'
#   --seed 2022 --deterministic

#  --resume-from './work_dirs/serialgfl_r50_fpn_1x_coco_[DGQP-c64]/epoch_8.pth'


# export CUDA_VISIBLE_DEVICES='0,1,2,3'
# ./tools/dist_train.sh './configs/serialgfl/serialgfl_r50_fpn_1x_coco_[DGQP-c64].py' 2 --seed 100 --deterministic
# ./tools/dist_train.sh './configs/serialgfl/serialgfl_r50_fpn_1x_coco_[DGQP-c64].py' 4 --seed 100 --deterministic
#./tools/dist_train.sh './configs/serialgfl/serialgfl_r50_fpn_1x_coco_[DGQP-c64].py' 8 --seed 2022
#  --resume-from "./work_dirs/serialgfl_r50_fpn_1x_coco_[DGQP-c64]/result(0,8)/latest.pth"
#  --seed 100 --deterministic



# ./tools/dist_train.sh './configs/serialgfl/FAQE_r2n101_dcn_fpn_ms2x.py' 8 --seed 2022
# ./tools/dist_train.sh './configs/serialgfl/FAQE_r2n101_fpn_ms2x.py' 8 --seed 2022 --resume-from "work_dirs/FAQE_r2n101_fpn_ms2x/epoch_2.pth"

# ./tools/dist_train.sh configs/serialgfl/FAQE_r50_fpn_1x.py 8 --seed 2022

#./tools/dist_train.sh configs/serialgfl/FAQE_x101_dcn_fpn_ms2x.py 8 --seed 2022

# ./tools/dist_train.sh configs/serialgfl/FAQE_x101_32x4d_mdcn_fpn_last_ms2x.py 8 --seed 2022
# ./tools/dist_train.sh configs/serialgfl/FAQE_r2n101_dcn_fpn_last_ms2x.py 8 --seed 2022

# ./tools/dist_train.sh configs/serialgfl/FAQE_r101_fpn_ms2x.py 8 --seed 2022

# ./tools/dist_train.sh configs/serialgfl/serialgfl_r50_fpn_wogflv2_2loss_1x_coco.py 8 --seed 2022


# ./tools/dist_train.sh configs/serialgfl/FAQE_r101_mdcn_fpn_last_ms2x.py 8 --seed 2022


#./tools/dist_train.sh configs/serialgfl/FAQE_x101_64x4d_mdcn_fpn_last_ms2x.py 8 --seed 2022


# ./tools/dist_train.sh configs/ama/AMADet_r50_fpn_1x.py 8 --seed 2022


# ./tools/dist_train.sh './configs/serialgfl/FAQE_r50_mdcn_fpn_ms2x.py' 8 --deterministic



#  --resume-from './work_dirs/FAQE_r2n101_dcn_fpn_ms2x/epoch_14.pth'

# ./tools/dist_train.sh './configs/serialgfl/FAQE_x101_dcn_fpn_ms2x.py' 8 --deterministic


# ./tools/dist_train.sh './configs/serialgfl/FAQE_r50_fpn_ms2x.py' 8 --deterministic


#  --resume-from './work_dirs/FAQE_r2n101_dcn_fpn_ms2x/latest.pth'


#  --resume-from './work_dirs/serialgfl_r50_fpn_1x_coco_[DGQP-c64]/epoch_11.pth'


# ./tools/dist_train.sh './configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py' 8 --deterministic



# ./tools/dist_train.sh './configs/atss/atss_r50_fpn_1x_coco.py' 8 --deterministic

# ./tools/dist_train.sh 'configs/gflv2/gflv2_r50_fpn_1x_coco.py' 8 --deterministic
#  --resume-from './work_dirs/gflv2_r50_fpn_1x_coco/epoch_1.pth'



# ./tools/dist_train.sh 'configs/gfl/gfl_r50_fpn_1x_coco.py' 8 --deterministic

