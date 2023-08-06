

# tools/dist_test.sh configs/serialgfl/FAQE_r2n101_dcn_fpn_ms2x.py work_dirs/FAQE_r2n101_dcn_fpn_ms2x/epoch_24.pth 8 --format-only --options "jsonfile_prefix=./FAQE_r2n101_dcn_fpn_ms2x_results_tta"

# tools/dist_test.sh configs/serialgfl/FAQE_r50_fpn_ms2x.py work_dirs/FAQE_r50_fpn_ms2x/epoch_24-466.pth 8 --format-only --options "jsonfile_prefix=./FAQE_r50_fpn_ms2x_results"

# tools/dist_test.sh configs/serialgfl/FAQE_x101_32x4d_mdcn_fpn_last_ms2x.py work_dirs/FAQE_x101_32x4d_mdcn_fpn_last_ms2x/epoch_23.pth 8 --format-only --options "jsonfile_prefix=./FAQE_x101_32x4d_mdcn_fpn_last_ms2x_results"

# tools/dist_test.sh configs/serialgfl/FAQE_r101_fpn_ms2x.py work_dirs/FAQE_r101_fpn_ms2x/epoch_24.pth 8 --format-only --options "jsonfile_prefix=./FAQE_r101_fpn_ms2x_results"

# tools/dist_test.sh configs/serialgfl/FAQE_r2n101_dcn_fpn_last_ms2x_tta.py work_dirs/FAQE_r2n101_dcn_fpn_last_ms2x/epoch_24.pth 8 --format-only --options "jsonfile_prefix=./FAQE_r2n101_dcn_fpn_last_ms2x_tta_results"


# tools/dist_test.sh configs/serialgfl/FAQE_r50_fpn_ms2x.py work_dirs/FAQE_r50_fpn_ms2x/epoch_24-466.pth 8 --eval bbox

# tools/dist_test.sh configs/serialgfl/FAQE_x101_64x4d_mdcn_fpn_last_ms2x.py work_dirs/FAQE_x101_64x4d_mdcn_fpn_last_ms2x/epoch_20.pth 8 --format-only --options "jsonfile_prefix=./FAQE_x101_64x4d_mdcn_fpn_last_ms2x_results"
# tools/dist_test.sh configs/serialgfl/FAQE_x101_64x4d_mdcn_fpn_last_ms2x_tta.py work_dirs/FAQE_x101_64x4d_mdcn_fpn_last_ms2x/epoch_20.pth 8 --format-only --options "jsonfile_prefix=./FAQE_x101_64x4d_mdcn_fpn_last_ms2x_tta_results"

# tools/dist_test.sh configs/serialgfl/FAQE_r101_mdcn_fpn_last_ms2x.py work_dirs/FAQE_r101_mdcn_fpn_last_ms2x/epoch_24.pth 8 --format-only --options "jsonfile_prefix=./FAQE_r101_mdcn_fpn_last_ms2x_results"
tools/dist_test.sh configs/serialgfl/FAQE_r101_mdcn_fpn_last_ms2x_tta.py work_dirs/FAQE_r101_mdcn_fpn_last_ms2x/epoch_24.pth 8 --format-only --options "jsonfile_prefix=./FAQE_r101_mdcn_fpn_last_ms2x_tta_results"
