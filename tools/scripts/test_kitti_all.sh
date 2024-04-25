PY_ARGS=${@:1}

python test.py --cfg_file cfgs/kitti_models/second_iou_car.yaml ${PY_ARGS} --extra_tag 16^ --eval_tag 16^ --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True

python test.py --cfg_file cfgs/kitti_models/second_iou_car.yaml ${PY_ARGS} --extra_tag 16 --eval_tag 16 --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True

python test.py --cfg_file cfgs/kitti_models/second_iou_car.yaml ${PY_ARGS} --extra_tag 32^ --eval_tag 32^ --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True

python test.py --cfg_file cfgs/kitti_models/second_iou_car.yaml ${PY_ARGS} --extra_tag 32 --eval_tag 32 --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True

python test.py --cfg_file cfgs/kitti_models/second_iou_car.yaml ${PY_ARGS} --extra_tag 64 --eval_tag 64 --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True