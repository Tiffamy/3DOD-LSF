BACKBONE=$1
PY_ARGS=${@:2}

if [ "$BACKBONE" = "secondiou" ]; then
    echo "Testing secondiou on Waymo"
    python test.py --cfg_file cfgs/waymo_models/secondiou.yaml ${PY_ARGS}

    echo "Testing secondiou on nuScenes"
    python test.py --cfg_file cfgs/da-waymo-nus_models/secondiou/secondiou.yaml ${PY_ARGS}

    echo "Testing secondiou on KITTI"
    python test.py --cfg_file cfgs/da-waymo-kitti_models/secondiou/secondiou_old_anchor.yaml ${PY_ARGS} --extra_tag 64 --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True
elif [ "$BACKBONE" = "pvrcnn" ]; then
    echo "Testing pvrcnn on Waymo"
    python test.py --cfg_file cfgs/waymo_models/pvrcnn.yaml ${PY_ARGS}

    echo "Testing pvrcnn on nuScenes"
    python test.py --cfg_file cfgs/da-waymo-nus_models/pvrcnn/pvrcnn.yaml ${PY_ARGS}

    echo "Testing pvrcnn on KTTI"
    python test.py --cfg_file cfgs/da-waymo-kitti_models/pvrcnn/pvrcnn_old_anchor.yaml ${PY_ARGS} --extra_tag 64 --set DATA_CONFIG_TAR.FOV_POINTS_ONLY True
else
    echo "Unknown backbone"
fi
