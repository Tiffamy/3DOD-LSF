from functools import partial
import numpy as np
from . import augmentor_utils, database_sampler
from ...utils import common_utils


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None, extra_disable_aug=[]):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.augmentor_configs = augmentor_configs

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST or cur_cfg.NAME in extra_disable_aug:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_object_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_rotation, config=config)

        gt_boxes, points = augmentor_utils.rotate_objects(
            data_dict['gt_boxes'],
            data_dict['points'],
            data_dict['gt_boxes_mask'],
            rotation_perturb=config['ROT_UNIFORM_NOISE'],
            prob=config['ROT_PROB'],
            num_try=50
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_statistical_normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_statistical_normalization, config=config)

        
        if 'RSN_factor' in data_dict.keys():
            size_res = data_dict['RSN_factor']
        else:
            delta_L = config['delta_L']
            delta_W = config['delta_W']
            delta_H = config['delta_H']

            # Randomly sample a length, width, height
            L = round(np.random.uniform(low=delta_L[0], high=delta_L[1]), 2)
            W = round(np.random.uniform(low=delta_W[0], high=delta_W[1]), 2)
            H = round(np.random.uniform(low=delta_H[0], high=delta_H[1]), 2)
            size_res = [L, W, H]

        points, gt_boxes = augmentor_utils.normalize_object_size(
            data_dict['gt_boxes'], data_dict['points'], data_dict['gt_boxes_mask'], size_res
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['RSN_factor'] = size_res
        return data_dict

    def random_world_sampling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_sampling, config=config)
        gt_boxes, points, gt_boxes_mask = augmentor_utils.global_sampling(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            sample_ratio_range=config['WORLD_SAMPLE_RATIO'],
            prob=config['PROB']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_boxes_mask'] = gt_boxes_mask
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            if 'random_flip_along_%s' % cur_axis in data_dict.keys():
                gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points, data_dict['random_flip_along_%s' % cur_axis]
                )
            else:
                gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points
                )
            data_dict['random_flip_along_%s' % cur_axis] = enable

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        
        if 'global_rotation' in data_dict.keys():
            gt_boxes, points, noise_rotation = augmentor_utils.global_rotation(
                data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, noise_rotation_gt=data_dict['global_rotation']
            )
        else:
            gt_boxes, points, noise_rotation = augmentor_utils.global_rotation(
                data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
            )
        data_dict['global_rotation'] = noise_rotation

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        
        if 'global_scaling' in data_dict.keys():
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], data_dict['global_scaling']
            )
        else:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
            )
        data_dict['global_scaling'] = noise_scale
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def normalize_object_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalize_object_size, config=config)
        points, gt_boxes = augmentor_utils.normalize_object_size(
            data_dict['gt_boxes'], data_dict['points'], data_dict['gt_boxes_mask'], config['SIZE_RES']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict

    def re_prepare(self, augmentor_configs=None, intensity=None):
        self.data_augmentor_queue = []

        if augmentor_configs is None:
            augmentor_configs = self.augmentor_configs

        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            # scale data augmentation intensity
            if intensity is not None:
                cur_cfg = self.adjust_augment_intensity(cur_cfg, intensity)
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def adjust_augment_intensity(self, config, intensity):
        adjust_map = {
            'random_object_scaling': 'SCALE_UNIFORM_NOISE',
            'random_object_rotation': 'ROT_UNIFORM_NOISE',
            'random_world_rotation': 'WORLD_ROT_ANGLE',
            'random_world_scaling': 'WORLD_SCALE_RANGE',
        }

        def cal_new_intensity(config, flag):
            origin_intensity_list = config.get(adjust_map[config.NAME])
            assert len(origin_intensity_list) == 2
            assert np.isclose(flag - origin_intensity_list[0], origin_intensity_list[1] - flag)
            
            noise = origin_intensity_list[1] - flag
            new_noise = noise * intensity
            new_intensity_list = [flag - new_noise, new_noise + flag]
            return new_intensity_list

        if config.NAME not in adjust_map:
            return config
        
        # for data augmentations that init with 1
        if config.NAME in ["random_object_scaling", "random_world_scaling"]:
            new_intensity_list = cal_new_intensity(config, flag=1)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        elif config.NAME in ['random_object_rotation', 'random_world_rotation']:
            new_intensity_list = cal_new_intensity(config, flag=0)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        else:
            raise NotImplementedError
