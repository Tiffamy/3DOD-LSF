import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .waymo.waymo_dataset import WaymoDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .lyft.lyft_dataset import LyftDataset


__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'WaymoDataset': WaymoDataset,
    'NuScenesDataset': NuScenesDataset,
    'LyftDataset': LyftDataset,
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        index_new = x1['index']
        extra_dict = {}
        extra_keys = ['random_flip_along_x', 'random_flip_along_y', 'global_rotation', 'global_scaling']
        for key in extra_keys:
            if key in x1.keys():
                extra_dict[key] = x1[key]
        self.dataset2.extra_dict = extra_dict
        x2 = self.dataset2[index_new]
        return x1, x2

    def __len__(self):
        return len(self.dataset1)

class MyDataset_all(Dataset):
    def __init__(self, dataset1, dataset2, dataset3, dataset4):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.dataset4 = dataset4

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        index_new = x1['index']
        extra_dict = {}
        extra_keys = ['random_flip_along_x', 'random_flip_along_y', 'global_rotation', 'global_scaling']

        for key in extra_keys:
            if key in x1.keys():
                extra_dict[key] = x1[key]
        self.dataset2.extra_dict = extra_dict
        x2 = self.dataset2[index_new]
        
        self.dataset3.extra_dict = extra_dict
        x3 = self.dataset3[index_new]

        self.dataset4.extra_dict = extra_dict
        x4 = self.dataset4[index_new]

        return x1, x2, x3, x4

    def __len__(self):
        return len(self.dataset1)

class MyDataset_all_KITTI(Dataset):
    def __init__(self, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.dataset4 = dataset4
        self.dataset5 = dataset5
        self.dataset6 = dataset6

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        index_new = x1['index']
        extra_dict = {}
        extra_keys = ['random_flip_along_x', 'random_flip_along_y', 'global_rotation', 'global_scaling']

        for key in extra_keys:
            if key in x1.keys():
                extra_dict[key] = x1[key]
        self.dataset2.extra_dict = extra_dict
        x2 = self.dataset2[index_new]
        
        self.dataset3.extra_dict = extra_dict
        x3 = self.dataset3[index_new]

        self.dataset4.extra_dict = extra_dict
        x4 = self.dataset4[index_new]

        self.dataset5.extra_dict = extra_dict
        x5 = self.dataset5[index_new]

        self.dataset6.extra_dict = extra_dict
        x6 = self.dataset6[index_new]

        return x1, x2, x3, x4, x5, x6

    def __len__(self):
        return len(self.dataset1)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, use_ori=None, set_sampler=None, beam_tag='default'):

    if use_ori is None:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )
    elif beam_tag != 'all':
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )

        dataset_teacher = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag=beam_tag,
            disable_aug = ["random_object_scaling", "random_statistical_normalization"]
        )
        dataset_merge = MyDataset(dataset, dataset_teacher)
    else:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            disable_aug = ["random_object_scaling", "random_statistical_normalization"]
        )

        dataset_student_64 = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag='64',
        )

        dataset_student_32 = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag='32',

        )

        dataset_student_16 = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag='16^',
        )

        dataset_merge = MyDataset_all(dataset, dataset_student_64, dataset_student_32, dataset_student_16)

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    if use_ori is None:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
            drop_last=False, sampler=sampler, timeout=0
        )
    else:
        dataloader = DataLoader(
            dataset_merge, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
            drop_last=False, sampler=sampler, timeout=0
        )

    return dataset, dataloader, sampler

def build_dataloader_KITTI(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, use_ori=None, set_sampler=None, beam_tag='default'):

    if use_ori is None:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )
    elif beam_tag != 'all':
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )

        dataset_teacher = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag=beam_tag,
        )
        dataset_merge = MyDataset(dataset, dataset_teacher)
    else:
        dataset = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )

        dataset_student_64 = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag='64',
        )

        dataset_student_32 = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag='32',
        )

        dataset_student_32_2 = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag='32^',
        )

        dataset_student_16 = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag='16',
        )

        dataset_student_16_2 = __all__[dataset_cfg.DATASET](
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
            use_ori=use_ori,
            beam_tag='16^',
        )

        dataset_merge = MyDataset_all_KITTI(dataset, dataset_student_64, dataset_student_32, dataset_student_32_2, dataset_student_16, dataset_student_16_2)

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    if use_ori is None:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
            drop_last=False, sampler=sampler, timeout=0
        )
    else:
        dataloader = DataLoader(
            dataset_merge, batch_size=batch_size, pin_memory=True, num_workers=workers,
            shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
            drop_last=False, sampler=sampler, timeout=0
        )

    return dataset, dataloader, sampler