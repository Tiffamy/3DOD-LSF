import glob
import os
import tqdm
import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable

from pcdet.models import load_data_to_gpu
import pdb
import numpy as np
import random

from pcdet.ops.iou3d_nms import iou3d_nms_utils

class gromovWasserstein(nn.Module):
    def __init__(self, lambda_=1, affinity_type='cosine', l_type='KL'):
        super(gromovWasserstein, self).__init__()
        self.affinity_type=affinity_type
        self.l_type=l_type
        self.lambda_ = lambda_
        self.rate = 0.99
    
    def forward(self, feat_stu, feat_tea, rois_diff=0):
        affinity_stu = self.affinity_matrix(feat_stu)
        affinity_tea = self.affinity_matrix(feat_tea)
        T = torch.eye(feat_stu.size(0)).cuda()

        if type(rois_diff)!=int:
            T = T + self.lambda_*1/(rois_diff + 1)
        T = T/T.sum()
        # print(T.shape)
        # print(T.max(), T.min())
        cost = self.L(affinity_stu,affinity_tea,T)
        loss = (cost * T).sum()
        return loss
    
    def affinity_matrix_cross(self, feat1, feat2):
        if self.affinity_type=='cosine':
            energy1 = torch.sqrt(torch.sum(feat1 ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            energy2 = torch.sqrt(torch.sum(feat2 ** 2, dim=1, keepdim=True))
            cos_sim = torch.matmul(feat1, torch.t(feat2)) / (torch.matmul(energy1, torch.t(energy2)))
            affinity = cos_sim
        else:
            pass
        return affinity

    def affinity_matrix(self, feat):
        if self.affinity_type=='cosine':
            energy = torch.sqrt(torch.sum(feat ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            cos_sim = torch.matmul(feat, torch.t(feat)) / (torch.matmul(energy, torch.t(energy)) )
            affinity = cos_sim
        else:
            feat = torch.matmul(feat, torch.t(feat))  # (batch_size, batch_size)
            feat_diag = torch.diag(feat).view(-1, 1).repeat(1, feat.size(0))  # (batch_size, batch_size)
            affinity = 1-torch.exp(-(feat_diag + torch.t(feat_diag) - 2 * feat)/feat.size(1))
        return affinity
    
    def L(self, affinity_stu, affinity_tea, T):
        stu_1 = Variable(torch.ones(affinity_stu.size(0), 1).cuda())
        tea_1 = Variable(torch.ones(affinity_tea.size(0), 1).cuda())
        p=T.mm(tea_1)
        q=T.t().mm(stu_1)
        if self.l_type == 'L2':
            f1_st = (affinity_stu ** 2).mm(p).mm(tea_1.t())  
            f2_st = stu_1.mm(q.t()).mm((affinity_tea ** 2).t())
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * affinity_stu.mm(T).mm(affinity_tea.t())
        elif self.l_type=='KL':
            f1_st = torch.matmul(affinity_stu * torch.log(affinity_stu+ 1e-7) - affinity_stu, p).mm(tea_1.t())
            f2_st = stu_1.mm(torch.matmul(torch.t(q), torch.t(affinity_tea)))
            cost_st = f1_st + f2_st
            cost = cost_st - torch.matmul(torch.matmul(affinity_stu, T), torch.t(torch.log(affinity_tea+1e-7)))
        return cost

def graph_loss(batch_teacher, batch, GW_Loss=True):
    rois = batch_teacher['rois_mimic'].detach() # (batch_size, 128, 7)
    batch_size = rois.size(0)

    teacher_bev_features = batch_teacher['spatial_features_2d'].detach()
    student_bev_features = batch['spatial_features_2d']
    rois_masks = batch['rois_masks'].detach()
    loss_node = torch.norm(teacher_bev_features - student_bev_features, p=2, dim=1)
    loss_FCA = (loss_node * rois_masks).sum() / batch_size / rois.size(1)

    # ver 2
    # teacher_pool_features = batch_teacher['pooled_features'].detach()
    # student_pool_features = batch['pooled_features']
    # loss_node = torch.norm(teacher_pool_features - student_pool_features, p=2, dim=1)
    # loss_FCA = loss_node.sum() / batch_size / rois.size(1)

    teacher_features = batch_teacher['shared_features'].detach()
    student_features = batch['shared_features']
    loss_edge_tensor = torch.zeros(batch_size).cuda()
    if GW_Loss:
        gw_loss = gromovWasserstein(affinity_type='cosine', l_type='KL') 
        
        student_features = student_features.view(batch_size, rois.size(1), -1)
        teacher_features = teacher_features.view(batch_size, rois.size(1), -1)
        
        # Compute weighted matrix
        # num_rois = rois.size(1)
        # teacher_rcnn_iou = batch_teacher['rcnn_iou'].detach() # (batch_size*128, 1)
        # student_rcnn_iou = batch['rcnn_iou'].detach() # (batch_size*128, 1)
        # rcnn_iou_t = teacher_rcnn_iou.view(batch_size, num_rois, 1)
        # rcnn_iou_s = student_rcnn_iou.view(batch_size, num_rois, 1)
        
        for i in range(batch_size):
            f_s = student_features[i]
            f_t = teacher_features[i]
            # rcnn_diff = (rcnn_iou_s[i].unsqueeze(1) - rcnn_iou_t[i].unsqueeze(0)).abs().sum(-1).cuda() #(num_stu, num_tea)

            curr_rois = rois[i]
            rois_diff = (curr_rois[:, 0:3].unsqueeze(1) - curr_rois[:, 0:3].unsqueeze(0)).pow(2).sum(-1).sqrt() + \
                        (curr_rois[:, 3:6].unsqueeze(1) - curr_rois[:, 3:6].unsqueeze(0)).pow(2).sum(-1).sqrt() + \
                        (curr_rois[:, 6:7].unsqueeze(1) - curr_rois[:, 6:7].unsqueeze(0)).abs().sum(-1)
            rois_diff = rois_diff.cuda()            

            loss_edge = gw_loss(f_s, f_t, rois_diff)
            loss_edge_tensor[i] = loss_edge
        loss_edge_total = loss_edge_tensor.sum() / batch_size

        return loss_FCA, loss_edge_total

    return loss_FCA, torch.zeros(1).cuda()

def update_ema_model(model, ema_model, alpha=0.999):
    multiplier = 1.0
    alpha = 1 - multiplier*(1-alpha)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

def confident_selection(model, batch_list, selection_record, confidence_record, num_pcd_beams):
    with torch.no_grad():
        model.eval()
        confidence_list = []
        for batch in batch_list:
            load_data_to_gpu(batch)
            pred_ret = model(batch)
            pred_dicts = pred_ret[0]
            gt_boxes = batch['gt_boxes']

            confidence_frame = []
            for batch_idx in range(len(pred_dicts)):
                box_preds = pred_dicts[batch_idx]['pred_boxes']
                cur_gt = gt_boxes[batch_idx]
                k = cur_gt.__len__() - 1
                while k > 0 and cur_gt[k].sum() == 0:
                    k -= 1
                cur_gt = cur_gt[:k + 1]
                
                if cur_gt.shape[0] > 0 and box_preds.shape[0] > 0:
                    iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
                    if iou3d_rcnn.shape[0] == 0:
                        confidence_frame.append(0)
                    else:
                        box_values = pred_dicts[batch_idx]['pred_scores'][iou3d_rcnn.max(dim=1)[0] > 0.7]
                        if box_values.shape[0] > 0:
                            aggregated_values = torch.mean(box_values)
                            confidence_frame.append(aggregated_values.item())
                        else:
                            confidence_frame.append(0)
                else:
                    confidence_frame.append(0)

            confidence_list.append(np.mean(np.array(confidence_frame)))

        # Compute weights for each beam type
        weights = np.array(num_pcd_beams)
        # Clip weights to avoid numerical instability
        weights = np.clip(weights, 1, None)

        weights = weights / weights.sum()
        confidence_list = np.array(confidence_list)
        confidence_list = confidence_list * weights

        confidence_record.append(confidence_list.tolist())
        selected_batch = np.argmin(np.array(confidence_list))
        selection_record.append(selected_batch)
        num_pcd_beams[selected_batch] += 1

        # print('weights: ', weights)
        # print('Confidence list: ', confidence_list)
        # print('Selected batch: ', selected_batch)
        
        return batch_list[selected_batch]

def train_one_epoch(model, model_teacher, optimizer, train_loader,
                    model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter,
                    alpha=1, beta=5, tb_log=None, leave_pbar=False, ema=False,
                    sel_record=None, conf_record=None, num_pcd_beams=None):

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch_teacher, batch_student_64, batch_student_32, batch_student_16 = next(dataloader_iter)

        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch_teacher, batch_student_64, batch_student_32, batch_student_16 = next(dataloader_iter)

            print('new iters')

        #batch_teacher = batch.copy()
        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        
        optimizer.zero_grad()

        batch_student_list = [batch_student_64, batch_student_32, batch_student_16]
        batch = confident_selection(model, batch_student_list, sel_record, conf_record, num_pcd_beams)
        # batch = random.choice(batch_student_list)

        model_teacher.eval()
        model.train()

        load_data_to_gpu(batch_teacher)
        batch_teacher['mimic'] = 'mimic'
        batch['mimic'] = 'mimic'
        with torch.no_grad():
            tb_dict_teacher, disp_dict_teacher, batch_teacher_new = model_teacher(batch_teacher)


        batch['rois_mimic'] = batch_teacher_new['rois_mimic'].clone()
        temp, batch_new = model_func(model, batch)
        loss, tb_dict, disp_dict = temp

        if alpha == 0 and beta == 0:
            loss_sum = loss
        else:
            GW_Loss = True
            loss_FCA, loss_GERA = graph_loss(batch_teacher_new, batch_new, GW_Loss=GW_Loss)
            if GW_Loss:
                loss_graph = alpha*loss_FCA + beta*loss_GERA
            else:
                loss_graph = alpha*loss_FCA
            
            loss_sum = loss + loss_graph

        loss_sum.backward()
        if ema:
            update_ema_model(model, model_teacher)
        # loss, tb_dict, disp_dict = model_func(model, batch)
        # loss.backward()
            
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        if alpha == 0 and beta == 0:
            disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        else GW_Loss:
            disp_dict.update({'loss': loss.item(), 'loss_FCA': alpha*loss_FCA.item(), 'loss_GERA': beta*loss_GERA.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_one_epoch_KITTI(model, model_teacher, optimizer, train_loader,
                    model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter,
                    alpha=1, beta=5, tb_log=None, leave_pbar=False, ema=False,
                    sel_record=None, conf_record=None, num_pcd_beams=None):

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch_teacher, batch_student_64, batch_student_32, batch_student_32_2, batch_student_16, batch_student_16_2 = next(dataloader_iter)

        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch_teacher, batch_student_64, batch_student_32, batch_student_32_2, batch_student_16, batch_student_16_2 = next(dataloader_iter)

            print('new iters')

        #batch_teacher = batch.copy()
        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        optimizer.zero_grad()

        batch_student_list = [batch_student_64, batch_student_32, batch_student_32_2, batch_student_16, batch_student_16_2]
        batch = confident_selection(model, batch_student_list, sel_record, conf_record, num_pcd_beams)
        # batch = random.choice(batch_student_list)

        model_teacher.eval()
        model.train()

        load_data_to_gpu(batch_teacher)
        batch_teacher['mimic'] = 'mimic'
        batch['mimic'] = 'mimic'
        with torch.no_grad():
            tb_dict_teacher, disp_dict_teacher, batch_teacher_new = model_teacher(batch_teacher)


        batch['rois_mimic'] = batch_teacher_new['rois_mimic'].clone()
        temp, batch_new = model_func(model, batch)
        loss, tb_dict, disp_dict = temp

        if alpha == 0 and beta == 0:
            loss_sum = loss
        else:
            GW_Loss = True
            loss_FCA, loss_GERA = graph_loss(batch_teacher_new, batch_new, GW_Loss=GW_Loss)
            if GW_Loss:
                loss_graph = alpha*loss_FCA + beta*loss_GERA
            else:
                loss_graph = alpha*loss_FCA
        
            loss_sum = loss + loss_graph

        loss_sum.backward()
        if ema:
            update_ema_model(model, model_teacher)
        # loss, tb_dict, disp_dict = model_func(model, batch)
        # loss.backward()
            
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        if alpha == 0 and beta == 0:
            disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        else:
            disp_dict.update({'loss': loss.item(), 'loss_FCA': alpha*loss_FCA.item(), 'loss_GERA': beta*loss_GERA.item(), 'lr': cur_lr})
        # disp_dict.update({'loss': loss.item(),  'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_model_general(model, model_teacher, optimizer, 
                train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, output_dir, ckpt_save_dir, ps_label_dir,
                source_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, 
                alpha=1, beta=5, ema=False):
    selection_record = []
    confidence_record = []
    num_pcd_beams = [0, 0, 0]
    accumulated_iter = start_iter

    if start_epoch > 0:
        selection_record = list(np.load(ckpt_save_dir / 'selection_record.npy'))
        confidence_record = list(np.load(ckpt_save_dir / 'confidence_record.npy'))
        num_pcd_beams = [np.sum(np.array(selection_record) == i) for i in range(3)]

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)

        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, model_teacher, 
                optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                alpha=alpha, beta=beta, ema=ema,
                sel_record=selection_record, 
                conf_record=confidence_record,
                num_pcd_beams=num_pcd_beams
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
                np.save(ckpt_save_dir / 'selection_record.npy', np.array(selection_record))
                np.save(ckpt_save_dir / 'confidence_record.npy', np.array(confidence_record))

    # Save selection record and confidence record
    selection_record_np = np.array(selection_record).reshape(total_epochs, -1)
    confidence_record_np = np.array(confidence_record).reshape(total_epochs, -1, len(confidence_record[0]))
    np.save(output_dir / 'selection_record.npy', selection_record_np)
    np.save(output_dir / 'confidence_record.npy', confidence_record_np)
    
def train_model_general_KITTI(model, model_teacher, optimizer, 
                train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, output_dir, ckpt_save_dir, ps_label_dir,
                source_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, 
                alpha=1, beta=5, ema=False):
    selection_record = []
    confidence_record = []
    num_pcd_beams = [0, 0, 0, 0, 0]
    accumulated_iter = start_iter
    if start_epoch > 0:
        selection_record = list(np.load(output_dir / 'selection_record.npy'))
        confidence_record = list(np.load(output_dir / 'confidence_record.npy'))
        num_pcd_beams = [np.sum(np.array(selection_record) == i) for i in range(5)]
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)

        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch_KITTI(
                model, model_teacher, 
                optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                alpha=alpha, beta=beta, ema=ema,
                sel_record=selection_record, 
                conf_record=confidence_record,
                num_pcd_beams=num_pcd_beams
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
                np.save(ckpt_save_dir / 'selection_record.npy', np.array(selection_record))
                np.save(ckpt_save_dir / 'confidence_record.npy', np.array(confidence_record))
    
    # Save selection record and confidence record
    selection_record_np = np.array(selection_record).reshape(total_epochs, -1)
    confidence_record_np = np.array(confidence_record).reshape(total_epochs, -1, len(confidence_record[0]))
    np.save(output_dir / 'selection_record.npy', selection_record_np)
    np.save(output_dir / 'confidence_record.npy', confidence_record_np)

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
