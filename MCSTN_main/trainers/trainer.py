import torch
import time
from tqdm import tqdm

from torch.autograd import Variable
from utils.metrics import AverageMeter
from utils.loss.utils import _sigmoid
def train_model(opts, model, train_dataloder, optimizer, criternions, epoch):
    # total loss
    losses_total = AverageMeter()
    # regression loss
    losses_reg_hm = AverageMeter()
    losses_reg_loc = AverageMeter()
    losses_reg_z = AverageMeter()

    # train model
    model.train()
    with tqdm(enumerate(train_dataloder), total=len(train_dataloder), ncols=opts.ncols) as t:
        for i, data in t:
            torch.cuda.synchronize()
            data = {key: Variable(value, requires_grad=False).to(opts.device) for key, value in data.items()}
            pred_hms, pred_locs, pred_z_axiss = model(data['template_pc'], data['search_pc'])

            # 改变loss
            pred_hm = []
            for i in range(len(pred_hms)):
                pred_hm_i = _sigmoid(pred_hms[i])
                pred_hm.append(pred_hm_i)

            # 3. calculate loss
            loss_reg_hm = []
            loss_reg_loc = []
            loss_reg_z = []
            for i in range(len(pred_locs)):
                loss_reg_hm_i = criternions['hm'](pred_hm[i], data['heat_map'])
                loss_reg_loc_i = criternions['loc'](pred_locs[i], data['index_offsets'], data['local_offsets'])
                loss_reg_z_i = criternions['z_axis'](pred_z_axiss[i], data['index_center'], data['z_axis'])

                loss_reg_hm.append(loss_reg_hm_i)
                loss_reg_loc.append(loss_reg_loc_i)
                loss_reg_z.append(loss_reg_z_i)
            # total loss
            num_block = len(loss_reg_hm)
            total_loss = []
            for i in range(num_block):
                if i != num_block-1:
                    total_loss_i = (1.0 * loss_reg_hm[i] + 1.0 * loss_reg_loc[i] + 2.0 * loss_reg_z[i])*0.1
                else:
                    total_loss_i = 1.0 * loss_reg_hm[i] + 1.0 * loss_reg_loc[i] + 2.0 * loss_reg_z[i]
                total_loss.append(total_loss_i)
            total_loss = sum(total_loss)
            # 4. calculate gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            #
            # 5. update infomation
            for i in range(num_block):
                if i != num_block-1:
                    loss_reg_hm[i] = 0.1 * loss_reg_hm[i]
                    loss_reg_loc[i] = 0.1 * loss_reg_loc[i]
                    loss_reg_z[i] = 0.1 * loss_reg_z[i]
            losses_reg_hm.update(1.0 * (sum(loss_reg_hm)).item())
            losses_reg_loc.update(1.0 * (sum(loss_reg_loc)).item())
            losses_reg_z.update(2.0 * (sum(loss_reg_z)).item())
            # total loss
            losses_total.update(total_loss.item())

            lr = optimizer.param_groups[0]['lr']
            t.set_description(f'Train {epoch}: '
                            f'Loss:{losses_total.avg:.3f} '
                            f'Reg:({losses_reg_hm.avg:.4f}, '
                            f'{losses_reg_loc.avg:.4f}, '
                            f'{losses_reg_z.avg:.3f}), '
                            f'lr:{1000*lr:.3f} '
                            )


    return losses_total.avg

def valid_model(opts, model, valid_dataloder, criternions, epoch):
    # total loss
    losses_total = AverageMeter()
    # regression loss
    losses_reg_hm = AverageMeter()
    losses_reg_loc = AverageMeter()
    losses_reg_z = AverageMeter()

    # evaluate model
    model.eval()

    with tqdm(enumerate(valid_dataloder), total=len(valid_dataloder), ncols=opts.ncols) as t:
        with torch.no_grad():
            end = time.time()
            for i, data in t:
                # 1. get inputs
                data = {key: Variable(value, requires_grad=False).to(opts.device) for key, value in data.items()}
            
                # 2. calculate outputs
                pred_hms, pred_locs, pred_z_axiss = model(data['template_pc'], data['search_pc'])

                num_block = len(pred_hms)
                pred_hm = []
                for i in range(len(pred_hms)):
                    pred_hm_i = _sigmoid(pred_hms[i])
                    pred_hm.append(pred_hm_i)

                # 3. calculate loss

                loss_reg_hm = []
                loss_reg_loc = []
                loss_reg_z = []
                for i in range(len(pred_locs)):
                    loss_reg_hm_i = criternions['hm'](pred_hm[i], data['heat_map'])
                    loss_reg_loc_i = criternions['loc'](pred_locs[i], data['index_offsets'], data['local_offsets'])
                    loss_reg_z_i = criternions['z_axis'](pred_z_axiss[i], data['index_center'], data['z_axis'])

                    loss_reg_hm.append(loss_reg_hm_i)
                    loss_reg_loc.append(loss_reg_loc_i)
                    loss_reg_z.append(loss_reg_z_i)
                # total loss
                total_loss = []
                for i in range(num_block):
                    if i != num_block - 1:
                        total_loss_i = (1.0 * loss_reg_hm[i] + 1.0 * loss_reg_loc[i] + 2.0 * loss_reg_z[i]) * 0.1
                    else:
                        total_loss_i = 1.0 * loss_reg_hm[i] + 1.0 * loss_reg_loc[i] + 2.0 * loss_reg_z[i]
                    total_loss.append(total_loss_i)
                total_loss = sum(total_loss)

                # 4. update infomation
                # 4.1 update training error
                # regression loss
                for i in range(num_block):
                    if i != num_block - 1:
                        loss_reg_hm[i] = 0.1 * loss_reg_hm[i]
                        loss_reg_loc[i] = 0.1 * loss_reg_loc[i]
                        loss_reg_z[i] = 0.1 * loss_reg_z[i]
                losses_reg_hm.update(1.0 * (sum(loss_reg_hm)).item())
                losses_reg_loc.update(1.0 * (sum(loss_reg_loc)).item())
                losses_reg_z.update(2.0 * (sum(loss_reg_z)).item())
                # total loss
                losses_total.update(total_loss.item())

                t.set_description(  f'Test  {epoch}: '
                                    f'Loss:{losses_total.avg:.3f} '
                                    f'Reg:({losses_reg_hm.avg:.4f}, '
                                    f'{losses_reg_loc.avg:.4f}, '
                                    f'{losses_reg_z.avg:.3f}), '
                                    )

    return losses_total.avg