print('loading...')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import tqdm
import dataset
from model import CLSModel, RGRModel

all_dataset = dataset.load_dataset('rgr_data.csv')


def cls_loss(cls_logits, gt_out):
    return F.cross_entropy(cls_logits, gt_out[:, 0].long())
    # return torch.abs(cls-gt_out[:,0]).mean()

def val_loss(val, gt_out):
    return torch.abs(val-gt_out).mean()



def get_base_input(target):
    data_in, data_out = all_dataset
    idx = [i for i in range(len(data_in))]
    idx = sorted(idx, key=lambda x: abs(data_out[x][1]-target))
    i = random.randint(0,4)
    return data_in[i:i+1]

def clip_loss(data_in):
    return F.relu(data_in[:,:2]-2).sum() + F.relu(data_in[:,2:4]-0.2).sum() + F.relu(data_in[:,4]-10).sum() + 100 * F.relu(-data_in).sum()

def optimize_input(cls_model, rgr_model, target, all_step=100000):
    
    base_in = get_base_input(target)
    base_in = torch.tensor(base_in).float()
    base_in = nn.Parameter(base_in)

    target = torch.tensor([1,target]).view(1,2).float()
    #target_val = torch.tensor([target]).view(1).float()

    lr = 1e-7
    opt = torch.optim.SGD([base_in],
                    lr=lr,
                    momentum=0.1,
                    dampening=0,
                    weight_decay=0,
                    nesterov=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, lr, 100000+100,
                pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    # scheduler.
    best = base_in.clone()
    best_loss = 100
    for i in range(all_step):
        cls, cls_logits = cls_model(base_in)
        val = rgr_model(base_in)
        loss = max(cls_loss(cls_logits, target), 0.2) + val_loss(val, target) + clip_loss(base_in)
        if loss < best_loss:
            best_loss = loss.item()
            best = base_in.clone()
        if i%5000 == 0:
            print(f'step: {i}, loss: {loss}')
        loss.backward()
        opt.step()
        scheduler.step()
    
    return best, best_loss


def prod_sum(lst):
    ret = 1
    for x in lst:
        ret = ret*x
    return ret

def get_condition(cls_model, rgr_model, targets): #采用枚举的方式寻找目标输入
    hcl = np.linspace(0, 1, 50)
    ch3cooh = np.linspace(0, 1, 50)
    zrcl4 = np.linspace(0, 0.1, 50)
    hfcl4 = np.linspace(0, 0.1, 50)
    h2o = np.linspace(0, 5, 50)

    tick_nums = [x.shape[0] for x in (hcl, ch3cooh, zrcl4, hfcl4, h2o)]
    
    grids = np.meshgrid(hcl,ch3cooh,zrcl4,hfcl4,h2o, copy=True, sparse=False, indexing='ij')
    grids = np.stack(grids, axis=-1)
    # print(grids.shape)
    
    grids = grids.reshape(prod_sum(tick_nums), -1)
    all_len=grids.shape[0]
    batch_size = 1000
    
    res = [[] for i in targets]
    
    for idx in tqdm.tqdm(range(0, all_len, batch_size)):
        if idx + batch_size > all_len:
            batch = grids[idx:]
        else:
            batch = grids[idx:idx+batch_size]
        with torch.no_grad():
            batch = torch.tensor(batch).float().cuda()
            cls, cls_logits = cls_model(batch)
            val = rgr_model(batch)
            for i, t in enumerate(targets):
                valid_idx = torch.logical_and((cls>0.9).view(-1), ((val-t).abs()<0.2).view(-1))
                x = torch.cat((batch[valid_idx], val[valid_idx], cls[valid_idx].view(-1,1)), dim=-1)
                res[i].append(x)

    
    for i, t in enumerate(targets):
        res[i] = torch.cat(res[i], dim=0)
        # print('-----------------------------')
        # print(t)
        # print(res[i])

    return res

def find_input_by_grident(cls_model, rgr_model, target=6):
    #采用梯度下降方式寻找目标输入
    best_input, best_loss = optimize_input(cls_model, rgr_model, target=target) 
    print(best_input)
    print(best_loss)
    best_out = rgr_model(best_input)
    print(best_out)

def find_input_by_grid_search(cls_model, rgr_model, target=6):
    #采用网格搜索方式寻找目标输入
    res = get_condition(cls_model, rgr_model, [target])

    # res = load_file('condition1.data')
    for i, t in enumerate([target]):
        print('-----------------')
        print(f'Top 5 conditions for candidates to reach target value {t}:')
        if(res[i].shape[0]<=0): continue
        order = torch.argsort((res[i][:,-2]-t).abs())
        for j in range(5):
            idx = order[j]
            for k in range(6):
                print('%.2f'%res[i][idx][k], end=' ')
            print()
        print('-----------------')

if __name__ == '__main__':

    cls_model = CLSModel().cuda()
    cls_model.load_state_dict(torch.load('cls_model.pt'))
    rgr_model = RGRModel().cuda()
    rgr_model.load_state_dict(torch.load('rgr_model.pt'))

    find_input_by_grid_search(cls_model, rgr_model, target=6) #set target I(111)/I(002) value, find some candidate conditions.
    

    