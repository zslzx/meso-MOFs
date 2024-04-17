import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np
import csv
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import tqdm
from sklearn.model_selection import StratifiedKFold, KFold

def save_file(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_file(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

plt.rcParams["font.family"] = "Arial"


use_ratio = False   

cls = input('for classification? Y/N.').strip()
while cls not in ['Y', 'N']:
    cls = input('for classification? Y/N.').strip()

if cls == 'Y':
    cls = True
else:
    cls = False   

filename = 'data1.csv'

inputs = []
outputs = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)
    idx = 0
    for row in csv_reader:            
        input = []
        input.append(float(row[1]))
        input.append(float(row[2]))
        input.append(float(row[4]))
        input.append(float(row[5]))
        input.append(float(row[7]))
        inputs.append(np.array(input))
        output = [row[10] == 'yes', 'FCC' in row[11], row[12] == 'yes', row[13] == 'yes', row[14] == 'yes']
        try:
            data1 = float(row[16])
            data2 = float(row[17])
        except:
            data1 = 0
            data2 = 0
        output.append(data1)
        output.append(data2)
        outputs.append(output)


def check_valid(out, pred_col):
    if pred_col == 0:
        return True
    for i in range(pred_col):
        if not out[i]:
            return False
    return True

def get_all_samples(xs, ys, cls=True):
    if cls:
        ys = [[check_valid(ys[i], pred_col=5)] for i in range(len(ys))]    
    else:
        good_idx = [i for i in range(len(ys)) if check_valid(ys[i], pred_col=5)]
        bad_idx = [i for i in range(len(ys)) if i not in good_idx]
        print(bad_idx)
        ys = [[ys[i][-1]] for i in good_idx]
        xs = [xs[i] for i in good_idx]
    return xs, ys



class CLSModel(nn.Module):
    def __init__(self):
        super(CLSModel,self).__init__()

        self.layer1 = nn.Linear(5, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 2)
    
    def forward(self, x, input_dim=5):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        cls = F.softmax(x, dim=-1)[:,-1]
        return cls, x

class RGRModel(nn.Module):
    def __init__(self):
        super(RGRModel,self).__init__()
        self.layer1 = nn.Linear(5, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 1)
    
    def forward(self, x, input_dim=5):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x 

all_dataset = get_all_samples(inputs, outputs, cls=cls)



def cls_loss(cls_logits, gt_out):
    return F.cross_entropy(cls_logits, gt_out[:, 0].long())
    # return torch.abs(cls-gt_out[:,0]).mean()

def val_loss(val, gt_out):
    return torch.abs(val-gt_out).mean()

def get_trained_model(data_in, data_out, is_cls, all_step=50000):
    data_in = torch.tensor(data_in).float().cuda() #N*5
    data_out = torch.tensor(data_out).float().cuda() #N*2
    if is_cls:
        model = CLSModel().cuda()
    else:
        model = RGRModel().cuda()
    opt = torch.optim.SGD(model.parameters(),
                    lr=1e-6,
                    momentum=0,
                    dampening=0,
                    weight_decay=1e-5,
                    nesterov=False)

    for i in range(all_step):
        if is_cls:
            cls, cls_logits = model(data_in)
            loss = cls_loss(cls_logits, data_out)
        else:
            val = model(data_in)
            loss = val_loss(val, data_out)
        if i%5000 == 0:
            print(f'step: {i}, loss: {loss}')
        loss.backward()
        opt.step()
    return model

def evaluate(model, data_in, data_out, cls):
    data_in = torch.tensor(data_in).float()
    data_out = np.array(data_out)
    if cls:
        cls, _ = model(data_in)
        cls = cls.detach().cpu().numpy()
        cls = cls > 0.5
        right_cls = 0
        for i in range(data_out.shape[0]):
            real = (data_out[i,0]) > 0
            if cls[i] == real:
                right_cls += 1
        return right_cls/data_out.shape[0]
    else:
        val = model(data_in)
        val = val.detach().cpu().numpy()
        mses = []
        pred_and_truth = []
        for i in range(data_out.shape[0]):
            mses.append(np.abs(val[i]-data_out[i,0])**2)
            pred_and_truth.append((val[i], data_out[i,0]))
        if len(mses) > 0:
            mse = np.mean(mses)
        else:
            mse = None
        return mse, pred_and_truth
    
def get_idx_split(lst, K):
    slice_len =[len(lst)//K  for i in range(K)]
    for i in range(len(lst) % K):
        slice_len[i] += 1

    rets = []
    start_pos = 0
    for i in range(K):
        rets.append(lst[start_pos:start_pos+slice_len[i]])
        start_pos += slice_len[i]
    return rets




def kfold_validataion(K = 5, cls=True, config={'all_step': 50000}):

    data_in, data_out = all_dataset
    if cls:
        skf = StratifiedKFold(n_splits=K, shuffle=True)
    else:
        skf = KFold(n_splits=K, shuffle=True)

    accs = []
    mses = []
    all_pred_and_truth = []
    for k, (train_idx, test_idx) in enumerate(skf.split(data_in, data_out)):
        train_in = [data_in[i] for i in train_idx]
        train_out = [data_out[i] for i in train_idx]

        test_in = [data_in[i] for i in test_idx]
        test_out = [data_out[i] for i in test_idx]

        model = get_trained_model(train_in, train_out, is_cls=cls, all_step=config['all_step'])
        if cls:
            acc = evaluate(model, test_in, test_out, cls)
            print(f'Fold {k} accuracy: {acc:.2f}')
            accs.append(acc)
        else:
            mse, pred_and_truth = evaluate(model, test_in, test_out, False)
            all_pred_and_truth.extend(pred_and_truth)
            mses.append(mse)
    if cls:
        print('mean acc:', np.mean(accs))
    else:
        print('mean mse:', np.mean(mses))
        all_pred_and_truth = np.array(all_pred_and_truth)
        np.save('all_pred_and_truth_1.npy', all_pred_and_truth)

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




def plot_contuor(X, Y, Z, x_name, y_name, xticks=None, yticks=None, vmin=0.1, vmax=3.0, ticks=None, contour_ticks=None, label_pos_ratio=0.4):
    if ticks is None:
        ticks = np.arange(vmin, vmax, 2.5)

    if contour_ticks is None:
        contour_ticks = np.arange(vmin, vmax, 1.0)

    levels = 1000
    level_boundaries = np.linspace(vmin, vmax, levels + 1)
    colored_contuor = plt.contourf(X,Y,Z, levels =level_boundaries, camp='cool', vmin = vmin,
                             vmax = vmax)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=colored_contuor.norm, cmap=colored_contuor.cmap),
            ticks=ticks,
            boundaries=level_boundaries,
            values=(level_boundaries[:-1] + level_boundaries[1:]) / 2)
    cbar.ax.tick_params(labelsize=24)

    
    CS=plt.contour(X,Y,Z, levels=contour_ticks, colors ='white', linewidth = 1.5 )
    center_point = np.array([(X[0]+X[-1])/2, (Y[0]+Y[-1])/2]).reshape(1,2)
    label_pos = []
    for line in CS.collections:
        for path in line.get_paths():
            logvert = path.vertices
            # find closest point
            if label_pos_ratio == 'center':
                logdist = np.linalg.norm(logvert-center_point, ord=2, axis=1)
                min_ind = np.argmin(logdist)
            else:
                min_ind = int(logvert.shape[0]*label_pos_ratio)
            label_pos.append(logvert[min_ind,:])
    
    CLS = plt.clabel(CS, inline=True, fontsize=24, fmt='%.1f', manual=label_pos)

    plt.tick_params(labelsize=24)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.xlabel(x_name, fontsize=28)
    plt.ylabel(y_name, fontsize=28)


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
    print(grids.shape)
    
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
        print('-----------------------------')
        print(t)
        print(res[i])

    return res

def contuor_around_point(model, point_value, var_ids, var_ranges, tag, **kargs):
    names = ['HCl (mL)', 'CH$_{3}$COOH (mL)', 'ZrCl4 (mmol)', 'HfCl4 (mmol)', 'H2O (mL)']
    all_inputs = []
    for x in var_ranges[0]:
        for y in var_ranges[1]:
            input_ = []
            for i in range(5):
                if i not in var_ids:
                    input_.append(point_value[i])
                elif i == var_ids[0]:
                    input_.append(x)
                else:
                    input_.append(y)
            # print(input_)
            all_inputs.append(input_)
    all_inputs = torch.tensor(np.array(all_inputs)).cuda().float()
    print(all_inputs.shape)
    all_outputs = model(all_inputs).cpu().detach().numpy()
    Z = all_outputs.reshape((var_ranges[0].shape[0], var_ranges[1].shape[0])).transpose([1, 0])
    plt.clf()
    plot_contuor(var_ranges[0], var_ranges[1], Z, names[var_ids[0]], names[var_ids[1]], **kargs)
    plt.scatter(point_value[var_ids[0]], point_value[var_ids[1]], marker='v', c='r',s=100)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)


    # plt.show()
    plt.tight_layout()
    plt.savefig(f'res_v4/contour_point_ANN_{names[var_ids[0]]}-{names[var_ids[1]]}_{tag}.eps', dpi=1000)
    plt.savefig(f'res_v4/contour_point_ANN_{names[var_ids[0]]}-{names[var_ids[1]]}_{tag}.png', dpi=1000)


def plot_samples(rgr_model):

    #contuor_around_point(rgr_model, [0.3, 0.7, 0.1, 0, 3], [0, 1], [np.linspace(0.1, 0.5, 50), np.linspace(0.5, 0.9, 50)], xticks=[0.1, 0.3, 0.5], yticks=[0.5, 0.7, 0.9], tag='max', vmin=0.5, vmax=8.0, ticks=np.arange(0.5, 8.1, 2.5), contour_ticks=np.arange(1.0,8.1,1.0), label_pos_ratio=0.6)
    #contuor_around_point(rgr_model, [0.3, 0.7, 0.1, 0, 3], [2, 3], [np.linspace(0.08, 0.14, 50), np.linspace(0, 0.04, 50)], tag='max', yticks=[0.00,0.02,0.04], vmin=0.5, vmax=8.0, ticks=np.arange(0.5, 8.1, 2.5), contour_ticks=np.arange(1.0,8.1,1.0), label_pos_ratio=0.6)
    #contuor_around_point(rgr_model, [0.3, 0.7, 0.1, 0, 3], [0, 4], [np.linspace(0.1, 0.5, 50), np.linspace(2.8, 3.2, 50)], xticks=[0.1, 0.3, 0.5], yticks=[2.8, 3.0, 3.2],tag='max', vmin=0.5, vmax=8.0, ticks=np.arange(0.5, 8.1, 2.5), contour_ticks=np.arange(1.0,8.1,0.5), label_pos_ratio=0.6)
    #contuor_around_point(rgr_model, [0.3, 0.7, 0.1, 0, 3], [0, 2], [np.linspace(0.1, 0.5, 50), np.linspace(0.08, 0.14, 50)], xticks=[0.1, 0.3, 0.5], tag='max', vmin=0.5, vmax=8.0, ticks=np.arange(0.5, 8.1, 2.5), contour_ticks=np.arange(1.0,8.1,1.0), label_pos_ratio=0.6)
    
    valid_range = [np.linspace(0.1, 0.5, 50), np.linspace(0.5, 0.9, 50), np.linspace(0.08, 0.14, 50), np.linspace(0, 0.04, 50), np.linspace(2.8, 3.2, 50)]
    valid_ticks = [[0.1, 0.3, 0.5], [0.5, 0.7, 0.9], [0.08, 0.10, 0.12, 0.14], [0.00,0.02,0.04], [2.8,3.0,3.2]]
    for x_id in range(5):
        for y_id in range(x_id+1, 5):
            contuor_around_point(rgr_model, [0.3, 0.7, 0.1, 0, 3], [x_id, y_id], [valid_range[x_id], valid_range[y_id]], xticks=valid_ticks[x_id], yticks=valid_ticks[y_id], tag='max', vmin=0.5, vmax=8.0, ticks=np.arange(0.5, 8.1, 2.5), contour_ticks=np.arange(1.0,8.1,1.0), label_pos_ratio=0.6)





    # contuor_around_point(rgr_model, [0.4, 0.6, 0.1, 0, 1.5], [0, 1], [np.linspace(0.30, 0.5, 50), np.linspace(0.55, 0.65, 50)], tag='max', yticks=[0.55, 0.60, 0.65], vmin=0.5, vmax=8.0, ticks=np.arange(0.5, 8.1, 2.5), contour_ticks=np.arange(1.0,8.1,1.0), label_pos_ratio=0.6)
    # contuor_around_point(rgr_model, [0.4, 0.6, 0.1, 0, 1.5], [2, 3], [np.linspace(0.08, 0.14, 50), np.linspace(0, 0.04, 50)], tag='max', yticks=[0.00,0.02,0.04], vmin=0.5, vmax=8.0, ticks=np.arange(0.5, 8.1, 2.5), contour_ticks=np.arange(1.0,8.1,1.0), label_pos_ratio=0.6)
    # contuor_around_point(rgr_model, [0.4, 0.6, 0.1, 0, 1.5], [0, 4], [np.linspace(0.30, 0.5, 50), np.linspace(1.4, 1.8, 50)], tag='max', yticks=[1.4,1.6,1.8], xticks=[0.3,0.4,0.5], vmin=0.5, vmax=8.0, ticks=np.arange(0.5, 8.1, 2.5), contour_ticks=np.arange(1.0,8.1,1.0), label_pos_ratio=0.6)


    # contuor_around_point(rgr_model, [0.8, 0.2, 0, 0.1, 1.5], [0, 1], [np.linspace(0.6, 0.85, 50), np.linspace(0.14, 0.4, 50)], tag='min', vmin=0.1, vmax=3.0, ticks=np.arange(0.5, 3.1, 1.0), contour_ticks=np.arange(0.5, 3.1, 0.5), label_pos_ratio=0.4)
    # contuor_around_point(rgr_model, [0.8, 0.2, 0, 0.1, 1.5], [2, 3], [np.linspace(0, 0.04, 50), np.linspace(0.06, 0.14, 50)], tag='min', yticks=[0.06,0.10,0.14], vmin=0.1, vmax=3.0, ticks=np.arange(0.5, 3.1, 1.0), contour_ticks=np.arange(0.5, 3.1, 0.5), label_pos_ratio=0.4)
    # contuor_around_point(rgr_model, [0.8, 0.2, 0, 0.1, 1.5], [0, 4], [np.linspace(0.70, 0.90, 50), np.linspace(1.4, 1.8, 50)], tag='min', yticks=[1.4, 1.6, 1.8], vmin=0.1, vmax=3.0, ticks=np.arange(0.5, 3.1, 1.0), contour_ticks=np.arange(0.5, 3.1, 0.5), label_pos_ratio=0.4)



def find_input_by_grident(cls_model, rgr_model):
    #采用梯度下降方式寻找目标输入
    best_input, best_loss = optimize_input(cls_model, rgr_model, target=0) 
    print(best_input)
    print(best_loss)
    best_out = rgr_model(best_input)
    print(best_out)

def find_input_by_enumeration(cls_model, rgr_model):
    #采用枚举方式寻找目标输入
    res = get_condition(cls_model, rgr_model, [1.5,2,2.5,3,3.5,6])

    # res = load_file('condition1.data')
    for i, t in enumerate([1.5,2,2.5,3,3.5,6]):
        print('-----------------')
        print(t)
        if(res[i].shape[0]<=0): continue
        order = torch.argsort((res[i][:,-2]-t).abs())
        for j in range(5):
            idx = order[j]
            for k in range(6):
                print('%.2f'%res[i][idx][k], end=' ')
            print()

if __name__ == '__main__':
    # kfold_validataion(K=10, cls=cls, config={'all_step': 100000})#K折验证


    # cls_model = CLSModel().cuda()
    # cls_model.load_state_dict(torch.load('cls_model_v0923-long.pt'))
    rgr_model = RGRModel().cuda()
    rgr_model.load_state_dict(torch.load('full_model_v0923.pt'))

    plot_samples(rgr_model)

    # find_input_by_grident(cls_model, rgr_model)
    # find_input_by_enumeration(cls_model, rgr_model)
    

    