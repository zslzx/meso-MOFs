import os
import torch
import random
import scipy
import numpy as np
import dataset
from model import CLSModel, RGRModel

import shap
import matplotlib.pyplot as plt



cls_model = CLSModel().cuda()
cls_model.load_state_dict(torch.load('cls_model.pt'))
rgr_model = RGRModel().cuda()
rgr_model.load_state_dict(torch.load('rgr_model.pt'))








def analyze(cls):
    all_dataset = dataset.load_dataset('all_data.csv', cls=cls)
    data_in, data_out = all_dataset
    feature_names = ['HCl (mL)', 'CH$_{3}$COOH (mL)', 'ZrCl4 (mmol)', 'HfCl4 (mmol)', 'H2O (mL)']
    os.makedirs('dependence', exist_ok=True)
    os.makedirs('waterfull', exist_ok=True)
    os.makedirs('overall', exist_ok=True)
    

    f_in = torch.tensor(data_in).cuda().float()
    data = f_in.cpu().detach().numpy()

    if cls:
        f = lambda x: cls_model(torch.from_numpy(x).cuda(), input_dim=5)[0].cpu().detach().numpy()
    else:
        f = lambda x: rgr_model(torch.from_numpy(x).cuda(), input_dim=5).cpu().detach().numpy()


    explainer = shap.KernelExplainer(f, data)

    if cls:
        shap_values = explainer(data)
    else:
        # shap_values = explainer.shap_values(data)
        shap_values = explainer(data)
    # print(shap_values)
    for i,j in [(0, 1), (3, 2), (4, 0), (4, 1)]:
        plt.clf()
        shap.dependence_plot(i, shap_values.values, data, feature_names=feature_names, interaction_index=j, show=False)
        plt.tight_layout()
        plt.savefig(f'dependence/{feature_names[i]}-{feature_names[j]}.png')
        plt.savefig(f'dependence/{feature_names[i]}-{feature_names[j]}.eps')

    if cls:
        false_list = []
        true_list = []
        for i in range(len(data_out)):
            # print(f'data {i}', data_out[i][0])
            if data_out[i][0]:
                true_list.append(i)
            else:
                false_list.append(i)

        random.shuffle(false_list)
        random.shuffle(true_list)



        exp = shap.Explanation(shap_values, explainer.expected_value, data=data, feature_names=feature_names)
        plt.clf()
        # print(false_list[0])
        # print(len(exp))
        shap.plots.waterfall(exp[false_list[0]], show=False)
        plt.tight_layout()
        plt.savefig(f'waterfull/cls_false_example.png')
        plt.savefig(f'waterfull/cls_false_example.eps')

        exp = shap.Explanation(shap_values, explainer.expected_value, data=data, feature_names=feature_names)
        plt.clf()
        shap.plots.waterfall(exp[true_list[0]], show=False)
        plt.tight_layout()
        plt.savefig(f'waterfull/cls_true_example.png')
        plt.savefig(f'waterfull/cls_true_example.eps')
    else:
        only_zr_list = []
        only_hf_list = []
        mixed_list = []
        for i in range(len(data_in)):
            if data_in[i][3] == 0:
                only_zr_list.append(i)
            elif data_in[i][2] == 0:
                only_hf_list.append(i)
            else:
                mixed_list.append(i)


        only_hf_list = sorted(only_hf_list, key=lambda x:data_out[x][0])
        mixed_list = sorted(mixed_list, key=lambda x:data_out[x][0])
        
        exp = shap.Explanation(shap_values[:,:,0], explainer.expected_value, feature_names=feature_names)
        plt.clf()
        shap.plots.waterfall(exp[only_hf_list[0]], show=False)
        plt.tight_layout()
        plt.savefig(f'waterfull/rgr_only_hfcl4.png')
        plt.savefig(f'waterfull/rgr_only_hfcl4.eps')

        plt.clf()
        shap.plots.waterfall(exp[mixed_list[0]], show=False)
        plt.tight_layout()
        plt.savefig(f'waterfull/rgr_mixed_example1.png')
        plt.savefig(f'waterfull/rgr_mixed_example1.eps')
        
        plt.clf()
        shap.plots.waterfall(exp[mixed_list[1]], show=False)
        plt.tight_layout()
        plt.savefig(f'waterfull/rgr_mixed_example2.png')
        plt.savefig(f'waterfull/rgr_mixed_example2.eps')

    # Plots
    
    # plt.clf()
    # shap.force_plot(explainer.expected_value, shap_values, feature_names, show=False)
    # plt.tight_layout()

    # if cls:
    #     plt.savefig(f'overall/cls_shap_force.png')
    #     plt.savefig(f'overall/cls_shap_force.eps')
    # else:
    #     plt.savefig(f'overall/rgr_shap_force.png')
    #     plt.savefig(f'overall/rgr_shap_force.eps')

    
    # print(np.abs(shap_values).mean(0))

    plt.clf()
    # print(shap_values.values.shape)
    # print(data.shape)
    # print(len(feature_names))
    if cls:
        shap.summary_plot(shap_values.values, data, feature_names, show=False)
    else:
        shap.summary_plot(shap_values.values[:,:,0], data, feature_names, show=False)
    plt.tight_layout()

    if cls:
        plt.savefig(f'overall/cls_shap_summary.png')
        plt.savefig(f'overall/cls_shap_summary.eps')
    else:
        plt.savefig(f'overall/rgr_shap_summary.png')
        plt.savefig(f'overall/rgr_shap_summary.eps')

    plt.clf()
    if cls:
        shap.summary_plot(shap_values.values, data, feature_names, plot_type="bar", color='#6699CC', show=False)
    else:
        shap.summary_plot(shap_values.values[:,:,0], data, feature_names, plot_type="bar", color='#6699CC', show=False)
    plt.tight_layout()

    if cls:
        plt.savefig(f'overall/cls_shap_summary1.png')
        plt.savefig(f'overall/cls_shap_summary1.eps')
    else:
        plt.savefig(f'overall/rgr_shap_summary1.png')
        plt.savefig(f'overall/rgr_shap_summary1.eps')



    corr_score = []
    for i in range(len(feature_names)):
        res = scipy.stats.pearsonr(data[:,i], np.array(data_out)[:,0])
        corr_score.append(res.statistic)
        # print(res.statistic)

    data = corr_score

    data[0], data[1] = data[1], data[0]
    feature_names[0], feature_names[1] = feature_names[1], feature_names[0]

    data = data[::-1]
    feature_names = feature_names[::-1]


    plt.clf()
    #绘图。
    fig, ax = plt.subplots()
    b = ax.barh(range(len(feature_names)), data, color='#6699CC')
    
    #为横向水平的柱图右侧添加数据标签。
    for rect in b:
        w = rect.get_width()
        ax.text(w-0.02, rect.get_y()+rect.get_height()/2, '%.2f' %
                w, ha='left', va='center')
    
    #设置Y轴纵坐标上的刻度线标签。
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    plt.tight_layout()
    if cls:
        plt.savefig(f'overall/cls_Pearson.png')
        plt.savefig(f'overall/cls_Pearson.eps')
    else:
        plt.savefig(f'overall/rgr_Pearson.png')
        plt.savefig(f'overall/rgr_Pearson.eps')

if __name__ == '__main__':
    analyze(cls=True)
    analyze(cls=False)