'''
@Time : 2024/5/13 16:47
@Auth : Qizhi Li
'''
import numpy as np
import matplotlib.pyplot as plt


def visualize_plots(file_name):
    iters, ce, md_adv, scl = [], [], [], []
    with open('../results/{}.txt'.format(file_name), 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.rstrip('\n').split('\t')[:-1]
        for part in parts:
            k, v = part.split(': ')
            if k == 'Iter':
                iters.append(int(v))
            elif k == 'ce':
                ce.append(float(v))
            elif k == 'md_adv':
                md_adv.append(float(v))
            elif k == 'scl':
                scl.append(float(v))

    iters = np.array(iters)
    ce_losses = np.array(ce)
    md_adv_losses = np.array(md_adv)
    scl_losses = np.array(scl)

    if 'loss' in file_name:
        label = 'loss'
        ylabel = 'Losses'
    else:
        label = 'weight'
        ylabel = 'Weights'

    plt.plot(iters, ce_losses, label='CE {}'.format(label))
    plt.plot(iters, md_adv_losses, label='MD_ADV {}'.format(label))
    plt.plot(iters, scl_losses, label='SCL {}'.format(label))

    plt.xlabel('Training Steps')
    plt.ylabel(ylabel)

    plt.legend()
    plt.savefig('../results/{}_curve.pdf'.format(label))
    plt.show()


def visualize_losses_curve():
    visualize_plots('losses_curve')


def visualize_loss_weights_curve():
    visualize_plots('weight_curve')


def heat_map():
    # 创建数据
    data = np.array(
        [[0.0683, 0.1017, 0.0333, 0.46, 0.1658],
         [0.9417, 1.0667, 1.1717, 1.1650, 1.0863],
         [0.0933, 0.3283, 0.0933, 0.0583, 0.1433],
         [0.0583, 0.0283, 0.0367, 0.05, 0.0433]]
    )

    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制热力图
    cax = ax.matshow(data, cmap='OrRd')

    # 添加颜色条
    fig.colorbar(cax)

    # 设置轴标签
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(['B', 'D', 'E', 'K', 'Avg.'])
    ax.set_yticklabels(['PDA', 'EAGLE', 'TACIT', 'ours'])

    # 旋转X轴标签
    plt.xticks()

    # 在每个格子中显示值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f'{data[i, j]}', ha='center', va='center', color='black')

    # 显示图形
    plt.savefig(r'D:\pycharm\workspace\KernelWord\results\PAD.pdf')
    plt.show()
