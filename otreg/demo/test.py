import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../otreg')

from otreg import OTReg

def plot_and_save(y_actual, y_hat_list, filename):
    with PdfPages(filename) as pdf:
        for idx, y_hat in enumerate(y_hat_list):
            fig = plt.figure(figsize=(2*800/72,800/72))    
            ax = fig.add_subplot(projection='3d')

            ax.scatter3D(y_actual[:,0], y_actual[:,2], y_actual[:,1], s=0.5, c='r', alpha=0.5)

            ax.scatter3D(y_hat[:,0], y_hat[:,2], y_hat[:,1], s=0.5, c='b', alpha=0.5)

            ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
            ax.view_init(10, 45)
            ax.set_title(f'Results at iteration {idx}')

            pdf.savefig(fig)
            plt.close(fig)

def load_and_reg(method, file_label, i, j):
    data = torch.load(f'../data/{file_label}_{i}_{j}.pt')
    per = 0
    x = data[f'X{per}']
    y = data[f'Y{per}']

    otreg = OTReg(x, y)
    record_indices = [0,10,50,99] if method == 'TPS' else [0,10,20,30,40,49]
    results = otreg.register(method, record_indices=record_indices)

    filename = f'{file_label}_{method}_results_{i}_{j}.pdf'
    plot_and_save(y, results[-1], filename) 

    return results

if __name__ == "__main__":
    i = 19
    j = 2
    label = "female"
    # load_and_reg('TPS', label, i, j)
    load_and_reg('Gaussian', label, i, j)
