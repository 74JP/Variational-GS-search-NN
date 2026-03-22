import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from neural_net import NN,cust_loss
from config import EPOCHS,DOMAIN,V,GRID_POINTS,LR,BETAS,DEVICE,SAVE_PTH


net = NN().to(DEVICE)
net.train()
optimiser = Adam(net.parameters(),lr=LR,betas=BETAS)
domain = torch.tensor((DOMAIN[0],DOMAIN[1]),dtype=torch.float,device=DEVICE)
loss_fn = cust_loss(domain=domain,grid_points=GRID_POINTS,f=net,V=V,device=DEVICE)

'''
Training loop.
'''
loss_hist = []
energy_hist = []
norm_hist = []
print('Begun training')
for epoch in range(EPOCHS):
    
    #recording training
    loss = loss_fn.get_loss()
    energy = loss_fn.get_energy()
    norm = loss_fn.get_norm()
    loss_hist.append(loss.item())
    energy_hist.append(energy.item())
    norm_hist.append(loss.item())
    
    #backprop
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()

    if epoch % int(EPOCHS/20) == 0:
        print(f'Epoch:{epoch} loss:{loss.item()} energy:{energy} norm:{norm}')

print('Finished training. Plotting training history')
#plotting wavefunction
with torch.no_grad():
    net.to('cpu')
    net.eval()
    points = torch.linspace(DOMAIN[0],DOMAIN[1],GRID_POINTS,device='cpu').unsqueeze(-1)
    ground_state = net(points).squeeze(-1)
    def plot_training(energy_hist, norm_hist, loss_hist, ground_state, x_space=None):
        # --- Figure 1: training histories ---
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(energy_hist))

        ax1.plot(x, energy_hist, label="Energy")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Energy")

        ax2 = ax1.twinx()
        ax2.plot(x, norm_hist, linestyle=":", label="Norm")
        ax2.set_ylabel("Norm")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.plot(x, loss_hist, linestyle="-.", label="Loss")
        ax3.set_yscale("log")
        ax3.set_ylabel("Loss (log scale)")

        lines, labels = [], []
        for ax in [ax1, ax2, ax3]:
            l, lab = ax.get_legend_handles_labels()
            lines += l
            labels += lab

        ax1.legend(lines, labels, loc="best")
        ax1.set_title("Training History")
        plt.tight_layout()

        # --- Figure 2: ground state over space ---
        gs = np.asarray(ground_state).squeeze()

        if x_space is None:
            x_space = np.arange(len(gs))
        else:
            x_space = np.asarray(x_space).squeeze()

        fig2, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x_space, gs, label="Ground State")
        ax.set_xlabel("Space")
        ax.set_ylabel("Ground State")
        ax.set_title("Ground State Over Space")
        ax.legend()
        plt.tight_layout()

        plt.show()

    plot_training(energy_hist=energy_hist,norm_hist=norm_hist,loss_hist=loss_hist,ground_state=ground_state)

    try:
        torch.save(net.state_dict(),SAVE_PTH)
        print(f'Succesfully saved weights to {SAVE_PTH}')
    except Exception as e:
        print(f'Failed to save weights. Exception: {e}')



