from neural_net import NN, cust_loss
from config import DOMAIN, SAVE_PTH,V
import torch
import numpy as np
import matplotlib.pyplot as plt

print('Loading neural net and parameters')

grid = torch.linspace(DOMAIN[0],DOMAIN[1],1000,requires_grad=True).unsqueeze(-1)
#load neural net
net = NN().to('cpu')
net.eval()
try:
    net.load_state_dict(torch.load(SAVE_PTH))
    print('Successfully loaded from :',SAVE_PTH)
except Exception as e:
    print('Failed to load parameters:',e)



#expected energy of this wavefucntion
loss_fn = cust_loss(torch.tensor(DOMAIN,device='cpu'),f=net,V=V,device='cpu')
E = loss_fn.get_energy()
psi = net(grid).squeeze(0).detach()
grid = grid.detach()
print(E)

#--------------plotting-----------------------

plt.figure(figsize=(8, 5))

# Plot psi vs grid
plt.plot(grid, psi, linewidth=2, label=r'$\psi$')

# Labels and title
plt.xlabel('Grid')
plt.ylabel(r'$\psi$')
plt.title(f'Expected Energy is {E:.5f}')

# Optional styling
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Tight layout for better spacing
plt.tight_layout()

# Show plot
plt.show()
