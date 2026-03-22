import torch.cuda

'''
Physical system parameters: domain and potential for Hamiltonian
'''
def V(x):
    return x**2
DOMAIN = [-10,10] 

'''
Inner product parameters: number of points on which to evaluate the inner product on (more = more accurate but slower)
'''
GRID_POINTS = 1000

'''
NN training parameters. Note we use ADAM optimiser.
'''
EPOCHS = 1000
LR = 0.001
BETAS = (0.9,0.99)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PTH = 'net_params_SHO'