import torch
import torch.nn as nn

#neural network : MLP with tanh activation. Input scalar output scalar
class NN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(nn.Linear(1,20),
                               nn.Tanh(),
                               nn.Linear(20,20),
                               nn.Tanh(),
                               nn.Linear(20,10),
                               nn.Tanh(),
                               nn.Linear(10,1)
                               )
    def forward(self,x):
        x=self.layers(x)
        return x**2

class cust_loss():
    '''
    Custom loss to minimise energy subject to normalisation and dirichlet boundary conditions
    '''
    def __init__(self,domain:list,f:callable,V:callable,device:str,grid_points:int=1,e_weight=1,bc_weight=1,norm_weight=5):
            """
            Packages loss function for a specified function and domain.

            Use methods to access loss, energy, norm of input function or compute inner product of 2 functions 

            Parameters
            ----------
            domain : list
                Specify start and end point of domain. Note: must have tensors as elements

            f : function
                Function which to evaluate loss for. Note: must be composed of torch functions, and doesn't have to be normalised

            V : function
                Potential function for the Hamiltonian to be evaluated with. Note units should be kept in form with h_bar²/2m = 1
            
            grid_points : int
                Number of points on grid with which to evaluate integral inner products
            
            device : str
                Where to evaluate operations (cpu/cuda)
            """
            self.domain  = domain
            self.grid_points = grid_points+1
            self.e_weight = e_weight
            self.bc_weight = bc_weight
            self.norm_weight = norm_weight
            self.f = f
            self.V = V
            self.device = device
    def Ham(self,x):
        '''
        Computes Hamiltonian for specified function at point x. Change here to your specific hamiltonian

        Parameters
        ----------
        x : float
            Point to evaluate Hamiltonian. Must be a tensor with requires_grad=True
        
        Returns
        ----------
        Tensor [H.f](x)
        '''

        psi = self.f(x)
        p = torch.autograd.grad(psi.sum(),x,create_graph=True)[0]
        #ddx -> kinetic energy
        T = torch.autograd.grad(p.sum(),x,create_graph=True)[0]
        #POTENTIAL ENERGY
        V = self.f(x)*self.V(x)
        return -T + V
    
    def braket(self,f,g):
        '''
        Computes inner product between f and g over specified domain.

        Assumes real valued f and g

        Parameters
        ----------
        f : function
            First function to be evaluated in the inner product
        g : function
            Other function to be evaluated in the inner product

        Returns
        ----------
        Tensor <f,g>
        '''
        #grid points
        grid = torch.linspace(self.domain[0],self.domain[1],self.grid_points,device=self.device,requires_grad=True,dtype=torch.float).unsqueeze(-1)
        grid_eval = (f(grid)*g(grid)).squeeze(-1)
        #weights for simpsons rule: have to be 1,2,4,2,4,2,...1
        weights = torch.arange(0,self.grid_points,dtype=torch.float,device=self.device)%2
        weights = weights*2 + 2
        weights[0] = 1
        weights[-1] = 1
        #step size
        h = (self.domain[1]-self.domain[0])/(self.grid_points-1)
        #using simpsons composite 1/3 rule
        return torch.dot(grid_eval,weights)*h/3
    
    def get_energy(self):
        '''
        Computes expected value of energy for the specified function

        Returns
        ----------
        Tensor <f|H|f>/<f,f>

        '''
        ket = self.Ham
        bra = self.f
        energy = self.braket(bra,ket)/self.get_norm()**2
        return energy
    
    def get_norm(self):
        '''
        Computes norm of specified function
        
        Returns
        ----------
        Tensor : sqrt(<f,f>)
        '''
        ket = self.f
        bra = self.f
        norm_sq = self.braket(bra,ket)
        return norm_sq**0.5
    
    def get_loss(self):
        r"""
        Computes loss for specified function

        The loss is defined as 
        L(f) = w1*(E_f)^2 + w2*(BC1^2 + BC2^2) + w3*(<f,f>-1)^2
        where E_f := <f|H|f>/<f,f> (expected energy of f)
              BC  := f(edge(domain))/|f| imposing f -> 0 at edges of domain
              (<f,f>-1)^2 imposes normalisation of f
              w_i := relative weight of each loss component

        Returns
        ----------
        Tensor : L(f)

        """
        #energy loss
        norm = self.get_norm()
        energy = self.get_energy()
        bc1 = self.f(self.domain[0].unsqueeze(-1))**2/norm
        bc2 = self.f(self.domain[1].unsqueeze(-1))**2/norm
        #enforcing boundary conditions:
        #psi ->0 at +- infinity or domain edges

        bc_loss = self.bc_weight*(bc1+bc2)
        energy_loss = self.e_weight*energy**2
        norm_loss = self.norm_weight*(norm-1)**2

        total_loss = energy_loss+bc_loss+norm_loss
        return total_loss
