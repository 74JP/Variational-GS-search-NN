# NN for finding 1-D Ground State solutions

## The variational approach
In quantum mechanics, the variational approach refers to a method to find the ground state (GS) for a certain time independent Hamiltonian.
This is done by noting that the GS will have the smallest energy, thus one may parametrise a function and minimise the energy w.r.t. function parameters.
The minimised function should then resemble the GS in a domain $\cal{D}$ if it is non-degenerate.
This is to say we introduce $\ket{f_{\vec{\lambda}}(\vec{r})}$ as our GS approximation where we find $f$ constrained to 

$\min_{\vec{\lambda}} \; \bra{f_{\vec{\lambda}}(\vec{r})} \hat{H} \ket{f_{\vec{\lambda}}(\vec{r})}$

This seems exactly like a Machine Learning problem, and since Neural Networks are Universal Function approximators, they seem like the perfect candidates for our function forms.

## The Neural Network
A wide and deep enough NN should be able to approximate any reasonable function, if anything at the cost of interpretability. Moreover, the training is easy enough. 
The loss function is defined as:

$\mathcal{L}(\vec{\lambda}) = \alpha \mathcal{L}_E + \beta \mathcal{L}_{\text{norm}} + \gamma \mathcal{L}_{\text{b.c.}}$

where:

$\mathcal{L}_E = \frac{\left|\langle f_{\vec{\lambda}}(\vec{r}) | \hat{H} | f_{\vec{\lambda}}(\vec{r}) \rangle\right|^2}{\left|\langle f_{\vec{\lambda}}(\vec{r}) | f_{\vec{\lambda}}(\vec{r}) \rangle\right|^2}$

$\mathcal{L}_{\text{norm}} = \left(\left|\langle f_{\vec{\lambda}}(\vec{r}) | f_{\vec{\lambda}}(\vec{r}) \rangle\right| - 1\right)^2$

$\mathcal{L}_{\text{b.c.}} = \frac{\left|f_{\vec{\lambda}}(\partial D)\right|^2}{\left|\langle f_{\vec{\lambda}}(\vec{r}) | f_{\vec{\lambda}}(\vec{r}) \rangle\right|}$

The first term works to minimise the energy, and the second ensures the wavefunction is normaliseable, and the third to impose Dirichlet boundary conditions. The hyperparameters, $({\alpha,\beta,\gamma})$ regulate the loss. It should be noted if we know any specific symmetries of the Hamiltonian we may change the loss to account for them.

The network used is a MLP with tanh activation layers. The width and depth of the network has been arbitrarily chosen for the most part. The only non arbitrary elements are the following:
 - Input : We restrict our input space to 1D for simplicity, though in theory this should be easy to generalise to 3D.
 - Output : Without losing generality, we can always let the output be a real, positive scalar.
 - Activation layers : It's important to use twice differentiable layers (i.e. not ReLU) as we calculate $\nabla^2$ in $\cal{\hat{H}}$.
---
## Configuration

The default configuration defines the physical system, numerical setup, and training parameters used in the model.

---

### Physical system

```python
def V(x):
    return x**2

DOMAIN = [-10, 10]
```

* `V(x)`: Potential function defining the Hamiltonian (default: harmonic oscillator (V(x)=x^2))
* `DOMAIN`: Spatial domain over which the system is evaluated

---

### Inner product / discretisation

```python
GRID_POINTS = 1000
```

* `GRID_POINTS`: Number of discretisation points used to approximate inner products

  * Higher values increase accuracy
  * Lower values improve speed

---

### Training parameters

```python
EPOCHS = 1000
LR = 0.001
BETAS = (0.9, 0.99)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PTH = 'net_params_SHO'
```

* `EPOCHS`: Number of training iterations
* `LR`: Learning rate for the Adam optimiser
* `BETAS`: Adam optimiser momentum parameters
* `DEVICE`: Uses GPU (`cuda`) if available, otherwise CPU
* `SAVE_PTH`: File name/path used to save trained model parameters

---

### Usage

The configuration is loaded automatically when running the training script. Simply run 'training.py' to train the network and after 'plotting.py' to see the output
To modify behaviour, edit the parameters directly in the config file.

---


