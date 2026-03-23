# Variational approach using Neural Network (MLP ReLU activation)

## The variational approach
In quantum mechanics, the variational approach refers to a method to find the ground state (GS) for a certain time independent Hamiltonian.
This is done by noting that the GS will have the smallest energy, thus one may parametrise a function and minimise the energy w.r.t. function parameters.
The minimised function should then resemble the GS in a domain $\cal{D}$ if it is non-degenerate.
This is to say we introduce $\ket{f_{\vec{\lambda}}(\vec{r})}$ as our GS approximation where we find $f$ constrained to 

$\min_{\vec{\lambda}} \; \bra{f_{\vec{\lambda}}(\vec{r})} \hat{H} \ket{f_{\vec{\lambda}}(\vec{r})}$

This seems exactly like a Machine Learning problem, and since Neural Networks are Universal Function approximators, they seem like the perfect candidates for our function forms.

## The Neural Network
A wide and deep enough NN should be able to approximate any reasonable function, if anything at the cost of interpretability. Moreover, the training is easy enough. We introduce a physics and minimisation loss as:

$\cal{L}(\vec{\lambda}) = \alpha* \cal{L}_E + \beta* \cal{L}_{norm} + \gamma*\cal{L}_{b.c.}$  

where
$\cal{L}_E = \frac{|\bra{f_{\vec{\lambda}}(\vec{r})} \hat{H} \ket{f_{\vec{\lambda}}(\vec{r})}|^2}{|\langle{f_{\vec{\lambda}}(\vec{r})}\ket{f_{\vec{\lambda}}(\vec{r})}|^2}$ 
$\cal{L}_{norm} = (|\langle{f_{\vec{\lambda}}(\vec{r})}\ket{f_{\vec{\lambda}}(\vec{r})}| -1)^2$
$\cal{L}_{b.c.} = \frac{|f_{\vec{\lambda}}({\partial D})|^2}{|\langle{f_{\vec{\lambda}}(\vec{r})}\ket{f_{\vec{\lambda}}(\vec{r})}|}$

The first term works to minimise the energy, and the second ensures the wavefunction is normaliseable, and the third to impose Dirichlet boundary conditions. The hyperparameters, $({\alpha,\beta,\gamma})$ regulate the loss. It should be noted if we know any specific symmetries of the Hamiltonian we may change the loss to account for them.

The network used is a MLP with tanh activation layers. The width and depth of the network has been arbitrarily chosen for the most part. The only non arbitrary elements are the following:
 - Input : We restrict our input space to 1D for simplicity, though in theory this should be easy to generalise to 3D.
 - Output : Without losing generality, we can always let the output be a real, positive scalar.
 - Activation layers : It's important to use twice differentiable layers (i.e. not ReLU) as we calculate $\nabla^2$ in $\cal{\hat{H}}$.
