# Variational approach using Neural Network (MLP ReLU activation)

## The variational approach
In quantum mechanics, the variational approach refers to a method to find the ground state (GS) for a certain time independent Hamiltonian.
This is done by noting that the GS will have the smallest energy, thus one may parametrise a function and minimise the energy w.r.t. function parameters.
The minimised function should then resemble the GS if it is non-degenerate.
This is to say we introduce $\ket{f_{\vec{\lambda}}(\vec{r})}$ as our GS approximation where we find $f$ constrained to 
$$
\min_{\vec{\lambda}} \; \bra{f_{\vec{\lambda}}(\vec{r})} \hat{H} \ket{f_{\vec{\lambda}}(\vec{r})} 
$$

This seems exactly like a Machine Learning problem, and since Neural Networks are Universal Function approximators, they seem like the perfect candidates for our function forms.

## The Neural Network
A wide and deep enough NN should be able to approximate any function, if anything at the cost of interpretability. Moreover, the training is easy enough. We introduce a physics and minimisation loss as follows:
$$\cal{L}(\vec{\lambda}) = |\bra{f_{\vec{\lambda}}(\vec{r})} \hat{H} \ket{f_{\vec{\lambda}}(\vec{r})}|^2 +  
{\mu}*(|\langle{f_{\vec{\lambda}}(\vec{r})}\ket{f_{\vec{\lambda}}(\vec{r})}|^2 -1)^2
$$
Where the first term works to minimise the energy, and the second ensures the wavefunction is normaliseable, with ${\mu}$ being a hyperparameter to regulate the loss. It should be noted if we know any specific symmetries of the Hamiltonian we may change the loss to account for them 