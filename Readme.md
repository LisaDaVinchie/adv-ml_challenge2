# Challenge 2

Explore the role of non linearity in neural networks by parametrising the non-linear *ReLU* function as
$$ReLU_\alpha = ReLU(x) - (1 - \alpha)ReLU(-x)$$
with $\alpha \in [0, 1]$; in this way:

- If $\alpha = 1$, then we get the normal *ReLU*
- If $\alpha = 0$, instead, we get $ReLU_\alpha = ReLU(x) - ReLU(-x) = x$, that is the linear function.

The task is the following: create a discriminative neural network, for example a 3-layers one that classifies images from the  *MNIST* or *FashionMNIST* datasets, using the $ReLU_\alpha$ activation function. Then, do one or both of the following tasks:

1. Fix one value of $\alpha$, $\alpha_i$, for each layer, initialising them as a learnable variable, that will be used for the activation function of that layer; put those values in a vector $\vec{\alpha} = (\alpha_1, \dots,\,\alpha_n)$.
Then, define the new loss function $\tilde{L}$ by imposing a penalty on non linearity on the original loss $L$.
$$\tilde{L} = L + \lambda ||\vec{\alpha}||_1$$
This loss will make the neural network linearize the layer whenever possible.
Once this is done, plot some of the filters, since linear transformations are very interpretable.

1. (Optional) Fix one value of $\alpha_i$ for each neuron

In both cases, plot the values of each $\alpha_i$ versus the number of epochs used to train the model and observe to which value do they converge. If one value of $\alpha_i$ oscillates even when the others converge, "switch off" $\lambda$ for that neuron, in order to make them converge to a value.
