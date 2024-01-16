# Neural network model

## Neural network layer

The fundamental building block of most modern neural networks is **a layer of neurons**. Now, we'll learn how to construct a layer of neurons and once we have that down, we'd be able to take those building blocks and put them together to form a large neural network.

Let's take a look at how a layer of neurons works. Here's the example we had from the demand prediction example where we had **four input features** that were set to **a layer of three neurons in the hidden layer**, that then sends its output to **the output layer with just one neuron**. 

Let's zoom in to the hidden layer to look at its computations:

![](./img/2024-01-16-11-41-48.png)


This hidden layer inputs four numbers and these four numbers are inputs to each of three neurons. **Each of these three neurons is just implementing a little logistic regression unit or a little logistic regression function.**

![](./img/2024-01-16-11-43-07.png)

Let's take the first neuron: it has two parameters, $w$ and $b$. In fact, to denote that, this is the first hidden unit, we're going to subscript this as $w_1$ and $b_1$. 


What this neuron does is output some activation value $a_1$, which is $g (\vec{\mathbf{w_1}} \vec{\mathbf{x_1}} + b_1)$, where this is the familiar z value that we have learned about in logistic regression and $g(z)$ is the familiar logistic function: $g(z) = \frac{1}{1 + e^(-z)}$.

![](./img/2024-01-16-11-48-57.png)


Let's suppose this ends up being a number $0.3$ and that's **the activation value a of the first neuron**. This means there's a 0.3 chance of this being highly affordable based on the input features. 

Now let's look at the second neuron. The second neuron has parameters $w_2$ and $b_2$, and these are the parameters of the second logistic unit. It computes $a_2$ equals the logistic function $g$, and this may be some other number, say $0.7$, because in this example, there's a 0.7 chance that the potential buyers will be aware of this t-shirt. 

![](./img/2024-01-16-11-51-17.png)


Similarly, the third neuron has a third set of parameters $w_3$, $b_3$. Similarly, it computes an activation value $a_3$ which equals $0.2$:

![](./img/2024-01-16-11-51-52.png)

And this, in this example, these three neurons output $0.3$, $0.7$, and $0.2$, and this vector of three numbers **becomes the vector of activation values $a$**, that is then passed to the final output layer of this neural network. 

![](./img/2024-01-16-11-52-51.png)

Now, when we build neural networks with multiple layers, **it'll be useful to give the layers different numbers**. By convention, the first layer is called layer 1 of the neural network and the second layer is called layer 2 of the neural network. **The input layer is also called layer 0**.

Today, there are neural networks that can have dozens or even hundreds of layers. So in order to introduce notation to help us distinguish between the different layers, we're going to use superscript square bracket $[number]$ to index into different layers. 

In particular, a superscript in square brackets $[1]$ is going to denote the output of layer 1 of this neural network, and similarly, $w_1$, $b_1$ here are the parameters of the first unit in layer 1 of the neural network, so we're also going to add a superscript in square brackets $[1]$ here, and $w_2$, $b_2$ are the parameters of the second hidden unit or the second hidden neuron in layer 1. 
Its parameters are also denoted here $w_1^{[1]}$ like so. Similarly, I can add superscripts square brackets to denote that these are the activation values of the hidden units of layer 1 of this neural network: $a^{[1]}$.

![](./img/2024-01-16-11-56-55.png)


So, the output of our layer 1 is the activation vector that we calculated, $a^{[1]}$. This output $a_1$ becomes the input to layer 2. So let's zoom into the computation of layer 2 of this neural network, which is also the output layer.

![](./img/2024-01-16-12-00-07.png)

The input to layer 2 is the output of layer 1, so $a_1$ is this vector $[0.3, 0.7, 0.2]$ that we just computed.

Because the output layer has just a single neuron, all it does is compute $a_1$ (that is, the output of this first and only neuron) as g, the sigmoid function applied to w $_1$ in a product with a^[1]:

$$ a_1^{[2]} = g(\mathbf{\vec{w_1}^{[2]}}  \mathbf{\vec{a_1}^{[2]}} + b_1) $$

Remember in this notation that:
- the **subscript denotes the number of the neuron in the layer** (in this case, we only have one neuron, so 1)
- the **superscript denotes the number of the layer** (here we are on our second and last layer, hence 2)


This results in a number, say $0.84$, then that becomes the output layer of the neural network.  (In this example, because the output layer has just a single neuron, this output is just a scalar, is a single number rather than a vector of numbers.)

![](./img/2024-01-16-12-09-24.png)

Once the neural network has computed our output there's one final optional step that we can choose to implement or not, which is if we want a binary prediction, 1 or 0, i.e. is this a top seller? Yes or no?

So we can take a_1^{[2]}, the number 0.84 that we computed, and threshold this at 0.5. 

![](./img/2024-01-16-12-12-52.png)

- If it's greater than 0.5, we can predict $\hat{y}$ equals 1  
- if it is less than 0.5, then predict our $\hat{y}$ equals 0.

## More complex neural networks

Let's use the layer from the previous section to build a more complex neural network so that the notation that we're using will become clearer and more concrete.


We'll use the following running example that we're going to use as an example of a more complex neural network. 

This network has four layers, not counting the input layer, which is also called Layer 0, where layers 1, 2, and 3 are hidden layers, and Layer 4 is the output layer, and Layer 0, as usual, is the input layer.

![](./img/2024-01-16-12-16-10.png)

**By convention, when we say that a neural network has four layers, that includes all the hidden layers and  the output layer, but we don't count the input layer.**.

![](./img/2024-01-16-12-17-21.png)

Let's zoom in to Layer 3, which is the third and final hidden layer to look at its computations.

![](./img/2024-01-16-12-17-52.png)

**Layer 3** inputs a vector, $\vec{\mathbf{a}}^{[2]}$ that was computed by the previous layer, and it outputs $\vec{\mathbf{a}}^{[3]}$, which is another vector. 

So, what is the computation that Layer 3 does in order to go from $\vec{\mathbf{a}}^{[2]}$ to $\vec{\mathbf{a}}^{[3]}$? 

If it has three neurons or we call it three hidden units, then it has parameters $w_1$, $b_1$, $w_2$, $b_2$, and $w_3$, $b_3$. And thus we can see that it computes the sigmoid function with those parameters, $g(z)$. And therefore the output of this layer is a vector, $\vec{\mathbf{a}}^{[3]}$,  comprising $\vec{\mathbf{a_1}}^{[3]}$, $\vec{\mathbf{a_2}}^{[3]}$, and $\vec{\mathbf{a_3}}^{[3]}$:



\[
\vec{\mathbf{a}}^{[3]} = \begin{bmatrix} \mathbf{a_1}^{[3]} \\ \mathbf{a_2}^{[3]} \\ \mathbf{a_3}^{[3]} \end{bmatrix} = \begin{bmatrix} g(\vec{\mathbf{w_1}}^{[3]} \cdot \mathbf{\vec{a}}^{[2]} + b_1^{[3]}) \\ g(\vec{\mathbf{w_2}}^{[3]} \cdot \mathbf{\vec{a}}^{[2]} + b_2^{[3]}) \\ g(\vec{\mathbf{w_3}}^{[3]} \cdot \mathbf{\vec{a}}^{[2]} + b_3^{[3]}) \end{bmatrix}
\]

Notice that in each calculation inside the sigmoid function, we **dot-multiply the whole activation vector from the previous layer**.


Now let's see the general form of this equation for **an arbitrary layer $l$** and for **an arbitrary unit (neuron) $j$**:


$$ \vec{\mathbf{a_j}}^{[l]} = g(\vec{\mathbf{w_j}}^{[l]} \cdot \vec{\mathbf{a_j}}^{[l-1]} + b_j^{[l]}) $$

In the context of a neural network, $g$ is called the **activation function**, because it outputs this activation value. So far, the only activation function we've seen is the sigmoid function other functions are possible.


## Inference: making predictions (forward propagation)

Let's take what we've learned and put it together into an algorithm to let our neural network make inferences or make predictions. This will be an algorithm called **forward propagation.**

We're going to use as a motivating example the task of handwritten digit recognition. For simplicity, we are just going to distinguish between the handwritten digits zero and one. So it's just **a binary classification problem** where we're going to input an image and classify: is this the digit zero or the digit one?

For the example of the slide, we're going to use an eight by eight image.

![](./img/2024-01-16-12-42-59.png)

And so this image of a one is this grid or matrix of eight by eight or 64 pixel intensity values where 255 denotes a bright white pixel and zero would denote a black pixel. And different numbers are different shades of gray in between the shades of black and white. 

**Given these 64 input features, we're going to use the neural network with two hidden layers:**

- the first hidden layer has **25 neurons** or 25 units
- the second hidden layer has **15 neurons** or 15 units. 
- the output layer gives the chance of this being 1 versus 0

![](./img/2024-01-16-12-44-45.png)

So let's step through the sequence of computations that in our neural network:  we need to go from the input $\mathbf{\vec{x}}$, (the 8x8 or 64 numbers vector) to the predicted probability  $\mathbf{\vec{a}^{[3]}}$. 

The first computation is to go from $\mathbf{\vec{x}}$ to $\mathbf{\vec{a}^{[1]}}$, and that's what the first layer of the first hidden layer does:

![](./img/2024-01-16-12-48-52.png)

$$ \mathbf{\vec{a}^{[1]}} = \begin{bmatrix} g(\vec{\mathbf{w_1}}^{[1]} \cdot \mathbf{\vec{a}}^{[0]} + b_1^{[1]}) \\ \vdots \\ g(\vec{\mathbf{w_{25}}}^{[1]} \cdot \mathbf{\vec{a}}^{[0]} + b_{25}^{[1]}) \end{bmatrix}$$


Notice that $\mathbf{\vec{a}^{[1]}}$ has 25 numbers because this hidden layer has 25 units.

Also, notice that in the slide, we used $\mathbf{\vec{X}}$, but in our formula, we used $\mathbf{\vec{a}^{[0]}}$. It's the same.


The next step is to compute $\mathbf{\vec{a}^{[2]}}$:

![](./img/2024-01-16-12-54-39.png)

$$ \mathbf{\vec{a}^{[2]}} = \begin{bmatrix} g(\vec{\mathbf{w_1}}^{[2]} \cdot \mathbf{\vec{a}}^{[1]} + b_1^{[2]}) \\ \vdots \\ g(\vec{\mathbf{w_{15}}}^{[2]} \cdot \mathbf{\vec{a}}^{[1]} + b_{15}^{[2]}) \end{bmatrix}$$

Notice that layer two has 15 neurons or 15 units, which is why the parameters here run from 1 to 15.

The final step is then to compute $\mathbf{\vec{a}^{[3]}}$ and we do so using a very similar computation, only now, this third layer (the output layer) has just one unit, which is why there's just one output here. So $\mathbf{\vec{a}^{[3]}}$ is just a scalar:

![](./img/2024-01-16-12-58-04.png)

$$ \mathbf{\vec{a}^{[2]}} = \begin{bmatrix} g(\vec{\mathbf{w_1}}^{[3]} \cdot \mathbf{\vec{a}}^{[2]} + b_1^{[3]}) \end{bmatrix} $$

And finally we can optionally take $\mathbf{\vec{a}^{[3]}}$ subscript one and threshold it at $0.5$ to come up with a binary classification label.

Since $\mathbf{\vec{a}^{[3]}}$ is the output of the neural network, we can also write it as $f(x)$:

$$ \mathbf{\vec{a}^{[3]}} = f(x) $$

**Because this computation goes from left to right, this algorithm is also called forward propagation because we're propagating the activations of the neurons.**

And this is in contrast to a different algorithm called **backward propagation or back propagation, which is used for learning.** 

Also, this type of neural network architecture where we have** more hidden units initially and then the number of hidden units decreases** as we get closer to the output layer is **a pretty typical choice when choosing neural network architectures**. 