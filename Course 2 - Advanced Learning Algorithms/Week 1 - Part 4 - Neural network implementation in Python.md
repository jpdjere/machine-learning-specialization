# Neural network implementation in Python

## Forward prop in a single layer

If we had to implement forward propagation ourselves from scratch in python, how would we go about doing so? 

Let's take a look at how we implement forward prop in a single layer, and for that, we're going to continue using the coffee roasting model. 

And let's look at how we would take an input feature vector $\mathbf{\vec{x}}$, and implement forward prop to get the output $\mathbf{\vec{a_2}}$.

 In this python implementation, we're going to use 1D arrays to represent all of these vectors and parameters, which is why we're going to use single square brackets here:

 ![](2024-01-19-14-09-29.png)

`np.array([200, 17])` is a 1D array in python rather than a 2D matrix, which is what we had when we had double square brackets. 

So the first value we need to compute is $\mathbf{a_1^{[1]}}$ which is the first activation value of the vector $\mathbf{\vec{a}^{[1]}}$. And it will have the formula:

$$\mathbf{a_1^{[1]}} = g(\mathbf{w_1^{[1]}} \cdot \mathbf{\vec{x}} + \mathbf{b_1^{[1]}} ) $$

We're going to use the convention on this slide that at a term like $\mathbf{w_1^{[2]}}$ is going to be represented in code as `w2_1`, with the layer number indicated first, and after the underscore, the number of the neuron. In this case, the second layer has only 1 neuron, so after the underscore, we'll only have a 1.

 So, to compute $\mathbf{{a_1}^{[1]}}$, we have parameters `w1_1` and `b1_1`:

 ```py
 w1_1 = np.array([1, 2])
 b1_1 = np.array([-1])
 ```
 We would then compute `z1_1` as the dot product between that parameter `w1_1` and the input `x`, and added to `b1_1`.  And then finally $a1_1$ is equal to g, the sigmoid function applied to `z1_1`. 
 
 ```py
z1_1 = np.dot(w1_1, x) + b1_1
a1_1 = sigmoid(z_1)
 ```
 
 In summary:

 ```py
w1_1 = np.array([1, 2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1, x) + b1_1
a1_1 = sigmoid(z_1)
 ```

![](2024-01-19-14-25-04.png)

And we can calculate similarly $\mathbf{{a_2}^{[1]}}$ and $\mathbf{{a_3}^{[1]}}$, the two remaining scalar numbers of the activation vector $\mathbf{\vec{a}^{[1]}}$:

For $\mathbf{{a_2}^{[1]}}$:
 ```py
w1_2_ = np.array([-3, 4])
b1_2_ = np.array([1])
z1_2_ = np.dot(w1_2, x) + b1_2
a1_2_ = sigmoid(z1_2_)
 ```
And for $\mathbf{{a_3}^{[1]}}$:
  ```py
w1_3 = np.array([5, -6])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3, x) + b1_3
a1_3 = sigmoid(z_3)
```

And we create our final activation vector `a1`:

```py
a1 = np.array([a1_1, a1_2, a1_3])
```

![](2024-01-19-14-29-45.png)

So now we've computed $\mathbf{\vec{a}^{[1]}}$, let's implement the second layer as well, to compute the output $\mathbf{\vec{a}^{[2]}}$:

$$\mathbf{a_1^{[2]}} = g(\mathbf{w_1^{[2]}} \cdot \mathbf{\vec{a}^{[1]}} + \mathbf{b_1^{[2]}} ) $$


```py
w2_1 = np.array([-7, 8, 9])
b2_1 = np.array([3])
z2_1 = np.dot(w2_1, a1) + b2_1
a2_1 = sigmoid(z2_1)
```

And that's it, that's how we implement forward prop using just Python and numpy. 

## General implementation of forward propagation

In the last section, we saw how to implement forward prop in Python, but by hard coding lines of code for every single neuron. 

Let's now take a look at the more general implementation of forward prop in Python.

What we'll do is** write a function to implement a dense layer**, that is **a single layer of a neural network**. We're going to define the `dense`` function, which takes as input:
1. the **activation** from the previous layer
2. the parameter w for the neurons in a given layer
2. the parameter b for the neurons in a given layer

```py
def dense(a_in, W, b):
```

Using the example from the previous video, the layer 1 has three neurons, and if $\mathbf{\vec{w_1}^{[1]}}$ and $\mathbf{\vec{w_2}^{[1]}}$ and $\mathbf{\vec{w_2}^{[1]}}$ are the following:

$$ \mathbf{\vec{w_1}^{[1]}} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \space  \space \space \mathbf{\vec{w_2}^{[1]}} = \begin{bmatrix} -3 \\ 4 \end{bmatrix} \space \space \space\space \mathbf{\vec{w_3}^{[1]}} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$$

What we have to do here is stack all of these vectors into a matrix: his is going to be a `2row x 3columns` matrix, where:

- the **first column** is the parameter $\mathbf{\vec{w_1}^{[1]}}$ 
- the **second column** is the parameter $\mathbf{\vec{w_2}^{[1]}}$ 
- the **third column** is the parameter $\mathbf{\vec{w_3}^{[1]}}$

```py
W = np.array([
      [1, -3, 5]
      [2, 4, -6]        # 2 rows x 3 colums matrix
])
```

Then in a similar way, if we have parameters $b$:

$$ \mathbf{b_1^{[1]}} = -1 \space  \space \space \mathbf{b_2^{[1]}} = 1\space \space \space\space \mathbf{b_3^{[1]}} = 2$$


we're going to stack these three numbers into a 1D array `b` as follows:

```py
b = np.array([-1, 1, 2])
```

What the dense function will do is take as inputs the activation from the previous layer, (which at the start of the network equal to $\mathbf{\vec{x}}$), or the activation from a later layer, as well as the $w$ parameters stacked in columns, as well as the $b$ parameters also stacked into a 1D array.

And what this function would do is input a to activation from the previous layer and will output the activations from the current layer. Let's step through the code for doing this:

```py
def dense(a_in, W, b):
  units = W.shape[1]
  a_out = np.zeros(units)
  for j in range(units):
    w = W[:, j]
    z = np.dot(w, a_in) + b[j]
    a_out[j] = sigmoid(z)
  return a_out
```

Let's add explanation to what each line is doing:

```py
def dense(a_in, W, b):
  # W is a 2rowsX3columns matrix. So the number of columns is 3.
  # That is equal to the number of units in this layer. We can extract
  # the number of columns by using (matrix).shape[1]
  units = W.shape[1]
  a_out = np.zeros(units)
  for j in range(units):
    w = W[:, j]
    z = np.dot(w, a_in) + b[j]
    a_out[j] = sigmoid(z)
  return a_out
```

Here's the code. First, units equals W.shape,1. W here is a two-by-three matrix, and so the number of columns is three. 

That's equal to the number of units in this layer. Here, units would be equal to three. Looking at the shape of w, is just a way of pulling out the number of hidden units or the number of units in this layer. 

Next, we set a to be an array of zeros with as many elements as there are units. In this example, we need to output three activation values, so this just initializes a to be zero, zero, zero, an array of three zeros. Next, we go through a for loop to compute the first, second, and third elements of a. 

For j in range units, so j goes from zero to units minus one. It goes from 0, 1, 2 indexing from zero and Python as usual. This command w equals W colon comma j, this is how we pull out the jth column of a matrix in Python. 

The first time through this loop, this will pull the first column of w, and so will pull out $w_1$,1. The second time through this loop, when we're computing the activation of the second unit, will pull out the second column corresponding to $w_1$, 2, and so on for the third time through this loop. Then we compute z using the usual formula, is a dot product between that parameter w and the activation that we have received, plus b, j. 

And then we compute the activation a, j, equals g sigmoid function applied to z. Three times through this loop and we compute it, the values for all three values of this vector of activation is a. Then finally we return a. 

What the dense function does is it inputs the activations from the previous layer, and given the parameters for the current layer, it returns the activations for the next layer. Given the dense function, here's how we can string together a few dense layers sequentially, in order to implement forward prop in the neural network. Given the input features x, we can then compute the activations $a_1$ to be $a_1$ equals dense of x, $w_1$, $b_1$, where here $w_1$, $b_1$ are the parameters, sometimes also called the weights of the first hidden layer. 

Then we can compute $a_2$ as dense of now $a_1$, which we just computed above. $W_2$, b-2 which are the parameters or weights of this second hidden layer. Then compute $a_3$ and $a_4$. 

If this is a neural network with four layers, then define the output f of x is just equal to $a_4$, and so we return f of x. Notice that here we're using W, because under the notational conventions from linear algebra is to use uppercase or a capital alphabet is when it's referring to a matrix and lowercase refer to vectors and scalars. So because it's a matrix, this is W. 

That's it. we now know how to implement forward prop yourself from scratch. we get to see all this code and run it and practice it yourself in the practice lab coming off to this as well. 

I think that even when we're using powerful libraries like TensorFlow, it's helpful to know how it works under the hood. Because in case something goes wrong, in case something runs really slowly, or we have a strange result, or it looks like there's a bug, our ability to understand what's actually going on will make we much more effective when debugging our code. When I run machine learning algorithms a lot of the time, frankly, it doesn't work. 

Sophie, not the first time. I find that my ability to debug my code to be a TensorFlow code or something else, is really important to being an effective machine learning engineer. Even when we're using TensorFlow or some other framework, I hope that we find this deeper understanding useful for our own applications and for debugging our own machine learning algorithms as well. 

That's it. That's the last required video of this week with code in it. In the next video, I'd like to dive into what I think is a fun and fascinating topic, which is, what is the relationship between neural networks and AI or AGI, artificial general intelligence? 

This is a controversial topic, but because it's been so widely discussed, I want to share with we some thoughts on this. When we are asked, are neural networks at all on the path to human level intelligence? we have a framework for thinking about that question. 

Let's go take a look at that fun topic, I think, in the next video.