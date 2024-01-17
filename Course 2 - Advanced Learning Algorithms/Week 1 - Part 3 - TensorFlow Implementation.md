# TensorFlow implementation

## Inference in Code

TensorFlow is one of the leading frameworks to implementing deep learning algorithms. The other popular tool is PyTorch. But we're going to focus in this specialization on TensorFlow. 

Let's take a look at how we can implement inferencing code using TensorFlow: we're going to use a coffe roasting example.

When we're roasting coffee, two parameters we get to control are the temperature at which we're heating up the raw coffee beans to turn them into roasted coffee beans, as well as the duration or how long are we going to roast the beans. 

In the following slightly simplified example, we've created the datasets of different temperatures and different durations, as well as labels showing whether the coffee we roasted is good-tasting coffee. The cross icons means a positive result, where y equals 1 and corresponds to good coffee, and all the circles are negative results, and correspond to bad coffee.

![](2024-01-17-12-30-48.png)

A way to think of this dataset is:
- if we cook it at too low temperature, it doesn't get roasted and it ends up undercooked.
- if we cook it not for long enough, the duration is too short, it's also not a nicely roasted set of beans
- if we were to cook it either for too long or for too higher temperature, then we end up with overcooked beans. 

![](2024-01-17-12-31-52.png)

It's only points within this little triangle here that corresponds to good coffee. 

The task is given a feature vector $\mathbf{\vec{x}}$ with both temperature and duration, say 200 degrees Celsius for 17 minutes, how can we do inference in a neural network to get it to tell us whether or not this temperature and duration setting will result in good coffee or not?

![](2024-01-17-12-35-36.png)

For that, we write the following code:

```py
# Set X to be an numpy array of two numbers, 
# one for temperature and one for time
x = np.array([[200.0, 17.0]])

# Create the first "hidden" layer of the network using Dense
layer_1 = Dense(units=3, activation='sigmoid')

# Next, we compute the output of the first layer, 
# the activation a1, by applying the values of x 
# to the layer that we just created
a1 = layer_1(x)

# The activation vector will have 3 numbers, 
# coming each out of the 3 neurons
a1
# [0.2 0.7 0.3] --> just an example
```
 
Next, for the second hidden layer, Layer 2, we have one unit and again to sigmoid activation function, and we can then compute $\mathbf{\vec{a_2}}$ by applying this Layer 2 function to the activation values from Layer 1 to $\mathbf{\vec{a_1}}$.

```py
layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)
```

That will give we the value of $\mathbf{\vec{a_2}}$, which for the sake of illustration is maybe 0.8. Finally, if we wish to threshold it at `0.5`, then we can just test if $\mathbf{\vec{a_2}}$ is greater and equal to `0.5` and set $\hat{y}$ equals to one or zero positive or negative cross accordingly. That's how we do inference in the neural network using TensorFlow.

![](2024-01-17-12-44-52.png)

Notice that in the example above, we are missing the import of the library and loading the weight and biases, but we will see that later.

Let's look at one more example and we're going to go back to the handwritten digit classification problem. In this example, $\mathbf{\vec{x}}$ is a list of the pixel intensity values. 

![](2024-01-17-12-46-57.png)

```py
x = np.array([[0.0, ...245, ...240, ...0]]) # array with 64 values (8x8 pixels)
layer_1 = Dense(units=25, activation='sigmoid')
a1 = layer_1(x)

layer_2 = Dense(units=15, activation='sigmoid')
a2 = layer_2(a1)

layer_3 = Dense(units=1, activation='sigmoid')
a3 = layer_3(a2)

if a3 >= 0.5:
  yhat = 1
else:
  yhat = 0 
```

In the next section, let's take a look at how TensorFlow handles data. 

## Data in Tensorflow

Let understand now how data is represented in NumPy and in TensorFlow, so that as we're implementing new neural networks, we can have a consistent framework to think about how to represent our data. 

Unfortunately there are some inconsistencies between how data is represented in NumPy and in TensorFlow.  It's good to be aware of these conventions so that we can implement correct code.

Let's start by taking a look at how TensorFlow represents data. Let's see we have a data set like this from the coffee example:

![](2024-01-17-12-52-27.png)

I mentioned that we would write $\mathbf{\vec{x}}$ as follows: 

```py
x = np.array([[200.0, 17]])
```

So why do we have this double square bracket here? 

Let's first take a look at how NumPy stores vectors and matrices:

Here is a matrix with 2 rows and 3 columns. Notice that there are 2 rows and 1, 2, 3 columns. So we call this a **2 x 3 matrix**.

$$\begin{bmatrix} 1 \ 2 \ 3 \\ 4 \ 5 \ 6 \end{bmatrix} \text{2x3 matrix}$$

The convention is **the dimension of the matrix is written as the number of rows by the number of columns**. 

In Numpy code to store this matrix, this 2 x 3 matrix, we just write:

```py
x = np.array([[1, 2, 3],
              [4, 5, 6]])
```

Notice that the square bracket tells we that 1, 2, 3 is the first row of this matrix and 4, 5, 6 is the second row of this matrix. And then the outer open square bracket groups the first and the second row together.

Let's look at one more example:

$$\begin{bmatrix} 0.1 \ 0.2 \\ -3 \ -4 \\ {-0.5} \ -0.6 \\ 7.0 \ 8.0 \end{bmatrix} \text{4x2 matrix}$$

It's a 4 x 2 matrix. And so to store this in code, we will write:

```py
x = np.array([[0.1, 0.2],
              [-3.0 -4.0],
              [-0.5 -0.6],
              [7.0 8.0]])
```

Matrices can have different dimensions. We saw an example of an 2 x 3 matrix and the 4 x 2 matrix. 

But matrices can also be of other dimensions like 1 x 2 or 2 x 1. Let's see:

```py
x = np.array([[200, 17]])   --->  [200  17] ---> 1x2 matrix --> "Row vector"

x = np.array([[200],        --->  [200      ---> 2x1 matrix --> "Column vector"
              [17]])                17]

# a 1D vector (with no rows or columns)
x = np.array([200, 17])
```

The difference between using double square brackets like the two examples above versus a single square bracket like in the third example, is that:
- the two examples on top are 2D arrays where one of the dimensions happens to be 1. 
- the third example results in a 1D vector. It is just a 1D array that has no rows or columns.

So on a contrast this with what we had previously done in the first course, which was to write x like this with a single square bracket. 

So whereas in the first Course, when we're working with linear regression and logistic regression, we used these 1D vectors to represent the input features $\mathbf{\vec{x}}$, **with TensorFlow the convention is to use matrices to represent the data.** 

(TensorFlow was designed to handle very large datasets and by representing the data in matrices instead of 1D arrays, it lets TensorFlow be a bit more computationally efficient internally.)

So going back to our original example, for the first training example in this dataset, with features 200°C and 17 minutes, we were represented like so:

```py
np.array([[200.0, 17.0]])
```

So this is actually a **1 x 2 matrix** that happens to have one row and two columns to store the numbers 200, 17.

Going back to the code for carrying out for propagation or influence in the neural network. When we compute $\mathbf{\vec{a_1}}$ by applying `layer_1` to $\mathbf{\vec{x}}$, what shape and format has the result?

```py
x = np.array([[200.0, 17.0]])

layer_1 = Dense(units=3, activation='sigmoid')

a1 = layer_1(x)
# tf.Tensor([[0.2 0.7 0.3]], shape=(1, 4), dtype=float32)
```

Well, $\mathbf{\vec{a_1}}$ is actually going to be, because it has three numbers, a **1 x 3 matrix**.

And we see that Tensorflow uses the word **tensor**. So what is the tensor? 

**A tensor here is a data type that the TensorFlow team had created in order to store and carry out computations on matrices efficiently.** So for now, we can think of it as a matrix, although technically a tensor is a little bit more general than the matrix.

If we want to take $\mathbf{\vec{a_1}}$ which is a **tensor** and want to convert it back to **NumPy array**, we can do so with the function:

```py
# Tensorflow's conversion utility to numpy type
a1.numpy()
```
This will take the same data and return it in the form of a NumPy array rather than in the form of a TensorFlow array or TensorFlow matrix. 

Now let's take a look at what the activations output the second layer would look like:

![](2024-01-17-13-18-59.png)

We see that the result it's a 1x1 matrix.

We're used to loading data and manipulating data in NumPy, but when we pass a NumPy array into TensorFlow, **TensorFlow likes to convert it to its own internal format.** The tensor and then operate efficiently using tensors. And when we read the data back out we can keep it as a tensor or convert it back to a NumPy array. 

## Building a neural network

Let's talk about how to build a neural network in TensorFlow.

If we want to do forward prop, we initialize the data $\mathbf{\vec{x}}$, create layer one, and then compute $\mathbf{\vec{a_1}}$, then create layer two and compute $\mathbf{\vec{a_2}}$. So this was an explicit way of carrying out forward prop one layer of computation at the time:

```py
x = np.array([[200.0, 17.0]])

layer_1 = Dense(units=3, activation='sigmoid')
a1 = layer_1(x)

layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_1(a1)
```


It turns out that tensor flow has a different way of implementing forward prop as well as learning. Let's see different way of building a neural network in TensorFlow: same as before we're going to create layer one and create layer two. But now instead of we manually taking the data and passing it to layer one and then taking the activations from layer one and pass it to layer two, we can instead tell tensor flow that we would like it to take layer one and layer two and string them together to form a neural network. 

```py
x = np.array([[200.0, 17.0]])

layer_1 = Dense(units=3, activation='sigmoid')
layer_2 = Dense(units=1, activation='sigmoid')

model = Sequential([layer_1, layer_2])
```

With this `Sequential`` framework Tensorflow can do a lot of work for us:

Let's say we have a training set, for the coffee example, as $\mathbf{\vec{x}}$ in our Numpy array, a four by two matrix. We also have the target labels $\mathbf{\vec{y}}$.

```py
x = np.array([[200.0, 17.0],
             [120.0, 5.0],
             [425.0, 20.0],
             [212.0, 18.0]])
y = np.array([1, 0, 0, 1])
```     

If we want to train this neural network, all we need to do is call two functions: 

1. `model.compile(...)` - which takes some parameters that we'll see later
2. `model.fit(x, y)` -  which tells Tensorflow to take this neural network that we created by sequentially stringing together layers one and two, and to train it on the data, X and Y. 

And then we can do forward prop if we have a new example, say `X_new`, which is a Numpy array with the two features, temperature and time: `model.predict(X_new)`.

Also, by convention we don't explicitly assign the two layers to two variables, but create them directly in the array that Sequential takes. Let's see all together:

```py
x = np.array([[200.0, 17.0],
             [120.0, 5.0],
             [425.0, 20.0],
             [212.0, 18.0]])

y = np.array([1, 0, 0, 1])

model = Sequential([
  Dense(units=3, activation='sigmoid'),
  Dense(units=1, activation='sigmoid')
])

model.compile(...)
model.fit(x, y)

model.predict(X_new)
```

Let's redo this for the digit classification example as well:

![](2024-01-17-13-38-37.png)

```py
x = np.array([[0.0, ...245, ...240, ...0],
             [[0.0, ...200, ...184, ...0]]) 

y = np.array([1, 0])

model = Sequential([
  Dense(units=25, activation='sigmoid'),
  Dense(units=15, activation='sigmoid'),
  Dense(units=1, activation='sigmoid')
])

model.compile(...)
model.fit(x, y)

model.predict(X_new)
```

## Lab: Coffee Roasting in Tensorflow