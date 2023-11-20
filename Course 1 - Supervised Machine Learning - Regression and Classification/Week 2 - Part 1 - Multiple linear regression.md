# Multiple linear regression

## Multiple features

Let's start by looking at the version of linear regression that looks at not just one feature, but a lot of different features. In the original version of linear regression, you had a single feature $x$, the size of the house, and you're able to predict $y$, the price of the house.

![](./img/2023-11-11-23-44-20.png)

But now, what if, on top of knowing the size of the house, you knew the number of bedrooms, the number of floors and the age of the home in years? This looks like it woud give you much more information with which to predict the price:

![](./img/2023-11-11-23-45-56.png)

We'l introduce new notation:

![](./img/2023-11-11-23-55-11.png)

See in the image above, how each feature is now represented by an $x$ with a subscript:

$x_j$, where $j$ is the $j^(th)$ feature
$n$ is the total number of features, in this case $4$

As before, we will use $x^{(i)}$ to denote the $i^{th}$ training example. But since now any $x^{(i)}$ will be a list of four numbers, $x$ will be a vector that includes all the features, and we represent it like so:

$$\vec{x}^{(i)}$$

![](./img/2023-11-12-00-02-33.png)

As a concrete example, when $i = 2$:

$$\vec{x}^{(2)} = [1416, 3, 2, 40]$$

To refer to a specific examples, we use:

${\vec{x}}_j^{(i)}$, where $j$ is the number of the feature in in $i^{th}$ training example.

So for example:

${\vec{x}}_3^{(2)} = 2$

![](./img/2023-11-12-00-08-14.png)

Let's now look what a model would look like:

![](./img/2023-11-12-00-09-05.png)

Notice that now for each feature, we have a parameter $w$, so for $n$ features, our model will look like:

$$f_{w,b} = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

Now we are going to introduce new notation to rewrite the equation above:

If we have a vector $\vec{w}$ that holds all the parameters $w$, like so:

$$ \vec{w} = [w_1\space w_2\space w_3\space ...\space w_n ]$$

In the formula above, we also have $b$, which is just a scalar number.

$\vec{w}$ and $b$ together are the **parameters of the model**.

Then we also have the vector $\vec{x}$ for all the features:

$$ \vec{x} = [x_1\space x_2\space x_3\space ...\space x_n ]$$

![](./img/2023-11-12-00-16-52.png)

So now we can rewrite the model as:

$$ f_{\vec{w}, b} (\vec{x}) = \vec{w} \cdot \vec{x} + b$$

where we use the dot product for the multiplication between the vectors.

![](./img/2023-11-12-00-19-53.png)

This is called **multiple linear regresssion**.

This is **NOT MULTIVARIATE REGRESSION**

## Vectorization - Part 1

When you're implementing a learning algorithm, using **vectorization** will not only make your code shorter but also make it run much more efficiently. Learning how to write vectorized code will allow you to also take advantage of modern numerical linear algebra libraries (such as numPy), as well as maybe even GPU hardware (graphics processing unit), hardware objectively designed to speed up computer graphics in your computer, but which can be used when you write vectorized code to help you execute your code much more quickly.

let's look at a concrete example of what vectorization means: 

Let's start with an example with parameters $\vec{w}$ and $b$, with $n = 3$ (number of features).

In Python code, using numPy, you define these like so:

```py
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array(10, 20, 30)
```

![](./img/2023-11-12-00-32-12.png)

Notice the difference in the indexing: **in linear algebra we start with 1, while in Python and numPy the first value is accessed with index 0**.

Now let's look an implementation without vectorization for this model:

$$f_{\vec{w},b} \vec{x} = w_1x_1 + w_2x_2 + w_3x_3 + b$$

In code, this would look like:

```py
f = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b
```

You could do it like this, but what if instead of $n =3$, you had $n = 10000$? It would be inefficient for you to code, and for the comptuer to compute. 

Another way, without using vectorization, but using summation and a for-loop in the code would be:

$$f_{\vec{w},b} (\vec{x})= \sum_{j=1}^n w_j  x_j + b$$

And the code:

```py
f = 0
for j in range(0, n):
  f = f + w[j] * x[j]
f = f + b
```

This still doesn't use vectorization and is not efficient.

So let's see how we can do this using **vectorization**:

$$f_{\vec{w},b} (\vec{x})= \vec{w} \cdot \vec{x} + b$$

```py
f = np.dot(w, x) + b
```

This implements the mathematical dot product between the vectors $\vec{w}$ and $\vec{x}$, and adds $b$ at the end.

Especially when $n$ is large, this will **run much faster than the two previous code examples**.

Vectorization has two distinct benefits:

- makes the code shorter and more concise, easier to understand
- makes the code run much faster 

The reason it runs faster is because, behind the scenes, the **numPy** dot functions uses parallel hardware in computers to do calculations (both in normal CPUs and in GPUs).

## Vectorization - Part 2

Let's figure out how vectorized calculations take much less time than doing non-vectorized calculations, by taking a look at how they work on a computer behind the scenes.

Let's first take a look at this for-loop, which runs without vectorization:

```py
for j in range(0, 16):
  f = f + w[j] * x[j]
```

We can see that `j` ranges from 0 to 15, and the computer runs the operations sequentially, one after the other.

![](./img/2023-11-12-18-01-00.png)

In contrast, the following function in numPy is implementened in the computer hardware with vectorization:

```py
np.dot(w, x)
```

Now, the computer gets all the values of vectors `w` and `x` and, **in a single step**, it multiples each pair within `w` and `x` with each other, parallely, at the same time. This is done in timestap $t_0$.

After that, in a subsequent step $t_1$, the computer takes the 16 numbers that are the result of the multiplications and adds them all together using specialized hardware, to add them altogether efficiently.

![](./img/2023-11-12-18-05-06.png)

This matters more when you're running algorithms on large data sets or trying to train large models, which is often the case with machine learning.

Let's now take a look at a concrete example, with 16 features and 16 parameters (apart from `b`, so 17 in total):

![](./img/2023-11-12-18-08-16.png)

And now you calculate 16 **derivative** terms for each of the 16 weights `w`, and store them in a numPy array:

![](./img/2023-11-12-18-09-07.png)

```py
w = np.array([0.5, 1.3, ..., 3.4])
d = np.array([0.3, 0,2, ..., 0.4])
```
For this example, let's first ignore the parameter **b**.

Now, we need to compute the update for each of these 16 parameters: so $w_j$ is updated by $w_j$ minus the learning rate $alpha$, times $d_j$ (the derivative), for $j$ being 1 through 16.

$$ w_j = w_j - 0.1 d_j \space\text{for}\space j = 1 ... 16 $$

In code, without vectorization you would be doing something like this:

```py
for j in range(16):
  w[j] = w[j] - 0.1*d[j]
```

![](./img/2023-11-12-18-17-09.png)

In contrast, with vectorization, you can imagine the computer parallel processing power like this: it takes all sixteen values in the vector and substracts, in parallel, `0.1` times the values in in vector `d` and assigns all sixteen calculations back to `w`, all at the same time and all in one step.

In code, you'd have:

```py
w = w - 0.1 * d
```

In the background, the comptuer takes these numPy arrays `w` and `d` and uses parallel processing to carry out the 16 calculations at the same time.

## Optional Lab - Python, NumPy and Vectorization

Importing numpy:

```py
import numpy as np
import time
```

$\vec{\mathbf{x}}$

Data creation routines in NumPy will generally have a first parameter which is the shape of the object. This can either be a single value for a 1-D result or a tuple (n,m,...) specifying the shape of the result. Below are examples of creating vectors using these routines.

**NumPy routines which allocate memory and fill arrays with value:**
```py
a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# np.zeros(4) :   a = [0. 0. 0. 0.], a shape = (4,), a data type = float64

a = np.zeros((4,))
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# np.zeros(4,) :  a = [0. 0. 0. 0.], a shape = (4,), a data type = float64

a = np.random.random_sample(4)
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# np.random.random_sample(4): a = [0.48086073 0.54874874 0.25354602 0.37859087], a shape = (4,), a data type = float64
```

Some data creation routines do not take a shape tuple:
**NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument**
```py
a = np.arange(4.)
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# np.arange(4.):     a = [0. 1. 2. 3.], a shape = (4,), a data type = float64

a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# np.random.rand(4): a = [0.36752644 0.77074865 0.44402351 0.37667294], a shape = (4,), a data type = float64
```

Values can be specified manually as well.
**NumPy routines which allocate memory and fill with user specified values**
```py
a = np.array([5,4,3,2]))
print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}"
# np.array([5,4,3,2]):  a = [5 4 3 2],     a shape = (4,), a data type = int64

a = np.array([5.,4,3,2])
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# np.array([5.,4,3,2]): a = [5. 4. 3. 2.], a shape = (4,), a data type = float64
```
These have all created a one-dimensional vector  `a` with four elements. `a.shape` returns the dimensions. Here we see a.shape = `(4,)` indicating a 1-d array with 4 elements.  

**Indexing in vectors**

```py
#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)
# [0 1 2 3 4 5 6 7 8 9]

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")
# a[2].shape: () a[2]  = 2, Accessing an element returns a scalar

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")
# a[-1] = 9

#indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
# The error message you'll see is:
# index 10 is out of bounds for axis 0 with size 10
```

**Slicing vectors**

Slicing creates an array of indices using a set of three values (`start:stop:step`). A subset of values is also valid. Its use is best explained by example:

```py
#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two 
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)

# a         = [0 1 2 3 4 5 6 7 8 9]
# a[2:7:1] =  [2 3 4 5 6]
# a[2:7:2] =  [2 4 6]
# a[3:]    =  [3 4 5 6 7 8 9]
# a[:3]    =  [0 1 2]
# a[:]     =  [0 1 2 3 4 5 6 7 8 9]
```

**Single vector operations**

```py
a = np.array([1,2,3,4])
print(f"a             : {a}")
# a             : [1 2 3 4]

# negate elements of a
b = -a 
print(f"b = -a        : {b}")
# b = -a        : [-1 -2 -3 -4]

# sum all elements of a, returns a scalar
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")
# b = np.sum(a) : 10

# get the mean of all values in a vector
b = np.mean(a)
print(f"b = np.mean(a): {b}")
# b = np.mean(a): 2.5

# exponentiate all values in vector to a given power
b = a**2
print(f"b = a**2      : {b}")
# b = a**2      : [ 1  4  9 16]
```

**Vector Vector element-wise operations**
Most of the NumPy arithmetic, logical and comparison operations apply to vectors as well. These operators work on an element-by-element basis. For example 
$$ c_i = a_i + b_i $$

```py
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")
# Binary operators work element wise: [0 0 6 8]
```

Of course, for this to work correctly, the vectors must be of the same size:

```py
#try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)
# The error message you'll see is:
# operands could not be broadcast together with shapes (4,) (2,) 
```

**Scalar-Vector operations**

Vectors can be 'scaled' by scalar values. A scalar value is just a number. The scalar multiplies all the elements of the vector.
```py
a = np.array([1, 2, 3, 4])

# multiply a by a scalar
b = 5 * a 
print(f"b = 5 * a : {b}")
# b = 5 * a : [ 5 10 15 20]
```

**Vector-vector dot product**

The dot product is a mainstay of Linear Algebra and NumPy. This is an operation used extensively in this course and should be well understood.

The dot product multiplies the values in two vectors element-wise and then sums the result.
Vector dot product requires the dimensions of the two vectors to be the same.

Let's implement our own version of the dot product below:

**Using a for loop**, implement a function which returns the dot product of two vectors. The function to return given inputs $a$ and $b$:
$$ x = \sum_{i=0}^{n-1} a_i b_i $$
Assume both `a` and `b` are the same shape.

```py
def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    # Loop through the number of features in any of the vectors,
    # (should match for both) - in this case, 4 (from 0 to 3 included)
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")
# my_dot(a, b) = 24
```

Note, the dot product is expected to return a scalar value. 

Let's try the same operations using `np.dot`.  

```py
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
# NumPy 1-D np.dot(a, b) = 24, np.dot(a, b).shape = () 

c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")
# NumPy 1-D np.dot(b, a) = 24, np.dot(a, b).shape = () 

```

**The Need for Speed: vector vs for loop**
We utilized the NumPy  library because it improves speed memory efficiency. Let's demonstrate:

```py
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory
```
And notice the results:
```py
# np.dot(a, b) =  2501072.5817
# Vectorized version duration: 157.3179 ms 
# my_dot(a, b) =  2501072.5817
# loop version duration: 9340.0886 ms 
```
So, vectorization provides a large speed up in this example. This is because NumPy makes better use of available data parallelism in the underlying hardware. GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel. This is critical in Machine Learning where the data sets are often very large.

**Vector-Vector operations in Course 1**
Vector-Vector operations will appear frequently in course 1. Here is why:
- Going forward, our examples will be stored in an array, `X_train` of dimension (m,n). This will be explained more in context, but here it is important to note it is a 2 Dimensional array or matrix (see next section on matrices).
- `w` will be a 1-dimensional vector of shape (n,).
- we will perform operations by looping through the examples, extracting each example to work on individually by indexing X. For example:`X[i]`
- `X[i]` returns a value of shape (n,), a 1-dimensional vector. Consequently, operations involving `X[i]` are often vector-vector.  

That is a somewhat lengthy explanation, but aligning and understanding the shapes of your operands is important when performing vector operations.

```py
# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")
# X[1] has shape (1,)
# w has shape (1,)
# c has shape ()
```

**Matrices**

Matrices, are two dimensional arrays. The elements of a matrix are all of the same type. In notation, matrices are denoted with capitol, bold letter such as $\mathbf{X}$. In this and other labs, `m` is often the number of rows and `n` the number of columns. The elements of a matrix can be referenced with a two dimensional index. In math settings, numbers in the index typically run from 1 to n. In computer science and these labs, indexing will run from 0 to n-1.  
<figure>
    <center> <img src="./img/C1_W2_Lab04_Matrices.png"  alt='missing'  width=900><center/>
    <figcaption> Generic Matrix Notation, 1st index is row, 2nd is column </figcaption>
</figure>

**Matrix creation**

The same functions that created 1-D vectors will create 2-D or n-D arrays. Here are some examples.

Below, the shape tuple is provided to achieve a 2-D result. Notice how NumPy uses brackets to denote each dimension. Notice further than NumPy, when printing, will print one row per line.

```py
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     
# a shape = (1, 5), a = [[0. 0. 0. 0. 0.]]

a = np.zeros((3, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 
# a shape = (3, 1), a = [[0.]
#  [0.]
#  [0.]]

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}")
# a shape = (1, 1), a = [[0.04997798]]
```

One can also manually specify data. Dimensions are specified with additional brackets matching the format in the printing above.
```py
a = np.array([[5], [4], [3]])
print(f" a shape = {a.shape}, np.array: a = {a}")
#  a shape = (3, 1), np.array: a = [[5]
#  [4]
#  [3]]

a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")
#  a shape = (3, 1), np.array: a = [[5]
#  [4]
#  [3]]
```
Notice how both matrices are actually the same.

**Indexing on matrices**

Matrices include a second index. The two indexes describe [row, column]. Access can either return an element or a row/column. See below:

```py
#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")
# a.shape: (3, 2), 
# a= [[0 1]
#  [2 3]
#  [4 5]]

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")
# a[2,0].shape:   (), a[2,0] = 4,     type(a[2,0]) = <class 'numpy.int64'> Accessing an element returns a scalar

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
# a[2].shape:   (2,), a[2]   = [4 5], type(a[2])   = <class 'numpy.ndarray'>
```
It is worth drawing attention to the last example. Accessing a matrix by just specifying the row will return a *1-D vector*.

**Reshape**  
The previous example used [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) to shape the array.  
```py
a = np.arange(6).reshape(-1, 2) 
```   
This line of code first created a *1-D Vector* of six elements. It then reshaped that vector into a *2-D* array using the reshape command. This could have been written:
```py
a = np.arange(6).reshape(3, 2)
```
To arrive at the same 3 row, 2 column array.
The -1 argument tells the routine to compute the number of rows given the size of the array and the number of columns.


**Slicing**

Slicing creates an array of indices using a set of three values (`start:stop:step`). A subset of values is also valid. Its use is best explained by example:

```py
#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")
# a = 
# [[ 0  1  2  3  4  5  6  7  8  9]
#  [10 11 12 13 14 15 16 17 18 19]]

#access 5 consecutive elements with (start:stop:step) from row of index 0
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")
# a[0, 2:7:1] =  [2 3 4 5 6] ,  a[0, 2:7:1].shape = (5,) a 1-D array

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")
# a[:, 2:7:1] = 
#  [[ 2  3  4  5  6]
#  [12 13 14 15 16]] ,  a[:, 2:7:1].shape = (2, 5) a 2-D array

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)
# a[:,:] = 
#  [[ 0  1  2  3  4  5  6  7  8  9]
#  [10 11 12 13 14 15 16 17 18 19]] ,  a[:,:].shape = (2, 10)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# a[1,:] =  [10 11 12 13 14 15 16 17 18 19] ,  a[1,:].shape = (10,) a 1-D array

# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
# a[1]   =  [10 11 12 13 14 15 16 17 18 19] ,  a[1].shape   = (10,) a 1-D array
```

## Gradient descent for multiple linear regression

Let's put together everything we've learnt to implement **gradient descent for multiple linear regression with vectorization**.

Let's first review what multiple linear regression looks like:

![](./img/2023-11-19-20-51-47.png)

Notice that now, $\vec{\mathbf{w}}$ and $\vec{\mathbf{x}}$ are vectors of length $n$, (multiplied by the dot product) while $b$ is still a number.

The whole formula for our multiple linear regression model is:

$$ f_{\vec{\mathbf{w}}, b} (\vec{\mathbf{x}})= \vec{\mathbf{w}} \cdot \vec{\mathbf{x}} + b$$

The same happens for our **cost function** and for our **gradient descent** algorithm:

![](./img/2023-11-19-21-02-02.png)

Let's take a look at how this looks like when you implement gradient descent, particularly at how the **derivate term** looks like.

So for **one feature**, we had:


$w = w - α \frac{∂}{∂w} J(w,b) $

$w = w - α ({1\over {m}} {\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}}))$

and:

$b = b - α \frac{∂}{∂w} J(w,b)$

$b = b - α ({1\over {m}} {\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)}}))$

So now when we have $n$ features, where $n \ge 2$, the gradient descent looks like this:

$w_1 = w_1 - α ({1\over {m}} {\sum_{i=1}^m (f_{\vec{\mathbf{w}},b}(\vec{\mathbf{x}}^{(i)}) - y^{(i)})x_1^{(i)}})) \space \text{where j = 1}$

\[
\vdots
\]

$w_n = w_n - α ({1\over {m}} {\sum_{i=1}^m (f_{\vec{\mathbf{w}},b}(\vec{\mathbf{x}}^{(i)}) - y^{(i)})x_1^{(i)}})) \space \text{where j = n}$

So this means that we have to update the parameters $w_1$, $w_2$... all the way to $w_n$.

And on top of that, also simultaneously update $b$:

$b = b - α ({1\over {m}} {\sum_{i=1}^m (f_{\vec{\mathbf{w}},b}(\vec{\mathbf{x}}^{(i)}) - y^{(i)})}))$. 

And that's it for gradient descent for multiple linear regression. Here's a summary;

![](./img/2023-11-19-21-15-18.png)

### Normal equation: alternative to gradient descent

![](./img/2023-11-19-21-16-51.png)

## Optional Lab: Multiple Variable Linear Regression

[[Link]](https://www.coursera.org/learn/machine-learning/ungradedLab/7GEJh/optional-lab-multiple-linear-regression/lab?path=%2Fnotebooks%2FC1_W2_Lab02_Multiple_Variable_Soln.ipynb)

[[Internal Link]](./Optional%20Lab%20-%20Multiple%20Variable%20Linear%20Regression.ipynb)

In this lab, you will extend the data structures and previously developed routines to support multiple features.

Our goals are: 

- Extend our regression model  routines to support multiple features
    - Extend data structures to support multiple features
    - Rewrite prediction, cost and gradient routines to support multiple features
    - Utilize NumPy `np.dot` to vectorize their implementations for speed and simplicity

**Problem statement**

The training dataset contains three examples with four features (size, bedrooms, floors and, age) shown in the table below.  Note that, unlike the earlier labs, size is in sqft rather than 1000 sqft. This causes an issue, which you will solve in the next lab!

| Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  

You will build a linear regression model using these values so you can then predict the price for other houses. For example, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.  

**Matrix X containing our examples**
Similar to the table above, examples are stored in a NumPy matrix `X_train`. Each row of the matrix represents one example. When you have $m$ training examples ( $m$ is three in our example), and there are $n$ features (four in our example), $\mathbf{X}$ is a matrix with dimensions ($m$, $n$) (m rows, n columns).


$$\mathbf{X} = 
\begin{pmatrix}
 x^{(0)}_0 & x^{(0)}_1 & \cdots & x^{(0)}_{n-1} \\ 
 x^{(1)}_0 & x^{(1)}_1 & \cdots & x^{(1)}_{n-1} \\
 \cdots \\
 x^{(m-1)}_0 & x^{(m-1)}_1 & \cdots & x^{(m-1)}_{n-1} 
\end{pmatrix}
$$
notation:
- $\mathbf{x}^{(i)}$ is vector containing example i. $\mathbf{x}^{(i)}$ $ = (x^{(i)}_0, x^{(i)}_1, \cdots,x^{(i)}_{n-1})$
- $x^{(i)}_j$ is element j in example i. The superscript in parenthesis indicates the example number while the subscript represents an element.  

Let's display the input data:

```py
# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

# X Shape: (3, 4), X Type:<class 'numpy.ndarray'>)
# [[2104    5    1   45]
#  [1416    3    2   40]
#  [ 852    2    1   35]]
# y Shape: (3,), y Type:<class 'numpy.ndarray'>)
# [460 232 178]
```

**Parameter vector w, b**

* $\mathbf{w}$ is a vector with $n$ elements.
  - Each element contains the parameter associated with one feature.
  - in our dataset, n is 4.
  - notionally, we draw this as a column vector

$$\mathbf{w} = \begin{pmatrix}
w_0 \\ 
w_1 \\
\cdots\\
w_{n-1}
\end{pmatrix}
$$
* $b$ is a scalar parameter.

$\mathbf{w}$ and $b$ will be loaded with some initial selected values that are near the optimal. $\mathbf{w}$ is a 1-D NumPy vector.

```py
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")
```

**Model Prediction With Multiple Variables**
The model's prediction with multiple variables is given by the linear model:

$$ f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b \tag{1}$$
or in vector notation:
$$ f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b  \tag{2} $$ 
where $\cdot$ is a vector `dot product`

To demonstrate the dot product, we will implement prediction using (1) and (2).

**Single Prediction element by element**
Our previous prediction multiplied one feature value by one parameter and added a bias parameter. A direct extension of our previous implementation of prediction to multiple features would be to implement (1) above using loop over each element, performing the multiply with its parameter and then adding the bias parameter at the end.

```py
def predict_single_loop(x, w, b): 
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i         
    p = p + b                
    return p
```

```py
# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
# x_vec shape (4,), x_vec value: [2104    5    1   45]

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
# f_wb shape (), prediction: 459.9999976194083
```
Note the shape of `x_vec`. It is a 1-D NumPy vector with 4 elements, (4,). The result, `f_wb` is a scalar.

**Single Prediction, vector**

Noting that equation (1) above can be implemented using the dot product as in (2) above. We can make use of vector operations to speed up predictions.

Recall from the Python/Numpy lab that NumPy `np.dot()`[[link](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)] can be used to perform a vector dot product.

```py
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p
```
```py
# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
# x_vec shape (4,), x_vec value: [2104    5    1   45]

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
# f_wb shape (), prediction: 459.99999761940825
```
Notice that the results and shapes are the same as the previous version which used looping. Going forward, `np.dot` will be used for these operations. Also notice that the prediction is now a single statement.

**Compute Cost With Multiple Variables**
The equation for the cost function with multiple variables $J(\mathbf{w},b)$ is:
$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \tag{3}$$ 
where:
$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{4} $$ 

In contrast to previous labs, $\mathbf{w}$ and $\mathbf{x}^{(i)}$ are vectors rather than scalars, now supporting multiple features.

Below is an implementation of equations (3) and (4). Note that this uses a *standard pattern for this course* where a for loop over all `m` examples is used.

```py
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost
```
```py
# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

# Cost at optimal w : 1.5578904880036537e-12
```

**Gradient Descent With Multiple Variables**

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$

where, n is the number of features, parameters $w_j$,  $b$, are updated simultaneously and where  

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
\end{align}
$$
* m is the number of training examples in the data set

    
*  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value

**Compute Gradient with Multiple Variables**
An implementation for calculating the equations (6) and (7) is below. There are many ways to implement this. In this version, there is an:
- outer loop over all m examples in the data
    - $\frac{\partial J(\mathbf{w},b)}{\partial b}$ for each example can be computed directly and accumulated
    - in a second loop, loop over all $n$ features:
        - $\frac{\partial J(\mathbf{w},b)}{\partial w_j}$ is computed for each $w_j$.

```py
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    # Loop over all examples (training data). 
    # Both dj_db and dj_dw need this.
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]
        # Loop over the n features. Only dj_dw needs this, 
        # because notice that when calculating dj_dw,
        # we multiply by X[i, j] which appears when we 
        # do th partial derivate with respect to w.
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw
```
```py
#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
# dj_db at initial w,b: -1.673925169143331e-06

print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
# dj_dw at initial w,b: 
#  [-2.73e-03 -6.27e-06 -2.22e-06 -6.92e-05]
```

**Gradient Descent With Multiple Variables**

The routine below implements equation (5) above.

```py
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing
```

Let's now test it:
```py
# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
```
```
Iteration    0: Cost  2529.46   
Iteration  100: Cost   695.99   
Iteration  200: Cost   694.92   
Iteration  300: Cost   693.86   
Iteration  400: Cost   692.81   
Iteration  500: Cost   691.77   
Iteration  600: Cost   690.73   
Iteration  700: Cost   689.71   
Iteration  800: Cost   688.70   
Iteration  900: Cost   687.69   
b,w found by gradient descent: -0.00,[ 0.2   0.   -0.01 -0.07] 
prediction: 426.19, target value: 460
prediction: 286.17, target value: 232
prediction: 171.47, target value: 178
```

![](./img/2023-11-19-22-21-58.png)

*These results are not inspiring*! Cost is still declining and our predictions are not very accurate. The next lab will explore how to improve on this.