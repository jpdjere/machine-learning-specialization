# Week 1 - Regression Model

## Linear regression model - Part 1

We'll see the first model of supervised learning: the **Linear Regression Model**.

This is just fitting a straight line to your data and it's probably the most widely used learning algorithm in the world today.

Many of the concepts here will also apply to other ML models we'll see later.

Let's use the example of prediciting the prices of a house based on its size. We're going to use a dataset from Portland, USA. 

Using this dataset, we can build a **linear regression model, which will fit a straight line to the data**, looking like this:

![](2023-09-04-23-27-34.png)

Using this model, you can see that if you want to predict the price of a house of 1250 sq. feet, it will cost around 220k dollars.

**Regression models** are called so because they predict numbers as the output, like prices in dollars.  **Any supervised learning model that predicts a number such as 220k, 1,5 or -33.2 is addressing what's called a regression problem**.

Linear regression is one example of many others.

An additional way of looking at the data from our house pricing example is via a table:

![](2023-09-04-23-30-43.png)

Each column represents the input and the output, the same as with the x-axis and y-axis of the graph.

### Terminology

**Training set:** the data that you use to train your model. (Notice that your client's house, from the example, is not part of the training set).

**Input variable, feature, or input feature:** the input, denoted with lowercase x. 

**Output variable or target variable:** the output, denoted with lowercase y.

See in the image below more: 

![](2023-09-04-23-36-53.png)

## Linear regression model - Part 2

Recall that in **supervised learning**, the **training set** contains both:
- **input features**: such as the size of the houses
- **output targets:** such as the price of the houses.

The output targets are the right answers to the model that we'll learn from.

To train the model, you feed the training set, both the input features and the output targets to your learning algorithm. Then your supervised learning algorithm will produce a **function**. We'll write this function as lowercase `f`. Historically, this function used to be called a **hypothesis**, but we'll call it a **function** `f` in this class. 

The job with `f` is to take a new input `x` and output an estimate or a prediction, which we will call `y-hat` (`ŷ`).

In ML, the convention is that **`ŷ` is the estimate or the prediction for y**.

The function `f` is called the **model**.

`x` is called the **input or input feature** and the **output of the model is the prediction `ŷ`**.

The model's prediction is the estimated value of y. When the symbol is just the letter `y`, then that refers to the **target**, which is the **actual true value in the training set.**

In contrast, `ŷ` is an estimate, which **may or may not be the actual true value.** In the case of the houses, the true price of the house is unknown until it is sold.

The model `f`, given a size, outputs a price which is the **estimator or prediction of what the true price will be.**

![](2023-09-04-23-48-35.png)

Now, **how do represent `f`**? What is the **math formula we use to compute `f`**?

For now, we are using a straight line, and the line is represented by:

![](2023-09-04-23-51-25.png)

**Why are we choosing a linear function (a straight line)** instead of some non-linear function like a curve or a parabola?

Sometimes you want to fit more complex non-linear functions as well, like a curve. But since this linear function is relatively simple and easy to work with, we'll first use a line as a foundation that will eventually help us to get to more complex models that are non-linear. 

This particular model has a name, it's called **linear regression**. More specifically, this is **linear regression with one variable**, where the phrase "one variable" means that there's a single input variable or feature `x`, namely the size of the house.

Another name for a linear model with one input variable is **univariate linear regression**.

![](2023-09-04-23-54-26.png)



## Optional Lab: Model Representation

![](2023-09-04-23-58-59.png)

### Problem statement

As in the lecture, you will use the motivating example of housing price prediction.
This lab will use a simple data set with only two data points - a house with 1000 square feet sold for `$`300,000 and a house with 2000 square feet sold for $500,000. These two points will constitute our data or training set. In this lab, the units of size are 1000 sqft and the units of price are 1000s of dollars.

| Size (1000 sqft)     | Price (1000s of dollars) |
| -------------------| ------------------------ |
| 1.0               | 300                      |
| 2.0               | 500                      |

You would like to fit a linear regression model (shown above as the blue straight line) through these two points, so you can then predict price for other houses - say, a house with 1200 sqft.

```py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# We will use m to denote the number of training examples. 
# Numpy arrays have a .shape parameter. x_train.shape returns a python tuple with an entry for each dimension. 
# x_train.shape[0] is the length of the array and number of examples as shown below.

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# One can also use the Python `len()` function as shown below.
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")
```

We will use (x$^{(i)}$, y$^{(i)}$) to denote the $i^{th}$ training example. Since Python is zero indexed, (x$^{(0)}$, y$^{(0)}$) is (1.0, 300.0) and (x$^{(1)}$, y$^{(1)}$) is (2.0, 500.0). 

To access a value in a Numpy array, one indexes the array with the desired offset. For example the syntax to access location zero of `x_train` is `x_train[0]`.

Run the next code block below to get the $i^{th}$ training example.

```py
i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```

You can plot these two points using the `scatter()` function in the `matplotlib` library, as shown in the cell below. 
- The function arguments `marker` and `c` show the points as red crosses (the default is blue dots).

You can use other functions in the `matplotlib` library to set the title and labels to display.

```py
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
```
![](2023-09-05-00-08-07.png)

 As described in lecture, the model function for linear regression (which is a function that maps from `x` to `y`) is represented as 

$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$

The formula above is how you can represent straight lines - different values of $w$ and $b$ give you different straight lines on the plot.

Let's try to get a better intuition for this through the code blocks below. Let's start with $w = 100$ and $b = 100$. 

```py
w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")
```

Now, let's compute the value of $f_{w,b}(x^{(i)})$ for your two data points. You can explicitly write this out for each data point as - 

- for $x^{(0)}$, `f_wb = w * x[0] + b`

- for $x^{(1)}$, `f_wb = w * x[1] + b`

For a large number of data points, this can get unwieldy and repetitive. So instead, you can calculate the function output in a `for` loop as shown in the `compute_model_output` function below.
> **Note**: The argument description `(ndarray (m,))` describes a Numpy n-dimensional array of shape (m,). `(scalar)` describes an argument without dimensions, just a magnitude.  
> **Note**: `np.zero(n)` will return a one-dimensional numpy array with $n$ entries   

```py
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
```

Now let's call the `compute_model_output` function and plot the output.

```py
tmp_f_wb = compute_model_output(x_train, w, b)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```
![](2023-09-05-00-13-29.png)
As you can see, setting  𝑤=100 and  𝑏=100 does not result in a line that fits our data.

If we adjust so that:

- $w = 200$
- $b = 100$

We now get:
![](2023-09-05-00-14-30.png)

Now that we have a model, we can use it to make our original prediction. Let's predict the price of a house with 1200 sqft. Since the units of $x$ are in 1000's of sqft, $x$ is 1.2.

```py
w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")
```

$340 thousand dollars