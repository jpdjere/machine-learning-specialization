# Cost function for logistic regression

## Cost function for logistic regression

Remember: **the cost function gives you a way to measure how well a specific set of parameters fit the training data**, and therefore a way to choose better parameters.

We'll see first how the sqaured error cost function is **not an ideal cost function for logistic regression.** And we'll choose a better cost function.

![](2024-01-04-09-48-47.png)

Above we see the training set for a logistic regression model.

Each row corresponds to a patient that paid a visit to the doctor, and their diagnosis.

- $i$ is the index of the training example
- $m$ is that amount of training examples
- $j$ is the index of the feature
- $n$ is the amount of features

So we have:

- features that go from $x_1$ up to $x_n$
- since this is a **binary classification task**, the target label $y$ takes values of either `0` or `1`

And the logistic regression model is defined by the equation:

$$ f_{\vec{\mathbf{w}}, b} (\vec{\mathbf{x}})= g(\vec{\mathbf{w}} \cdot \vec{\mathbf{x}} + b) = \frac{1}{1 + e^{-(\vec{\mathbf{w}} \cdot \vec{\mathbf{x}} + b)}}$$

The question that we want to answer is: **given this training set, how can we calculate parameters $\mathbf{w}$ and $\mathbf{b}$?**

Recall, from linear regression **the squared error cost function**:

$$J(\vec{\mathbf{w}},\mathbf{b}) = \frac{1}{2m} \sum\limits_{i = 1}^{m} (f_{\mathbf{w},b}(\vec{\mathbf{x}}^{(i)}) - y^{(i)})^2$$ 

Recall that in **linear regression**:

- we have the model $ f_{\vec{\mathbf{w}}, b} (\vec{\mathbf{x}})= \vec{\mathbf{w}} \cdot \vec{\mathbf{x}} + b$
- the cost function takes a **bowl shape**, which is **convex**, and allows our **gradient descent algorithm** to find the global minium.

However, in **logistic regression**:
- we have the model $ f_{\vec{\mathbf{w}}, b} (\vec{\mathbf{x}})= \frac{1}{1 + e^{-(\vec{\mathbf{w}} \cdot \vec{\mathbf{x}} + b)}}$
- if we plot the cost function using this values of $f(x)$, we get a **non-convex** line, and if we tried to use gradient descent, there are lots of local minima that we can get stuck in and never find the global minimum.

![](2024-01-04-10-12-14.png)

So for **logistic regression**, the squared error cost function is not a good choice.

In order to build a new cost function for logistic regression, we need to change the definition of the cost function that we have above: we'll call the term inside the sumation the **loss of a training example**:

$$ \text{Loss} \Rightarrow L(f_{\mathbf{\vec{w}},b}(\vec{\mathbf{x}}^{(i)}), y^{(i)})  $$

So we are denoting the loss $L$ as a function of:
- the prediction of the learning algorithm $f_{\mathbf{\vec{w}},b}(\vec{\mathbf{x}}^{(i)})$
- the true label $y^i$

The loss, given those two variables, is one half of the squared difference:

$$ L(f_{\mathbf{\vec{w}},b}(\vec{\mathbf{x}}^{(i)}), y^{(i)}) = \frac{1}{2} (f_{\mathbf{\vec{w}},b}(\vec{\mathbf{x}}^{(i)}) - y^{(i)}) ^ 2$$

By choosing a different form for this loss function, we can keep the overall cost function to be a convex curve.

### Logistic regression cost function

The loss function inputs $f$ and the true label $y$, and tells us how well we are doing in that example.

Let's first write the definition of the loss function that we will use for this regresssion:

$$   
L(f_{\mathbf{\vec{w}},b}(\vec{\mathbf{x}}^{(i)}), y^{(i)}) = 
     \begin{cases}
       -\log{(f_{\mathbf{\vec{w}},b}(\vec{\mathbf{x}}^{(i)}))} &\quad\text{if } y^{(i)} = 1\\
       -\log{(1 - f_{\mathbf{\vec{w}},b}(\vec{\mathbf{x}}^{(i)}))} &\quad\text{if } y^{(i)} = 0\\
     \end{cases}
$$

Let's take a look at why this loss function makes sense.

Let's consider first the case of $y = 1$ and plot what this function looks like, to get some intuition into what this function is doing. 

Remember that the loss function measures **how well you are doing on one example**, while it si by summing up the losses in all your examples that you can get the cost function, which measures **how well you are doing in the entire training set.**

So if we plot $\log(f)$ and $-\log(f)$, we get:

![](2024-01-04-23-29-44.png)

Notice that both curves intersect the $x$ axis on 1, and this is important for us in $-\log(f)$, since after 1, all values of the curve are negative.

Now, $f$ is the output of a logistic regression. This it is always between `0` and `1`. Therefor the only part of the curve that is relevant is what's between `0` and `1`, marked in the following graph:

![](2024-01-04-23-33-06.png)

Now, let's zoom in and take a closer look at that part of the graph:

![](2024-01-04-23-34-25.png)

#### Graph when $y^{(i)} = 1$
If the algorithm predicts a probability close to `1`, and the true label $y^{(i)} = 1$, then the loss is very small, approaches `0`, **because we are very close to the right answer.**.

Now, continuing with the example of the true label being `1`, $y^{(i)} = 1$, - that is, the example really being a malignant tumor -, then if the algorithm predicts `0.5`, we would get a higher value. And if it predicted `0.1`, it would be much higher again.

This is because our algorithm would be predicting that there is only a 10% chance of the tumor being malignant, when the true value actually says that tumor is indeed malignat. That is why the loss is so high:

![](2024-01-04-23-39-54.png)

So, when $y^{(i)} = 1$, the loss function incentivizes, or nudges, pushed the algorithm to make more accurate predictions, because the **loss is lowest when $f_{\mathbf{\vec{w}},b}(\vec{\mathbf{x}}^{(i)})$ predicts values close to the true label $y^{(i)}$ close to `1`.**

#### Graph when $y^{(i)} = 0$
