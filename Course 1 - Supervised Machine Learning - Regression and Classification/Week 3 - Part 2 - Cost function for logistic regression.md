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