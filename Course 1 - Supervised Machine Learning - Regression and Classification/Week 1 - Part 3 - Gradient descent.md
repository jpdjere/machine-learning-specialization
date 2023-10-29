# Gradient descent

## Visualization of gradient descent

We can use the algorithm called **gradient descent** to find the values of $w$ and $b$ that result in the smallest possible cost in a systematic way, instead of trial and error.

Gradient descent is used all over the place in ML, not just for linear regression. For example, in deep learning.

Here's an overview of what we'll do with gradient descent.

![](./img/2023-10-29-21-48-11.png)

We have the cost function $J(w,b)$, that we want to minimize by finding the proper value for those parameteres $w$ and $b$. In the example we've seen so far, this is a cost function for a linerat regression, but gradient descent wcan be used to minimize any function. So if we have a cost function with $w_1$, $w_2$, etc, and $b$, we want to minimize $J$ over those parameters, i.e: pick values for them that minimize $J$.

![](./img/2023-10-29-21-52-49.png)

What we are going to do, just to start off, is to **do some initial guesses for $w$ and $b$**. In linear regression, it won't matter what the initial values are so it is common to initialize both of them to 0.

![](./img/2023-10-29-21-54-45.png)

With the gradient descent algorithm, what we are going to do is to **keep changing $w$ and $b$ a bit to try to reduce the cost $J$**, until it hopefully settles at or near a minimum.

![](./img/2023-10-29-21-58-02.png)

One important note: for some functions $J$ might not have a soupbowl or hammock shape: **it's possible for there to be more than one minimum**.

Let's look at an example, and notice that **this cost function is not for a linear regression, i.e. the cost function is not a squared error function.**

![](./img/2023-10-29-21-59-29.png)

What you see above is the cost function that you might get if you are training a neural network model.

If we imagine we are standing on a point of that surface, on a hill.

![](./img/2023-10-29-22-01-06.png)

Your goal is to start there and get to the bottom of one of the valleys as efficiently as possible.

**What the gradient descent algorithm does is:** take a 360 degree look around and ask myself, if I were to take a step in one direction, and I want to go downhill as quickly as possible, what direction do I choose to take that baby step?

In our example, starting where the man is drawn, the best direction is roughly:

![](./img/2023-10-29-22-04-05.png)

That is the **direction of steepest descent**. That means that a step in that direction makes you go faster and more efficiently to a valley that a step in any other direction.

After taking that first step, we repeat the process. And we keep going and repeating the process until we find ourselves at the bottom of the valley, at a local minimum.

![](./img/2023-10-29-22-06-49.png)

We just went through **multiple steps of gradient descent**.

However, it is important to know that **gradient descent has an intersting property:** since we chose $w$ and $b$ arbitrarily, if we had chosen other starting values and repeated the process of gradient descent, we would have ended on a totally different valley.

![](./img/2023-10-29-22-13-08.png)

The bottom of **both the first and second valleys are called local minima**, because if you start going dow the first valley, gradient descent won't lead you to the second valley, and the same is true if you started going down the second valley, but vicecersa.

## Implementing gradient descent

Let's write down the gradient descent algorithm:

$w = w - α \frac{∂}{∂w} J(w, b)$

$b = b - α \frac{∂}{∂b} J(w, b)$

What this expression is saying is: update your value of $w$ by taking its current value and adjusting it a small amount (the expression on the right).

First: notice that we are using assignment and not mathematical equality for the equal notation.

Second, $α$ is the **learning rate**. It is a number **between 0 and 1** and it **controls how "big of a step" you take downhill**. If $α$ is large, that corresponds to a very aggressive gradient descent procedure, where you take huge steps downhill, and viceversa.

Finally, the term:

$\frac{∂}{∂w} J(w, b)$

is the **derivative term of the cost function $J$**. For now, you can think of it as the **direction in which you want to take your step downhill**.

One important thing: remember that our model has two parameters, $w$ and $b$.

So we are going to repeat the gradient descent step for the two equations above until the algorithm converges, reaching the local minimum where those two parameters no longer change (much).

**IMPORTANT:** we always want to do **simultaneous update of $w$ and $b$**. That means calculating the right side of the equation simultaneously, before assigning the results to the updated value.

## Gradient descent intuiton

Let's get a better intuition of what gradient descent is doing.

Let's use a slightly simpler example where we work on minimizing just one parameter, $w$, and thus the cost function is $J(w)$. That means that the gradient descent now looks like this:

$w = w - α \frac{∂}{∂w} J(w)$

And we are only trying to minimize the cost by minimizing the parameter $w$.

So this is like our previous example where we had temporarily set $b = 0$, and we can look at a two-dimensional graph of the cost function $J$.

![](./img/2023-10-29-22-48-40.png)

So let's first inititialize gradient descent with a starting value for $w$ at random.

And then let's use the gradient descent formula to update $w$.

![](./img/2023-10-29-22-50-00.png)

But we first need to understand what the derivative term $\frac{∂}{∂w} J(w)$ means.

We can think of the derivative on that point in the line that we have picked randomly by drawing the tangent line to the curve at that point. And **the slope of that line is the derivative of the function $J$ at that point.**

You can get the slop by dividing the height vs width of triangle. When the slope is positive (the tangent line pointing upwards toward the right of the graph), then **the derivative is a positive number.**

So in this case, the updated $w$ will be smaller than the original $w$, since **the learning rate is always a positive number**, and in this case the derivative is positive.

![](./img/2023-10-29-22-56-13.png)

And you can see that in the graph, $w$ decreases, i.e. moves to the left, and approaches the minimum of the cost $J(w)$ function.

![](./img/![](./img/2023-10-29-22-57-43.png).png)

And the same will be true, but inversed if the starting point of $w$ is at a place of the curve where the slope is negative: the derivative will be negative, and since we are multiplying by $-α$, the updated value for $w$ will be greater than the original.

That means we are moving toward the right of the curve, and thus toward the minimum of our cost function.

![](./img/2023-10-29-22-59-45.png)

## Learning rate

The choice of the learning rate $α$ will have a huge impact on the efficiency of your implementation of gradient descent. If $α$ is chosen poorly rate of descent may not even work at all. 

Let's take a deeper look at the learning rate to help us choose better learning rates for our implementations of gradient descent.

Let's first see what could happen if the learning rate $α$ is **too small**.

If we start our gradient descent algorithm at a random place in our $J(w)$ curve, like so:

![](./img/2023-10-29-23-17-42.png)

And we select a very small learning rate $α$ like $0.0000001$, we will give miniscule steps as update $w$ and approach the minimum.

The outcome of this process is that we do end up decreasing the cost $J(w)# (i.e. approaching its minimum), but **incredibly slowly.**

![](./img/2023-10-29-23-19-57.png)

Now, if the learning rate $α$ is **too large**:

We start with a point over the curve $J(w)$ that is already close to the minimum, and its derivative is negative, so we should update $w$ to the right.

![](./img/2023-10-29-23-21-48.png)

However, since the learning rate $α$ is too large, we will take a step that crosses over to the other side of the minimum (overshoots), and actually gets away from it, so the cost $J(w)$ has actually **increased**.

![](./img/2023-10-29-23-24-43.png)

Now, at this point, the derivate tells us to **decrease** $w$, but since the learning rate is too large, we might take a huge step, going all the way to the other side of the minimum, further increasing $J(w)$.

![](./img/2023-10-29-23-33-07.png)

And we can continue this process overshooting continually, getting further and further away from the minimum. So, if the learning rate is too large, **gradient descent may overshoot and never reach a minimum.** It will **fail to converge, or diverge.**

![](./img/2023-10-29-23-34-49.png)

Here's another question, that you might be wondering: what happens when the param $w$ is already located at a minimum and we want to calculate gradient descent?

![](./img/2023-10-29-23-36-15.png)

If we are located at the point shown in the graph, with, for e.g. a value of $5$, and we draw its **tangent** there, it will be completely horizontal, which means it has a **slope of value $0$**.

Thus, the derivative term is $0$ for the current value of $w$:

![](./img/2023-10-29-23-38-06.png)

So basically, $w$ will not update, it will stay at $5$. That means that if we have reached a local minimum, further steps of gradient descent will not update the value of the parameters.

This also explains **why we can reach a local minimum if we have a _fixed_ learning rate**. Let's visualize a cost function and an initial point on the curve up to the right:

![](./img/2023-10-29-23-40-14.png)

With the first step, the slope is pretty vertical, so the derivative is large. Thus the step will also be large.

![](./img/2023-10-29-23-41-03.png)

But on the second step, the slope will be a little less vertical, and thus the step to take will be smaller.

![](./img/2023-10-29-23-41-45.png)

And thus for each subsequent step, as we approach the minimum, the derivative gets smaller and smaller, closer to 0, until we reach the minimum.

![](./img/2023-10-29-23-42-45.png)

## Gradient descent for linear regression

Up until now, we've seen the linear regression model and then its cost function $J(w)$, and then the gradient descent algorithm. 

Now, we're going to put everything together and use the squared error cost function for the linear regression model with gradient descent. This will allow us to train the linear regression model to fit a straight line through the training data.

Let's see now what the actual gradient descent formulas for $w$ and $b$ look like when we calculate the partial derivatives of the cost function:

![](./img/2023-10-29-23-51-44.png)

So the formulas result in:

$w = w - α \frac{∂}{∂w} J(w,b) $

$w = w - α ({1\over {m}} {\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}}))$

and:

$b = b - α \frac{∂}{∂w} J(w,b)$

$b = b - α ({1\over {m}} {\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)}}))$

The derivates come from:

![](./img/2023-10-30-00-02-22.png)

Remember that on a cost function of some unknown function such as one produced by a depp learning network, your cost function can have multiple local minimum.

However, for a **linear regression**, using the **squared error cost function**, the graph will always have a soupbowl shape. It is a **convex function**, which has a unique **global minimum**:

![](./img/2023-10-30-00-04-54.png)

And if you run gradient descent for such a cost function, it **will always converge to that global minimum**, provided you don't provided a learning rate that causes the algorithm to overshoot.

## Running gradient descent

Let's see what happens when you run gradient descent for linear regression with the algorithm in action.

First we have a plot of our data, and our contour plot and 3D plot of $J(w,b)$.

Let's initialize our data to $b = 900$ and $w = - 0.1$, and see the model line (straight line fit) that we get (pretty far off!):

![](./img/2023-10-30-00-12-21.png)

Now, let's take one step with the **gradient descent algorith** and see how our fitted line is updated, and how the point in the contour plot moves:

![](./img/2023-10-30-00-13-14.png)

And let's continue now, until we converge:

![](./img/2023-10-30-00-13-44.png)

That's **gradient descent**. And now you can use the $f_{(w,b)}(x)$ model to predict the price of a house based on its price.

To be more precise: this process of gradient descent is called **batch gradient descent**. This means that **for every step of the algorithm we use all the training examples to recalculate our parameters**.

![](./img/2023-10-30-00-16-24.png)

There are **other versions** of gradient descent that do not use the whole training set, but small subsets of the data at each step.

