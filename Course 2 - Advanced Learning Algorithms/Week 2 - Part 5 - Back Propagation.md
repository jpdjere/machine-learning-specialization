# Back Propagations

## Computation graph

The **computation graph** is a key idea in deep learning, and it is also how programming frameworks like TensorFlow, automatically **compute derivatives of our neural networks**. 

Let's take a look at how it works with a small neural network exampleof just one layer, which is also the output layer, and just one unit in the output layer. It takes us inputs $x$, applies a linear activation function and outputs the activation $a$. It outputs $wx + b$. 

This is basically linear regression, but expressed as a neural network with one output unit. Given the output, the cost function $J$ is:

$$J(w,b) = \frac{1}{2} (a - y)^2$$

![](2024-02-13-00-16-16.png)

For this small example, we're only going to have a single training example, where the training example is the input $x$ equals $-2$. The ground truth output value $y$ equals $2$, and the parameters of this network are; $w$ equals 2 and $b$ equals 8. 

![](2024-02-13-00-17-40.png)

Let's show how the computation of the cost function $J$ can be computed step by step using a computation graph. 


Let's take the computation of $J$ and break it down into individual steps. 

 - First, we have the parameter $w$ that is an input to the cost function $J$, and then we first need to compute $w$ times $x$. Let's call that $c$; $w$ is equal to 2, $x$ is equal to negative 2, and so $c$ would be $-4$.

 ![](2024-02-13-00-22-48.png)
 
- The next step is then to compute $a$, which is $wx + b$. In another node here, this needs to input $b$, the other parameter that is input to the cost function $J$, and $a = wx + b$ is equal to $c + b$. If we add these up, that turns out to be 4.

![](2024-02-13-00-24-18.png)

This is starting to build up a computation graph in which the steps we need to compute the cost function $J$ are broken down into smaller steps. 

- The next step is to then compute $a - y$, which we're going to call $d$. $y$ is equal to 2, so $4 - 2 = 2$. 

![](2024-02-13-00-25-39.png)

- Then finally, $J$ is is 1/2 of $a$ - $y$ squared, or 1/2 of $d$ squared, which is just equal to 2. 

![](2024-02-13-00-26-30.png)

What we've just done is build up a computation graph. It shows the forward prop step of how we compute the output a of the neural network. But then also go further than that so also compute the value of the cost function $J$.

![](2024-02-13-00-27-10.png)

But the question now is, **how do we find the derivative of $J$ with respect to the parameters $w$ and $b$?** 

Let's take a look at that next. Here's the computation graph from the previous slide and we've completed for a prop where we've computed that $J$, the cost function, is equal to 2 through all these steps going from left to right in the computation graph.

![](2024-02-13-00-28-27.png)

What we'd like to do now is compute the derivative of $J$ with respect to $w$ and the derivative of $J$ with respect to $b$. 

It turns out that whereas for a prop was a left to right calculation, **computing the derivatives will be a right to left calculation, which is why it's called backprop**, was going backwards from right to left. 

So, the final computation nodes of this graph is the one to the outmost right, which computes $J = 1/2 d^2$. 

The first step of backprop will ask:

- if the value of $d$, which was the input to this last node, were to change a little bit, how much does the value of $J$ change? Specifically, if $d$ were to go up by a little bit, say 0.001, and that'll be our value of $\epsilon$ in this case, how would the value of $J$ change? 

![](2024-02-13-00-31-33.png)

In this case if $d$ goes from 2 to 2.001, then $J$ goes from 2 to 2.002. So if $d$ goes up by $\epsilon$, $J$ goes up by roughly two times $\epsilon$. **We conclude that the derivative of $J$ with respect to this value $d$ that is inputted this final node is equal to two.**

![](2024-02-13-00-33-02.png)

The first step of backprop is be to fill in this value $2$, the derivative of $J$ with respect to this input value $d$, in the node that connects $d$ as an input to $J$. We know if $d$ changes a little bit, $J$ changes by twice as much because this derivative is equal to two.

![](2024-02-13-00-34-45.png)

- The next step is to look at the node before that and ask what is the derivative of $J$ with respect to $a$? 

To answer that, we have to ask: if $a$ goes up by 0.001, how does that change $J$? We know that if $a$ goes up by 0.001, $d$ is just $a - y$. If $a$ becomes 4.001, $d$ which is $a - y$, becomes 4.001 - 2, so it becomes 2.001. 

So, $a$ goes up by 0.001, $d$ also goes up by 0.001. 

But we'd already concluded previously that if $d$ goes up by 0.001, $J$ goes up by twice as much. Now we know if $a$ goes up by 0.001, $d$ goes up by 0.001, then $J$ goes up roughly by two times 0.001. This tells us that the derivative of $J$ with respect to $a$ is also equal to two. 

![](2024-02-13-00-40-32.png)

This is the derivative of $J$ with respect to $a$, just as this was the derivative of $J$ respect to $d$. This step of computation that we just did is actually relying on the chain rule for calculus.The derivative of $J$ with respect to $a$ is asking: how much does $d$ change respect to $a$, which is derivative of $d$ respect to $a$, multiplied by the derivative of $J$ with respect to $d$:

![](2024-02-13-00-43-42.png)

The calculation on top of the image showed that the partial of $d$ with respect to $a$ is 1, and we showed previously that the derivative of $J$ with respect to $d$ is equal to 2, which is why the derivative of $J$ with respect to $a$ is $1$ times $2$, which is equal to $2$.


![](2024-02-13-00-46-15.png)

We keep on going right to left as we do in backprop. 

- The next step then is: how much does a little change in $c$ cause $J$ to change, and how much does a change in $b$ cause $J$ to change? 

![](2024-02-13-00-48-00.png)

The way we figure that out is to ask, what if $c$ goes up by $\epsilon$ = 0.001, how much does $a$ change? Sincee $a = c + b$, it turns out that if $c$ ends up being negative 3.999, then $a$, which is -3.999, plus 8, becomes 4.001. 

So, if $c$ goes up by $\epsilon$, $a$ goes up by $\epsilon$. We know if $a$ goes up by $\epsilon$, then because the derivative of $J$ with respect to$ $a$ is two, we know that this in turn causes $J$ to go up by two times $\epsilon$.

![](2024-02-13-00-50-12.png)

We can conclude that if $c$ goes up by a little bit, $J$ goes up by twice as much. We know this because we know the derivative of $J$ with respect to $a$ is 2. This allows us to conclude that the derivative of $J$ with respect to $c$ is also equal to 2:

![](2024-02-13-00-50-46.png)

With chain rule, another way to write this, is: the derivative of $J$ respect to $c$, is the derivative of $a$ with respect to $c$ (which is 1) times the derivative of $J$ respect to $a$ (which we have previously figured out was equal to 2), so it ends up being equal to 2. 

![](2024-02-13-00-52-40.png)

By a similar calculation, if $b$ goes up by 0.001, then $a$ also goes up by 0.001 and $J$ goes up by 2 times 0.001, which is why this derivative is also equal to 2. 

![](2024-02-13-00-53-38.png)

- Now one final step: what is the derivative of $J$ with respect to $w$? 

If $w$ goes up by 0.001, what happens? $c$ is $w$ times $x$. If $w$ were 2.001, $c$ which is $w$ times $x$, becomes -2 times 2.001, so $c$ becomes negative 4.002.

![](2024-02-13-00-56-56.png)

If $w$ goes up by $\epsilon$, $c$ goes down by 2 times 0.001, or equivalently $c$ goes up by negative 2 times 0.001.

![](2024-02-13-00-57-27.png)

We know that if $c$ goes up by  -2 times 0.001, since the derivative of $J$ with respect to $c$ is 2, this means that $J$ will go up by  -4 times 0.001: because if $c$ goes up by a certain amount, $J$ changes by 2 times as much. **This allows us to conclude that if $w$ goes up by 0.001, $J$ goes up by  -4 times 0.001.**

**So, the derivative of $J$ with respect to $w$ is $-4$**.

![](2024-02-13-00-59-51.png)

Once again, the chain rule calculation is:

![](2024-02-13-01-00-17.png)

To wrap up what we've just done this manually carry out backprop in this computation graph. 

In fact, let's double-check the computation that we just did. $J$ with these values of w, b, $x$ and $y$ is equal to one-half times wx plus $b$ minus $y$ squared, which is one-half times 2 times negative 2 plus 8 minus 2 squared, which is equal to 2:

![](2024-02-13-01-01-57.png)

Now if $w$ were to go up by 0.001, then $J$ becomes 1/2 times (() 2.001 times -2, plus $b$ which is 8) minus $y$ squared. If we calculate this out, this turns out to be 1.996002.

![](2024-02-13-01-03-23.png)

**Roughly $J$ has gone from 2 down to 1.996, and $J$ has therefore gone down by 4 times $\epsilon$. This shows that if $w$ goes up by $\epsilon$, $J$ goes down by four times $\epsilon$ , or equivalently, $J$ goes up by -4 times $\epsilon$. This is why the derivative of $J$ with respect to $w$ is -4.**

--- 
**Why do we use the backprop algorithm to compute derivatives?** 

Backdrop is an efficient way to compute derivatives. The reason we sequence this as a right-to-left calculation is: if we were to start off and ask what is the derivative of $J$ with respect to $w$?

![](2024-02-13-01-06-23.png)

Then, to know how much change in $w$ affects change in $J$, if $w$ were to go up by $\epsilon$, how much does $J$ go by $\epsilon$? Well, the first thing we want to know is, what is the derivative of $J$ with respect to $c$? Because change in $w$ will change $c$, the first quantity after the first node:

![](2024-02-13-01-07-17.png)

So, to know how much change in $w$ affects $J$, we want to know how much does change in $c$ affects $J$. But to know how much change in $c$ affects $J$, the most useful thing to know would be how does a change in $c$ change $a$:

![](2024-02-13-01-08-25.png)

And, we want to know how much this change in $a$ effect $J$, and so on. 

That's why backprop a sequence as a right-to-left calculation. Because if we do the calculation from right to left, we can find out how does change in $d$ affect change in $J$. Then we can find out how much this change in a effect $J$, and so on, until we find the derivatives of each of these intermediate quantities, $c$, $a$, and $d$, as well as the parameters $w$ and b.

![](2024-02-13-01-09-52.png)

One thing that makes backprop efficient is that, when we do the right-to-left calculation, we had to compute some terms, for example, the derivative of $J$ with respect to $a$ just once, although this quantity is then used to compute both the derivative of $J$ with respect to $w$ and the derivative of $J$ with respect to $b$. 


![](2024-02-13-01-11-24.png)

It turns out that if a computation graph has $n$ nodes and $p$ parameters (two parameters in this case $w$ and $b$), this procedure allows us to compute all the derivatives of $J$ with respect to all the parameters in roughly: n plus p steps, rather than n times p steps.

![](2024-02-13-01-12-38.png)

If we have a neural network with 10,000 nodes and maybe 100,000 parameters, being able to compute the derivatives in 10,000 plus 100,000 steps is much better than calcualting 10,000 times 100,000 steps, which would be a billion steps. 

The backpropagation algorithm done using the computation graph gives we a very efficient way to compute all the derivatives. That's why it is such a key idea in how deep learning algorithms are implemented today. 



## Large neural network example

In this final section on intuition for backprop, let's take a look at how the computation draft works on a larger neural network example. Here's the network we will use with a single hidden layer, with a single hidden unit that outputs $a^{[1]}$, that feeds into the output layer that outputs the final prediction $a^{[2]}$. 

![](2024-02-13-01-15-14.png)

To make the math more tractable, we're going to continue to use just a single training example with inputs $x$ = 1, $y$ = 5. The parameters are in the screenshot below. And throughout we're going to use the ReLU activation functions of $g(z) = max(0, z)$. 

![](2024-02-13-01-16-37.png)

So for prop in our network looks like this: as usual, $a^{[1]}$ equals $g(w_1 x + b_1)$. And so it turns out $w_1 x + b$ will be positive, so we're in the $max(0, z) = z$ part of this activation function. 

So that's just equal to: 2 times 1 + 0, which is equal to 2. 

![](2024-02-13-01-19-13.png)

And then similarly, $a^{[2]}$ equals: 

![](2024-02-13-01-19-57.png)

Finally, we'll use the squared error cost function:

So: 

$$J(w, b) = \frac{1}{2}(a^{[2]}- y)^2 = 1/2 (7-5)^2 = 2$$

So, let's take this calculation that we just did and write it down in the form of a computation graph. 

To carry out the computation step by step, first thing we need to do is take $w_1$ and multiply that by $x$. So we have $w_1$ that feeds into computation node that computes $w_1$ times $x$. And we're going to call this a temporary variable $t^{[1]}$:

![](2024-02-13-01-23-32.png)

Next, we compute $z^{[1]}$, which is $t^{[1]}$ + $b_1$. So we also have the input $b_1$:

![](2024-02-13-01-24-17.png)

And finally $a^{[1]}$ equals $g(z^{[1]})$. We apply the activation function, and so we end up with again the value 2:

![](2024-02-13-01-25-18.png)

And then next we have to compute $t^{[2]}$, which is $w^{[2]}$ times $a^{[1]}$:

![](2024-02-13-01-25-55.png)

And so, with $w_2$, that gives us this value which is 6:

![](2024-02-13-01-26-14.png)

Then $z^{[2]}$, which is $w^{[2]} a^{[1]} + b^{[2]}, gives us 7. 

![](2024-02-13-01-28-27.png)

And finally apply the activation function, g, we still end up with 7:

![](2024-02-13-01-29-19.png)

And lastly, $J = \frac{1}{2}(a^{[2]}-y)^2$. And that gives us 2, which was this cost function here. 

So this is how we take the step by step computations for larger neural network and write it in the computation graph. 

If we were to carry out backprop, the first thing we do is ask, what is the derivative of the cost function $J$ respect to a2? And it turns out if we calculate that, it turns out to be 2. 

So we'll fill that in here. And the next step will be asked, what's the derivative of the cost $J$ respect to z2. And using this derivative that we computed previously, we can figure out that this turns out to be 2. 

Because if z goes up by $\epsilon$, we can show that for the current setting of all the parameters a2 will go up by $\epsilon$. And therefore, $J$ will go up by 2 times $\epsilon$. So this derivative is equal to 2, and so on. 

Step by step. We can then find out that the derivative of $J$ respective $b_2$ is also equal to 2. The derivative respect to t2 is equal to 2, and so on, and so forth. 

Until eventually we've computed the derivative of $J$ with respect to all the parameters $w_1$, $b_1$, $w_2$, and $b_2$. And so that's backprop.

![](2024-02-13-01-31-38.png)

Let me just double check one of these examples. So we saw here that the derivative of $J$ respect $w_1$ is equal to 6. So what this is predicting is that, if $w^1$ goes up by epsilon, $J$ should go up by roughly 6 times epsilon. Let's step through the map and see if that really is true.

![](2024-02-13-01-32-25.png)

And so if $w$ which was 2 were to be 2.001 goes up by epsilon, then a1 becomes, let's see, instead of 2, this is 2.001 as well. So a1 instead of 2 is now 2.001. So 3 $x$ 2.001 + 1, this gives us 7.003.

![](2024-02-13-01-33-16.png)

And if a2 is 7.003, then just becomes 7.003-5 squared. And so this becomes 2.003 squared over 2, which turns out to be equal to 2.006005.

√

So ignoring some of the extra digits, we see from this calculation that, if $w^1$ goes up by 0.001, $J$ of $w$ has gone up from 2 to 2.006 roughly. 

![](2024-02-13-01-34-02.png)


So 6 times as much. And so the derivative of $J$ with respect to $w_1$ is indeed equal to 6. 

![](2024-02-13-01-34-18.png)

---

So, the backprop procedure gives we a very efficient way to compute all of these derivatives, which we can then feed into the gradient descent algorithm or the Adam optimization algorithm, to then train the parameters of our neural network. 

And again, the reason we use background for this is, is a very efficient way to compute all the derivatives of $J$ respect to $w_1$, $J$ respect to $b_1$, $J$ respect to $w_2$, and $J$ respect to $b_2$.

![](2024-02-13-01-35-38.png)

I did just illustrate how we could bump up $w_1$ by a little bit and see how much $J$ changes. But that was a left to right calculation. And then we had to do this procedure for each parameter, one parameter at a time. We had to increase $w$ by 0.001 to see how that changes $J$, increase $b_1$ by a little bit to see how that changes $J$, and increase every parameter, one at a time by a little bit to see how that changes $J$. Then this becomes a very inefficient calculation. And if we had N nodes in our computation graph and P parameters, this procedure would end up taking N times P steps, which is very inefficient. 

Whereas we got all four of these derivatives N + P, rather than N times P steps. And this makes a huge difference in practical neural networks, where the number of nodes and the number of parameters can be really large.

## Optional Lab: Derivatives

This lab will give you a more intuitive understanding of derivatives. It will show you a simple way of calculating derivatives arithmetically. It will also introduce you to a handy Python library that allows you to calculate derivatives symbolically.

```py
from sympy import symbols, diff
```

### Informal definition of derivatives

The formal definition of derivatives can be a bit daunting with limits and values 'going to zero'. The idea is really much simpler. 

The derivative of a function describes how the output of a function changes when there is a small change in an input variable.

Let's use the cost function $J(w)$ as an example. The cost $J$ is the output and $w$ is the input variable.  
Let's give a 'small change' a name *epsilon* or $\epsilon$. We use these Greek letters because it is traditional in mathematics to use *epsilon*($\epsilon$) or *delta* ($\Delta$) to represent a small value. You can think of it as representing 0.001 or some other small value.  

$$
\begin{equation}
\text{if } w \uparrow \epsilon \text{ causes }J(w) \uparrow \text{by }k \times \epsilon \text{ then}  \\
\frac{\partial J(w)}{\partial w} = k \tag{1}
\end{equation}
$$

This just says if you change the input to the function $J(w)$ by a little bit and the output changes by $k$ times that little bit, then the derivative of $J(w)$ is equal to $k$.

Let's try this out.  Let's look at the derivative of the function $J(w) = w^2$ at the point $w=3$ and $\epsilon = 0.001$

```py
J = (3)**2
J_epsilon = (3 + 0.001)**2
k = (J_epsilon - J)/0.001    # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")
# J = 9, J_epsilon = 9.006001, dJ_dw ~= k = 6.001000 
```

We have increased the input value a little bit (0.001), causing the output to change from 9 to 9.006001, an increase of 6 times the input increase. Referencing (1) above, this says that $k=6$, so $\frac{\partial J(w)}{\partial w} \approx 6$. If you are familiar with calculus, you know, written symbolically,  $\frac{\partial J(w)}{\partial w} = 2 w$. With $w=3$ this is 6. Our calculation above is not exactly 6 because to be exactly correct $\epsilon$ would need to be [infinitesimally small](https://www.dictionary.com/browse/infinitesimally) or really, really small. That is why we use the symbols $\approx$ or ~= rather than =. Let's see what happens if we make $\epsilon$ smaller.

```py
J = (3)**2
J_epsilon = (3 + 0.000000001)**2
k = (J_epsilon - J)/0.000000001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
# J = 9, J_epsilon = 9.000000006, dJ_dw ~= k = 6.000000496442226 
```

The value gets close to exactly 6 as we reduce the size of $\epsilon$. Feel free to try reducing the value further.

### Finding symbolic derivatives
In backprop it is useful to know the derivative of simple functions at any input value. Put another way, we would like to know the 'symbolic' derivative rather than the 'arithmetic' derivative. An example of a symbolic derivative is,  $\frac{\partial J(w)}{\partial w} = 2 w$, the derivative of $J(w) = w^2$ above.  With the symbolic derivative you can find the value of the derivative at any input value $w$.  

If you have taken a calculus course, you are familiar with the many [differentiation rules](https://en.wikipedia.org/wiki/Differentiation_rules#Power_laws,_polynomials,_quotients,_and_reciprocals) that mathematicians have developed to solve for a derivative given an expression. Well, it turns out this process has been automated with symbolic differentiation programs. An example of this in python is the [SymPy](https://www.sympy.org/en/index.html) library. Let's take a look at how to use this.

#### $J = w^2$
Define the python variables and their symbolic names.

```py
J, w = symbols('J, w')
```

Define and print the expression. Note SymPy produces a [latex](https://en.wikibooks.org/wiki/LaTeX/Mathematics) string which generates a nicely readable equation.

```py
J=w**2
J
```

Use SymPy's `diff` to differentiate the expression for $J$ with respect to $w$. Note the result matches our earlier example.

```py
dJ_dw = diff(J,w)
dJ_dw
# 2𝑤
```

Evaluate the derivative at a few points by 'substituting' numeric values for the symbolic values. In the first example, $w$ is replaced by $2$.

```py
dJ_dw.subs([(w,2)])    # derivative at the point w = 2
# 4

dJ_dw.subs([(w,3)])    # derivative at the point w = 3
# 6

dJ_dw.subs([(w,-3)])    # derivative at the point w = -3
# -6
```

![](2024-02-13-01-46-59.png)

![](2024-02-13-01-47-07.png)

![](2024-02-13-01-47-18.png)

![](2024-02-13-01-47-29.png)

## Optional Lab: Back propagation using a computation graph

Working through this lab will give you insight into a key algorithm used by most machine learning frameworks. Gradient descent requires the derivative of the cost with respect to each parameter in the network.  Neural networks can have millions or even billions of parameters. The *back propagation* algorithm is used to compute those derivatives. *Computation graphs* are used to simplify the operation. Let's dig into this below.

```py
from sympy import *
import numpy as np
import re
%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets
from lab_utils_backprop import *
```

### Computation Graph
A computation graph simplifies the computation of complex derivatives by breaking them into smaller steps. Let's see how this works.

Let's calculate the derivative of this slightly complex expression, $J = (2+3w)^2$. We would like to find the derivative of $J$ with respect to $w$ or $\frac{\partial J}{\partial w}$.

```py
plt.close("all")
plt_network(config_nw0, "./images/C2_W2_BP_network0.PNG")
```

![](2024-02-13-01-54-08.png)

Above, you can see we broke the expression into two nodes which we can work on independently. If you already have a good understanding of the process from the lecture, you can go ahead and fill in the boxes in the diagram above. You will want to first fill in the blue boxes going left to right and then fill in the green boxes starting on the right and moving to the left.
If you have the correct values, the values will show as green or blue. If the value is incorrect, it will be red. Note, the interactive graphic is not particularly robust. If you run into trouble with the interface, run the cell above again to restart.

If you are unsure of the process, we will work this example step by step below.

### Forward Propagation   
Let's calculate the values in the forward direction.

>Just a note about this section. It uses global variables and reuses them as the calculation progresses. If you run cells out of order, you may get funny results. If you do, go back to this point and run them in order.

```py
w = 3
a = 2+3*w
J = a**2
print(f"a = {a}, J = {J}")
# a = 11, J = 121
```

### Backprop
 Backprop is the algorithm we use to calculate derivatives. As described in the lectures, backprop starts at the right and moves to the left. The first node to consider is $J = a^2 $ and the first step is to find $\frac{\partial J}{\partial a}$ 

 ### $\frac{\partial J}{\partial a}$ 
#### Arithmetically
Find $\frac{\partial J}{\partial a}$ by finding how $J$ changes as a result of a little change in $a$. This is described in detail in the derivatives optional lab.

```py
a_epsilon = a + 0.001       # a epsilon
J_epsilon = a_epsilon**2    # J_epsilon
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")
# J = 121, J_epsilon = 121.02200099999999, dJ_da ~= k = 22.000999999988835 
```

$\frac{\partial J}{\partial a}$ is 22 which is $2\times a$. Our result is not exactly $2 \times a$ because our epsilon value is not infinitesimally small. 

#### Symbolically
Now, let's use SymPy to calculate derivatives symbolically as we did in the derivatives optional lab. We will prefix the name of the variable with an 's' to indicate this is a *symbolic* variable.

```py
sw,sJ,sa = symbols('w,J,a')
sJ = sa**2
sJ

sJ.subs([(sa,a)])

dJ_da = diff(sJ, sa)
```
![](2024-02-13-01-55-34.png)
![](2024-02-13-01-55-44.png)

So, $\frac{\partial J}{\partial a} = 2a$. When $a=11$, $\frac{\partial J}{\partial a} = 22$. This matches our arithmetic calculation above.
If you have not already done so, you can go back to the diagram above and fill in the value for $\frac{\partial J}{\partial a}$.

### $\frac{\partial J}{\partial w}$ 
Moving from right to left, the next value we would like to compute is $\frac{\partial J}{\partial w}$. To do this, we first need to calculate $\frac{\partial a}{\partial w}$ which describes how the output of this node, $a$, changes when the input $w$ changes a little bit.

#### Arithmetically
Find $\frac{\partial a}{\partial w}$ by finding how $a$ changes as a result of a little change in $w$.

```py
w_epsilon = w + 0.001       # a  plus a small value, epsilon
a_epsilon = 2 + 3*w_epsilon
k = (a_epsilon - a)/0.001   # difference divided by epsilon
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")

# a = 11, a_epsilon = 11.003, da_dw ~= k = 3.0000000000001137 
```

Calculated arithmetically,  $\frac{\partial a}{\partial w} \approx 3$. Let's try it with SymPy.

```py
sa = 2 + 3*sw
sa

da_dw = diff(sa,sw)
da_dw
```

![](2024-02-13-01-57-09.png)

```py
dJ_dw = da_dw * dJ_da
dJ_dw
```
![](2024-02-13-01-57-25.png)

And $a$ is 11 in this example so $\frac{\partial J}{\partial w} = 66$. We can check this arithmetically:

```py
w_epsilon = w + 0.001
a_epsilon = 2 + 3*w_epsilon
J_epsilon = a_epsilon**2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
# J = 121, J_epsilon = 121.06600900000001, dJ_dw ~= k = 66.0090000000082 
```

**Another view**  
One could visualize these cascading changes this way:  

![](2024-02-13-01-58-46.png)

A small change in $w$ is multiplied by $\frac{\partial a}{\partial w}$ resulting in a change that is 3 times as large. This larger change is then multiplied by $\frac{\partial J}{\partial a}$ resulting in a change that is now $3 \times 22 = 66$ times larger.

### Computation Graph of a Simple Neural Network
Below is a graph of the neural network used in the lecture with different values. Try and fill in the values in the boxes. Note, the interactive graphic is not particularly robust. If you run into trouble with the interface, run the cell below again to restart.

![](2024-02-13-01-59-16.png)

Below, we will go through the computations required to fill in the above computation graph in detail. We start with the forward path.

### Forward propagation
The calculations in the forward path are the ones you have recently learned for neural networks. You can compare the values below to those you calculated for the diagram above.

```py
# Inputs and parameters
x = 2
w = -2
b = 8
y = 1
# calculate per step values   
c = w * x
a = c + b
d = a - y
J = d**2/2
print(f"J={J}, d={d}, a={a}, c={c}")
# J=4.5, d=3, a=4, c=-4
```

### Backward propagation (Backprop)
As described in the lectures, backprop starts at the right and moves to the left. The first node to consider is $J = \frac{1}{2}d^2 $ and the first step is to find $\frac{\partial J}{\partial d}$

### $\frac{\partial J}{\partial d}$ 

#### Arithmetically
Find $\frac{\partial J}{\partial d}$ by finding how $J$ changes as a result of a little change in $d$.

```py
d_epsilon = d + 0.001
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dd ~= k = {k} ")
# J = 4.5, J_epsilon = 4.5030005, dJ_dd ~= k = 3.0004999999997395 
```

$\frac{\partial J}{\partial d}$ is 3, which is the value of $d$. Our result is not exactly $d$ because our epsilon value is not infinitesimally small. 
#### Symbolically
Now, let's use SymPy to calculate derivatives symbolically, as we did in the derivatives optional lab. We will prefix the name of the variable with an 's' to indicate this is a *symbolic* variable.

```py
sx,sw,sb,sy,sJ = symbols('x,w,b,y,J')
sa, sc, sd = symbols('a,c,d')
sJ = sd**2/2
sJ

sJ.subs([(sd,d)])

dJ_dd = diff(sJ, sd)
dJ_dd
```

![](2024-02-13-02-00-52.png)

So, $\frac{\partial J}{\partial d}$ = d. When $d=3$, $\frac{\partial J}{\partial d}$ = 3. This matches our arithmetic calculation above.

If you have not already done so, you can go back to the diagram above and fill in the value for $\frac{\partial J}{\partial d}$.

### $\frac{\partial J}{\partial a}$ 
![](2024-02-13-02-01-29.png)

Moving from right to left, the next value we would like to compute is $\frac{\partial J}{\partial a}$. To do this, we first need to calculate $\frac{\partial d}{\partial a}$ which describes how the output of this node changes when the input $a$ changes a little bit. (Note, we are not interested in how the output changes when $y$ changes since $y$ is not a parameter.)

#### Arithmetically
Find $\frac{\partial d}{\partial a}$ by finding how $d$ changes as a result of a little change in $a$.

```py
a_epsilon = a + 0.001         # a  plus a small value
d_epsilon = a_epsilon - y
k = (d_epsilon - d)/0.001   # difference divided by epsilon
print(f"d = {d}, d_epsilon = {d_epsilon}, dd_da ~= k = {k} ")
# d = 3, d_epsilon = 3.0010000000000003, dd_da ~= k = 1.000000000000334 
```
Calculated arithmetically,  $\frac{\partial d}{\partial a} \approx 1$. Let's try it with SymPy.
#### Symbolically

```py
sd = sa - sy
sd

dd_da = diff(sd,sa)
dd_da
```

![](2024-02-13-02-02-49.png)

Calculated arithmetically,  $\frac{\partial d}{\partial a}$ also equals 1.  
>The next step is the interesting part, repeated again in this example:
> - We know that a small change in $a$ will cause $d$ to change by 1 times that amount.
> - We know that a small change in $d$ will cause $J$ to change by $d$ times that amount. (d=3 in this example)    
 so, putting these together, 
> - We  know that a small change in $a$ will cause $J$ to change by $1\times d$ times that amount.
> 
>This is again *the chain rule*.  It can be written like this: 
 $$\frac{\partial J}{\partial a} = \frac{\partial d}{\partial a} \frac{\partial J}{\partial d} $$
 
Let's try calculating it:

```py
dJ_da = dd_da * dJ_dd
dJ_da
```
![](2024-02-13-02-03-15.png)

And $d$ is 3 in this example so $\frac{\partial J}{\partial a} = 3$. We can check this arithmetically:

```py
a_epsilon = a + 0.001
d_epsilon = a_epsilon - y
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")
```

> **The steps in backprop**   
>Now that you have worked through several nodes, we can write down the basic method:\
> working right to left, for each node:
>- calculate the local derivative(s) of the node
>- using the chain rule, combine with the derivative of the cost with respect to the node to the right.   

The 'local derivative(s)' are the derivative(s) of the output of the current node with respect to all inputs or parameters.

Let's continue the job. We'll be a bit less verbose now that you are familiar with the method.

### $\frac{\partial J}{\partial c}$,  $\frac{\partial J}{\partial b}$

![](2024-02-13-02-04-00.png)

The next node has two derivatives of interest. We need to calculate  $\frac{\partial J}{\partial c}$ so we can propagate to the left. We also want to calculate   $\frac{\partial J}{\partial b}$. Finding the derivative of the cost with respect to the parameters $w$ and $b$ is the object of backprop. We will find the local derivatives,  $\frac{\partial a}{\partial c}$ and  $\frac{\partial a}{\partial b}$ first and then combine those with the derivative coming from the right, $\frac{\partial J}{\partial a}$.

```py
# calculate the local derivatives da_dc, da_db
sa = sc + sb
sa

da_dc = diff(sa,sc)
da_db = diff(sa,sb)
print(da_dc, da_db)

dJ_dc = da_dc * dJ_da
dJ_db = da_db * dJ_da
print(f"dJ_dc = {dJ_dc},  dJ_db = {dJ_db}")
```

![](2024-02-13-02-04-36.png)

And in our example, d = 3.

###  $\frac{\partial J}{\partial w}$

![](2024-02-13-02-05-02.png)

The last node in this example calculates `c`. Here, we are interested in how J changes with respect to the parameter w. We will not back propagate to the input $x$, so we are not interested in $\frac{\partial J}{\partial x}$. Let's start by calculating $\frac{\partial c}{\partial w}$.

```py
# calculate the local derivative
sc = sw * sx
sc

dc_dw = diff(sc,sw)
dc_dw
```
![](2024-02-13-02-05-35.png)

This derivative is a bit more exciting than the last one. This will vary depending on the value of $x$. This is 2 in our example.

Combine this with $\frac{\partial J}{\partial c}$ to find $\frac{\partial J}{\partial w}$.

```py
dJ_dw = dc_dw * dJ_dc
dJ_dw
# 𝑑𝑥

print(f"dJ_dw = {dJ_dw.subs([(sd,d),(sx,x)])}")
# dJ_dw = 2*d
```

$d=3$,  so $\frac{\partial J}{\partial w} = 6$ for our example.   
Let's test this arithmetically:

```py
J_epsilon = ((w+0.001)*x+b - y)**2/2
k = (J_epsilon - J)/0.001  
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

They match! Great.

### Congratulations!
You've worked through an example of back propagation using a computation graph. You can apply this to larger examples by following the same node by node approach. 