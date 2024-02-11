# Back Propagations

## Computation graph

The computation graph is a key idea in deep learning, and it is also how programming frameworks like TensorFlow, automatic compute derivatives of our neural networks. Let's take a look at how it works. 

Let me illustrate the concept of a computation graph with a small neural network example. This neural network has just one layer, which is also the output layer, and just one unit in the output layer. It takes us inputs x, applies a linear activation function and outputs deactivation a. 

More specifically this output is a equals wx plus b. This basically linear regression, but expressed as a neural network with one output unit. Given the output, the cause function is then 1/2a, that is the predicted value minus the actual observed value of y. 

For this small example, we're only going to have a single training example, where the training example is the input $x$ equals negative 2. The ground truth output value $y$ equals 2, and the parameters of this network are; $w$ equals 2 and $b$ equals 8. What I'd like to do is show how the computation of the cause function J can be computed step by step using a computation graph. 

Just as a reminder, when learning, we like to view the cause function J as a function of the parameters $w$ and b. Let's take the computation of J and break it down into individual steps. First, we have the parameter $w$ that is an input to the cause function J, and then we first need to compute $w$ times x. 

Let me just call that c as follows; $w$ is equal to 2, $x$ is equal to negative 2, and so c would be negative 4. we're just going to write the value here on top of this arrow to show the value that is our output on this arrow. The next step is then to compute a, which is wx plus b. 

So let me create another node here. This needs to input b, the other parameter that is input to the cause function J, and a equals wx plus $b$ is equal to c plus b. If we add these up, that turns out to be 4. 

This is starting to build up a computation graph in which the steps we need to compute the cause function J are broken down into smaller steps. The next step is to then compute a minus y, which we're going to call d. Let me have that node d, which computes a minus y. 

Y is equal to 2, so 4 minus 2 is 2. Then finally, J is the cause is 1/2 of a minus $y$ squared, or 1/2 of d squared, which is just equal to 2. What we've just done is build up a computation graph. 

This is a graph, not in a sense of plots with $x$ and $y$ axes, but this is the other sense of the word graph using computer science, which is that this is a set of nodes that is connected by edges or connected by arrows in this case. This computation graph shows the forward prop step of how we compute the output a of the neural network. But then also go further than that so also compute the value of the cause function J. 

The question now is, how do we find the derivative of J with respect to the parameters $w$ and b? Let's take a look at that next. Here's the computation graph from the previous slide and we've completed for a problem where we've computed that J, the cause function is equal to 2 through all these steps going from left to right for a prop in the computation graph. 

What we'd like to do now is compute the derivative of J with respect to $w$ and the derivative of J with respect to b. It turns out that whereas for a prop was a left to right calculation, computing the derivatives will be a right to left calculation, which is why it's called backprop, was going backwards from right to left. The final computation nodes of this graph is this one over here, which computes J equals 1/2 of d squared. 

The first step of backprop will ask if the value of d, which was the input to this node where the change a little bit. How much does the value of j change? Specifically, will ask if d were to go up by a little bit, say 0.001, and that'll be our value of Epsilon in this case, how would the value of j change? 

It turns out in this case if d goes from 2-2.01, then j goes from 2-2.02. So if d goes up by Epsilon, j goes up by roughly two times Epsilon. We conclude that the derivative of J with respect to this value d that is inputted this final node is equal to two. 

The first step of backprop would be to fill in this value two over here, where this value is the derivative of j with respect to this input value d. We know if d changes a little bit, j changes by twice as much because this derivative is equal to two. The next step is to look at the node before that and ask what is the derivative of j with respect to a? 

To answer that, we have to ask, well, if a goes up by 0.001, how does that change j? Well, we know that if a goes up by 0.001, d is just a minus y. If a becomes 4.001, d which is a minus y, becomes 4.001 minus $y$ equals 2, so becomes 2.001, sub a goes up by 0.001, d also goes up by 0.001. 

But we'd already concluded previously that the d goes up by 0.001, j goes up by twice as much. Now we know if a goes up by 0.001, d goes up by 0.001, then j goes up roughly by two times 0.001. This tells us that the derivative of j with respect to a is also equal to two. 

So we're going to fill in that value over here. That this is the derivative of j with respect to a. Just as this was the derivative of j respect to d. 

If we've taken a calculus class before and if we've heard of the chain rule, we might recognize that this step of computation that I just did is actually relying on the chain rule for calculus. If we're not familiar with the chain rule, don't worry about it. we won't need to know it for the rest of these sections. 

But if we have seen the chain rule, we might recognize that the derivative of j with respect to a is asking, how much does d change respect to a, which is derivative of d respect to a times the derivative of j with respect to d, and does little calculation on top showed that the partial of t with respect to a is one, and we'd show EZ that the derivative of J with respect to d is equal to two, which is why the derivative of J with respect to a is one times two, which is equal to two. That's the value we got. But again, if we're not familiar with the chain rule, don't worry about it. 

The logic that we just went through here is why we know j goes up by twice as much as a does. That's why this derivative term is equal to two. The next step then is to keep on going right to left as we do in backprop. 

We'll ask, how much does a little change in c cause j to change, and how much does $y$ change in $b$ cause j to change? The way we figure that out is to ask, what if c goes up by Epsilon 0.001, how much does a change? Well, a is equal to c plus b. 

It turns out that if c ends up being negative 3.999, then a, which is negative 3.999 plus 8, becomes 4.001. If c goes up by Epsilon, a goes up by Epsilon. We know if a up by epsilon, then because the derivative of J with respect to a is two, we know that this in turn causes j to go up by two times Epsilon. 

We can conclude that if c goes up by a little bit, J goes up by twice as much. We know this because we know the derivative of J with respect to a is 2. This allows us to conclude that the derivative of J with respect to c is also equal to 2. 

we're going to fill in that value over here. Again, only if we're familiar with chain rule another way to write this is derivative of J respect to c, is the derivative of a respect to c. This turns out to be 1 times the derivative of J respect to a, which we have previously figured out was equal to 2, so that's why this ends up being equal to 2. 

By a similar calculation, the $b$ goes up by 0.001, then a also goes up by 0.001 and J goes up by 2 times 0.001, which is why this derivative is also equal to 2. We filled in here the derivative of J respect to b, and here the derivative of J respect to c. Now one final step, which is, what is the derivative of J with respect to w? 

W goes up by 0.001. What happens? C which is $w$ times x, if $w$ were 2.001, c which is $w$ times x, becomes negative 2 times 2.001, so it becomes negative 4.002. 

If $w$ goes up by epsilon, c goes down by 2 times 0.001, or equivalently c goes up by negative 2 times 0.001. We know that if c goes up by negative 2 times 0.001, because the derivative of J with respect to c is 2, this means that J will go up by negative 4 times 0.001, because if c goes up by a certain amount, J changes by 2 times as much, so negative 2 times this is negative 4 times this. This allows us to conclude that if $w$ goes up by 0.001, J goes up by negative 4 times 0.001. 

The derivative of J with respect to $w$ is negative 4. we're going to write negative 4 over here because has the derivative of J with respect to w. Once again, the chain rule calculation, if we're familiar with it, is this. 

It is the derivative of c respect to $w$ times derivative of J with respect to c. This is 2 and this is negative 2, which is why we end up with negative 4, but again, don't worry about it if we're not familiar with chain rule. To wrap up what we've just done this manually carry out backprop in this computation graph. 

Whereas forward prop was a left-to-right computation where we had $w$ equals 2, that allowed us to compute c. Then we had $b$ and that allows us to compute a and then d, and then J backprop went from right-to-left and we would first compute the derivative of J with respect to d and then go back to compute the derivative of J with respect to a, then the derivative of J with respect to b, derivative of J with respect to c and finally the derivative of J with respect to w. So that's why backprop is a right-to-left computation, whereas forward prop was a left-to-right computation. 

In fact, let's double-check the computation that we just did. J with these values of w, b, $x$ and $y$ is equal to one-half times wx plus $b$ minus $y$ squared, which is one-half times 2 times negative 2 plus 8 minus 2 squared, which is equal to 2. Now if $w$ were to go up by 0.001, then J becomes one-half times, $w$ is now 2.001 times x, which is negative 2, plus $b$ which is 8 minus $y$ squared. 

If we calculate this out, this turns out to be 1.996002. Roughly J has gone from 2 down to 1.996, and then an extra 002 and J has therefore gone down by 4 times epsilon. This shows that if W goes up by Epsilon, J goes down by four times Epsilon ball equivalent the J goes up by negative four times Epsilon, which is y. 

The derivative of j with respect to $w$ is negative 4, which is what we have worked out over here. If we want, feel free to pause the section and double-check this math yourself as well for what happens in b, the other parameter goes up by Epsilon, and hopefully we'll find that the derivative of j with respect to $b$ is indeed two. That $b$ goes up by Epsilon, j goes up by two times Epsilon as predicted by this derivative calculation. 

Why do we use the backprop algorithm to compute derivatives? It turns out that backdrop is an efficient way to compute derivatives. The reason we sequence this as a right-to-left calculation is, if we were to start off and ask what is the derivative of j with respect to w? 

Then to know how much change in $w$ affects change in j, if $w$ were to go up by Epsilon, how much does j go by Epsilon? Well, the first thing we want to know is, what is the derivative of j with respect to c? Because change in $w$ will change c this first quantity here. 

To know how much change in $w$ affects j, we want to know how much does change in c affects j. But to know how much change in c affects j, the most useful thing to know to compute this would be change in c changes a. we want to know how much this change in a effect j and so on. 

That's why backprop a sequence as a right-to-left calculation. Because if we do the calculation from right to left, we can find out how does change in d affect change in j. Then we can find out how much this change in a effect j and so on. 

Until we find the derivatives of each of these intermediate quantities, c, a, and d, as well as the parameters $w$ and b. That we can find out with one right-to-left calculation how much change in any of these intermediate quantities, c, a, or d, as well as the input parameters $w$ and b. How much change in any of these things will affect the final output value j. 

One thing that makes backprop efficient is we notice that when we do the right-to-left calculation, we had to compute this term, the derivative of j with respect to a just once. This quantity is then used to compute both the derivative of g with respect to $w$ and the derivative of j with respect to b. It turns out that if a computation graph has n nodes, meaning our n of these boxes and p parameters, so we have two parameters in this case. 

This procedure allows us to compute all the derivatives of j with respect to all the parameters in roughly n plus p steps, rather than n times p steps. If we have a neural network with, said, 10,000 nodes and maybe 100,000 parameters. This would not be considered even a very large neural network by modern standards. 

Being able to compute the derivatives and 10,000 plus 100,000 steps, which is punch and 10,000 is much better than meeting 10,000 times 100,000 steps, which would be a billion steps. The backpropagation algorithm done using the computation graph gives we a very efficient way to compute all the derivatives. That's why it is such a key idea in how deep learning algorithms are implemented today. 

In this section, we saw how the computation graph takes all the steps of the calculation needed to compute the output of a neural network a as well as the cost function j. Takes a step-by-step computations and breaks them into the different nodes of computation graph. Then uses a left-to-right computation for a prop to compute the cost function J. 

Then a right-to-left or backpropagation calculation to computes all the derivatives. In this section, we saw these ideas apply to a small neural network example. In the next section, let's take these ideas and apply them to a larger neural network. 

Let's go on to the next section. ## Large neural network example

In this final section on intuition for backprop, let's take a look at how the computation draft works on a larger neural network example. Here's the network we will use with a single hidden layer, with a single hidden unit that outputs a1, that feeds into the output layer that outputs the final prediction a2. 

To make the math more tractable, we're going to continue to use just a single training example with inputs $x$ = 1, $y$ = 5. And these will be the parameters of the network. And throughout we're going to use the ReLU activation functions of g(z) = max(0, z). 

So for prop in our network looks like this. As usual, a1 equals g($$w_1$$ times $x$ + $$b_1$$). And so it turns out w1x + $b$ will be positive. 

So we're in the max(0, z) = z, parts of this activation function. So that's just equal to this, which is 2 times 1, that's $$w_1$$ is 2 times x1 + 0, that's $$b_1$$, which is equal to 2. And then similarly, a2 equals this, g(w2a1 + $$b_2$$) which is $$w_2$$ times a1 + b. 

Again, because we're in the positive part of the ReLU activation function, which is 3 $x$ 2 + 1 = 7. Finally, we'll use the squared error cost function. So j(w, b) is 1/2(a2- y) squared = 1/2(7-5) squared, which is 1/2 of 2 squared, which is just equal to 2. 

So, let's take this calculation that we just did and write it down in the form of a computation graph. To carry out the computation step by step, first thing we need to do is take $$w_1$$ and multiply that by x. So we have $$w_1$$ that feeds into computation node that computes $$w_1$$ times x. 

And we're going to call this a temporary variable t1. Next, we compute z1, which is this term here, which is t1 + $$b_1$$. So we also have this input $$b_1$$ over here. 

And finally a1 equals g(z1). We apply the activation function, and so we end up with again this value here, 2. And then next we have to compute t2, which is $$w_2$$ times a1. 

And so, with $$w_2$$, that gives us this value which is 6. Then z2, which is this quantity, we had to $$b_2$$ it and that gives us 7. And finally apply the activation function, g. 

We still end up with 7. And lastly, j is 1/2(a2- y) squared. And that gives us 2. 

Which was this cost function here. So this is how we take the step by step computations for larger neural network and write it in the computation graph. we've already seen in the last section, the mechanics of how to carry out backprop. 

we're not going to go through the step by step calculations here. But if we were to carry out backprop, the first thing we do is ask, what is the derivative of the cost function j respect to a2? And it turns out if we calculate that, it turns out to be 2. 

So we'll fill that in here. And the next step will be asked, what's the derivative of the cost j respect to z2. And using this derivative that we computed previously, we can figure out that this turns out to be 2. 

Because if z goes up by epsilon, we can show that for the current setting of all the parameters a2 will go up by epsilon. And therefore, j will go up by 2 times epsilon. So this derivative is equal to 2, and so on. 

Step by step. We can then find out that the derivative of j respective $$b_2$$ is also equal to 2. The derivative respect to t2 is equal to 2, and so on, and so forth. 

Until eventually we've computed the derivative of j with respect to all the parameters $$w_1$$, $$b_1$$, $$w_2$$, and $$b_2$$. And so that's backprop. And again, I didn't go through the mechanical steps of every single step of backprop. 

But it's basically the process that we saw in the previous section. Let me just double check one of these examples. So we saw here that the derivative of j respect $$w_1$$ is equal to 6. 

So what this is predicting is that, if $$w_1$$ goes up by epsilon, j should go up by roughly 6 times epsilon. Let's step through the map and see if that really is true. These are the calculations that we did, again. 

And so if $w$ which was 2 were to be 2.001 goes up by epsilon, then a1 becomes, let's see, instead of 2, this is 2.001 as well. So a1 instead of 2 is now 2.001. So 3 $x$ 2.001 + 1, this gives us 7.003. 

And if a2 is 7.003, then just becomes 7.003- 5 squared. And so this becomes 2.003 squared over 2, which turns out to be equal to 2.006005. So ignoring some of the extra digits, we see from this little calculation that, if $$w_1$$ goes up by 0.001, j of $w$ has gone up from 2 to 2.006 roughly. 

So 6 times as much. And so the derivative of j with respect to $$w_1$$ is indeed equal to 6. And so the backprop procedure gives we a very efficient way to compute all of these derivatives. 

Which we can then feed into the gradient descent algorithm or the Adam optimization algorithm, to then train the parameters of our neural network. And again, the reason we use background for this is, is a very efficient way to compute all the derivatives of j respect to $$w_1$$, j respect to $$b_1$$, j respect to $$w_2$$, and j respect to $$b_2$$. I did just illustrate how we could bump up $$w_1$$ by a little bit and see how much j changes. 

But that was a left to right calculation. And then we had to do this procedure for each parameter, one parameter at a time. If we had to increase $w$ by 0.001 to see how that changes j. 

Increase $$b_1$$ by a little bit to see how that changes j, and increase every parameter, one at a time by a little bit to see how that changes j. Then this becomes a very inefficient calculation. And if we had N nodes in our computation graph and P parameters, this procedure would end up taking N times P steps, which is very inefficient. 

Whereas we got all four of these derivatives N + P, rather than N times P steps. And this makes a huge difference in practical neural networks, where the number of nodes and the number of parameters can be really large. So, that's the end of the section for this week. 

Thanks for sticking with me through the end of these optional sections. And I hope that we now have an intuition for when we use a program frameworks, like tensorflow, to train a neural network. What's actually happening under the hood and how is using the computation graph to efficiently compute derivatives for we. 

Many years ago, before the rise of frameworks like tensorflow and pytorch, researchers used to have to manually use calculus to compute the derivatives of the neural networks that they wanted to train. And so in modern program frameworks we can specify forwardprop and have it take care of backprop for we. Many years ago, researchers used to write down the neural network by hand, manually use calculus to compute the derivatives. 

And then neural implement a bunch of equations that they had laboriously derived on paper, to implement backprop. Thanks to the computation graph and these techniques for automatically carrying out derivative calculations. Is sometimes called autodiff, for automatic differentiation. 

This process of researchers manually using calculus to take derivatives is no longer really done. At least, I've not had to do this for many years now myself, because of autodiff. So, many years ago, to use neural networks, the bar for the amount of calculus we have to know actually used to be higher. 

But because of automatic differentiation algorithms, usually based on the computation graph, we can now implement a neural network and get derivatives computed for we easier than before. So maybe with the maturing of neural networks, the amount of calculus we need to know in order to get these algorithms work, has actually gone down. And that's been encouraging for a lot of people. 

And so, that's it for the sections for this week. I hope we enjoy the labs and I look forward to seeing we next week.