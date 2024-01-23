# Activation functions

## Alternatives to the sigmoid activation

So far, we've been using the sigmoid activation function in all the nodes in the hidden layers and in the output layer. And we have started that way because we were building up neural networks by taking logistic regression and creating a lot of logistic regression units and string them together. 

But if we use other activation functions, our neural network can become much more powerful. Let's take a look at how to do that. Recall the demand prediction example from last week where given price, shipping cost, marketing, material, we would try to predict if something is highly affordable. 

If there's good awareness and high perceived quality and based on that try to predict it was a top seller. But this assumes that awareness is maybe binary is either people are aware or they are not. But it seems like the degree to which possible buyers are aware of the t shirt we're selling may not be binary, they can be a little bit aware, somewhat aware, extremely aware or it could have gone completely viral. 

So rather than modeling awareness as a binary number 0, 1, that we try to estimate the probability of awareness or rather than modeling awareness is just a number between 0 and 1. Maybe awareness should be any non negative number because there can be any non negative value of awareness going from 0 up to very very large numbers. So whereas previously we had used this equation to calculate the activation of that second hidden unit estimating awareness where g was the sigmoid function and just goes between 0 and 1. 

If we want to allow a,1, 2 to potentially take on much larger positive values, we can instead swap in a different activation function. It turns out that a very common choice of activation function in neural networks is this function. It looks like this. 

It goes if z is this, then g(z) is 0 to the left and then there's this straight line 45° to the right of 0. And so when z is greater than or equal to 0, g(z) is just equal to z. That is to the right half of this diagram. 

And the mathematical equation for this is g(z) equals max(0, z). Feel free to verify for yourself that max(0, z) results in this curve that I've drawn over here. And if a 1, 2 is g(z) for this value of z, then a, the deactivation value cannot take on 0 or any non negative value. 

This activation function has a name. It goes by the name ReLU with this funny capitalization and ReLU stands for again, somewhat arcane term, but it stands for rectified linear unit. Don't worry too much about what rectified means and what linear unit means. 

This was just the name that the authors had given to this particular activation function when they came up with it. But most people in deep learning just say ReLU to refer to this g(z). More generally we have a choice of what to use for g(z) and sometimes we'll use a different choice than the sigmoid activation function. 

Here are the most commonly used activation functions. we saw the sigmoid activation function, g(z) equals this sigmoid function. On the last slide we just looked at the ReLU or rectified linear unit g(z) equals max(0, z). 

There's one other activation function which is worth mentioning, which is called the linear activation function, which is just g(z) equals to z. Sometimes if we use the linear activation function, people will say we're not using any activation function because if a is g(z) where g(z) equals z, then a is just equal to this w.x plus b z. And so it's as if there was no g in there at all. 

So when we are using this linear activation function g(z) sometimes people say, well, we're not using any activation function. Although in this class, I will refer to using the linear activation function rather than no activation function. But if we hear someone else use that terminology, that's what they mean. 

It just refers to the linear activation function. And these three are probably by far the most commonly used activation functions in neural networks. Later this week, we'll touch on the fourth one called the softmax activation function. 

But with these activation functions we'll be able to build a rich variety of powerful neural networks. So when building a neural network for each neuron, do we want to use the sigmoid activation function or the ReLU activation function? Or a linear activation function? 

How do we choose between these different activation functions? Let's take a look at that in the next video. ## Choosing activation functions

Let's take a look at how we can choose the activation function for different neurons in our neural network. 

We'll start with some guidance for how to choose it for the output layer. It turns out that depending on what the target label or the ground truth label y is, there will be one fairly natural choice for the activation function for the output layer, and we'll then go and look at the choice of the activation function also for the hidden layers of our neural network. Let's take a look. 

we can choose different activation functions for different neurons in our neural network, and when considering the activation function for the output layer, it turns out that there'll often be one fairly natural choice, depending on what is the target or the ground truth label y. Specifically, if we are working on a classification problem where y is either zero or one, so a binary classification problem, then the sigmoid activation function will almost always be the most natural choice, because then the neural network learns to predict the probability that y is equal to one, just like we had for logistic regression. My recommendation is, if we're working on a binary classification problem, use sigmoid at the output layer. 

Alternatively, if we're solving a regression problem, then we might choose a different activation function. For example, if we are trying to predict how tomorrow's stock price will change compared to today's stock price. Well, it can go up or down, and so in this case y would be a number that can be either positive or negative, and in that case I would recommend we use the linear activation function. 

Why is that? Well, that's because then the outputs of our neural network, f of x, which is equal to a^3 in the example above, would be g applied to z^3 and with the linear activation function, g of z can take on either positive or negative values. So y can be positive or negative, use a linear activation function. 

Finally, if y can only take on non-negative values, such as if we're predicting the price of a house, that can never be negative, then the most natural choice will be the ReLU activation function because as we see here, this activation function only takes on non-negative values, either zero or positive values. In choosing the activation function to use for our output layer, usually depending on what is the label y we're trying to predict, there'll be one fairly natural choice. In fact, the guidance on this slide is how I pretty much always choose my activation function as well for the output layer of a neural network. 

How about the hidden layers of a neural network? It turns out that the ReLU activation function is by far the most common choice in how neural networks are trained by many practitioners today. Even though we had initially described neural networks using the sigmoid activation function, and in fact, in the early history of the development of neural networks, people use sigmoid activation functions in many places, the field has evolved to use ReLU much more often and sigmoids hardly ever. 

Well, the one exception that we do use a sigmoid activation function in the output layer if we have a binary classification problem. So why is that? Well, there are a few reasons. 

First, if we compare the ReLU and the sigmoid activation functions, the ReLU is a bit faster to compute because it just requires computing max of 0, z, whereas the sigmoid requires taking an exponentiation and then a inverse and so on, and so it's a little bit less efficient. But the second reason which turns out to be even more important is that the ReLU function goes flat only in one part of the graph; here on the left is completely flat, whereas the sigmoid activation function, it goes flat in two places. It goes flat to the left of the graph and it goes flat to the right of the graph. 

If we're using gradient descent to train a neural network, then when we have a function that is fat in a lot of places, gradient descents would be really slow. I know that gradient descent optimizes the cost function J of W, B rather than optimizes the activation function, but the activation function is a piece of what goes into computing, and that results in more places in the cost function J of W, B that are flats as well and with a small gradient and it slows down learning. I know that that was just an intuitive explanation, but researchers have found that using the ReLU activation function can cause our neural network to learn a bit faster as well, which is why for most practitioners if we're trying to decide what activation functions to use with hidden layer, the ReLU activation function has become now by far the most common choice. 

In fact that we're building a neural network, this is how I choose activation functions for the hidden layers as well. To summarize, here's what I recommend in terms of how we choose the activation functions for our neural network. For the output layer, use a sigmoid, if we have a binary classification problem; linear, if y is a number that can take on positive or negative values, or use ReLU if y can take on only positive values or zero positive values or non-negative values. 

Then for the hidden layers I would recommend just using ReLU as a default activation function, and in TensorFlow, this is how we would implement it. Rather than saying activation equals sigmoid as we had previously, for the hidden layers, that's the first hidden layer, the second hidden layer as TensorFlow to use the ReLU activation function, and then for the output layer in this example, I've asked it to use the sigmoid activation function, but if we wanted to use the linear activation function, is that, that's the syntax for it, or if we wanted to use the ReLU activation function that shows the syntax for it. With this richer set of activation functions, we'll be well-positioned to build much more powerful neural networks than just once using only the sigmoid activation function. 

By the way, if we look at the research literature, we sometimes hear of authors using even other activation functions, such as the tan h activation function or the LeakyReLU activation function or the swish activation function. Every few years, researchers sometimes come up with another interesting activation function, and sometimes they do work a little bit better. For example, I've used the LeakyReLU activation function a few times in my work, and sometimes it works a little bit better than the ReLU activation function we've learned about in this video. 

But I think for the most part, and for the vast majority of applications what we learned about in this video would be good enough. Of course, if we want to learn more about other activation functions, feel free to look on the Internet, and there are just a small handful of cases where these other activation functions could be even more powerful as well. With that, I hope we also enjoy practicing these ideas, these activation functions in the optional labs and in the practice labs. 

But this raises yet another question. Why do we even need activation functions at all? Why don't we just use the linear activation function or use no activation function anywhere? 

It turns out this does not work at all. In the next video, let's take a look at why that's the case and why activation functions are so important for getting our neural networks to work. ## Why do we need activation functions? 

Let's take a look at why neural networks need activation functions and why they just don't work if we were to use the linear activation function in every neuron in the neural network. Recall this demand prediction example. What would happen if we were to use a linear activation function for all of the nodes in this neural network? 

It turns out that this big neural network will become no different than just linear regression. So this would defeat the entire purpose of using a neural network because it would then just not be able to fit anything more complex than the linear regression model that we learned about in the first course. Let's illustrate this with a simpler example. 

Let's look at the example of a neural network where the input x is just a number and we have one hidden unit with parameters w1 and b1 that outputs a1, which is here, just a number, and then the second layer is the output layer and it has also just one output unit with parameters w2 and b2 and then output a2, which is also just a number, just a scalar, which is the output of the neural network f of x. Let's see what this neural network would do if we were to use the linear activation function g of z equals z everywhere. So to compute a1 as a function of x, the neural network will use a1 equals g of w1 times x plus b1. 

But g of z is equal to z. So this is just w1 times x plus b1. Then a2 is equal to w2 times a1 plus b2, because g of z equals z. 

Let me take this expression for a1 and substitute it in there. So that becomes w2 times w1 x plus b1 plus b2. If we simplify, this becomes w2, w1 times x plus w2, b1 plus b2. 

It turns out that if I were to set w equals w2 times w1 and set b equals this quantity over here, then what we've just shown is that a2 is equal to w x plus b. So a2 is just a linear function of the input x. Rather than using a neural network with one hidden layer and one output layer, we might as well have just used a linear regression model. 

If we're familiar with linear algebra, this result comes from the fact that a linear function of a linear function is itself a linear function. This is why having multiple layers in a neural network doesn't let the neural network compute any more complex features or learn anything more complex than just a linear function. So in the general case, if we had a neural network with multiple layers like this and say we were to use a linear activation function for all of the hidden layers and also use a linear activation function for the output layer, then it turns out this model will compute an output that is completely equivalent to linear regression. 

The output a4 can be expressed as a linear function of the input features x plus b. Or alternatively, if we were to still use a linear activation function for all the hidden layers, for these three hidden layers here, but we were to use a logistic activation function for the output layer, then it turns out we can show that this model becomes equivalent to logistic regression, and a4, in this case, can be expressed as 1 over 1 plus e to the negative wx plus b for some values of w and b. So this big neural network doesn't do anything that we can't also do with logistic regression. 

That's why a common rule of thumb is don't use the linear activation function in the hidden layers of the neural network. In fact, I recommend typically using the ReLU activation function should do just fine. So that's why a neural network needs activation functions other than just the linear activation function everywhere. 

So far, we've learned to build neural networks for binary classification problems where y is either zero or one. As well as for regression problems where y can take negative or positive values, or maybe just positive and non-negative values. In the next video, I'd like to share with we a generalization of what we've seen so far for classification. 

In particular, when y doesn't just take on two values, but may take on three or four or ten or even more categorical values. Let's take a look at how we can build a neural network for that type of classification problem.