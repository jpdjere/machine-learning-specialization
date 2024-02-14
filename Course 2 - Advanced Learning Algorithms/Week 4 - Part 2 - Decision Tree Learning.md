# Decision tree learning

## Measuring purity

In this section, we'll look at the way of **measuring the purity of a set of examples**. 

If the examples are all cats of a single class then that's very pure, if it's all "not cats" that's also very pure. But if it's somewhere in between how do we quantify how pure is the set of examples? 

Let's take a look at **the definition of entropy, which is a measure of the impurity of a set of data**:

Given a set of six examples, three cats and three dogs, let's define $p_1$ to be the fraction of examples that are cats, that is, the fraction of examples with label one. $p_1$ in this example is equal to 3/6.

![](2024-02-14-13-33-14.png)

We're going to measure the impurity of a set of examples using a function called the entropy function, which looks as follows:

![](2024-02-14-13-34-07.png)

The entropy function is conventionally denoted as **H** of the number $p_1$: $H(p_1)$.

The function curve has as its horizontal axis $p_1$, the fraction of cats in the sample, and the vertical axis is the value of the entropy. In this example where $p_1$ is 3/6 or 0.5, the value of the entropy of $p_1$ would be equal to one.

![](2024-02-14-13-36-04.png)

Notice that this curve is highest when our set of examples is 50-50, so it's most impure: it has an impurity or entropy of 1 when our set of examples is 50-50. In contrast, if our set of examples was either all cats or "not cats" then the entropy is 0.

![](2024-02-14-13-37-35.png)

Let's just go through a few more examples to gain further intuition about entropy and how it works. Here's a different set of examples with five cats and one dog, so $p_1$ the fraction of positive examples, a fraction of examples labeled one is 5/6 and so $p_1$ is about 0.83. If we read off that value at about 0.83 we find that the entropy of $p_1$ is about 0.65.

![](2024-02-14-13-38-22.png)

One more examples: the following sample of six images has all cats. So $p_1$ is 6/6 because all six are cats and the entropy of $p_1$ is this point over here which is zero. We see that **as we go from 3/6 to 6/6 cats, the impurity decreases from 1 to 0 or in other words, the purity increases as we go from a 50-50 mix of cats and dogs to all cats. **

![](2024-02-14-13-39-38.png)

Let's look at a few more examples. 

Here's another sample with two cats and four dogs: so $p_1$ here is 2/6 which is 1/3. If we read off the entropy at 0.33 it turns out to be about 0.92. 

![](2024-02-14-13-41-05.png)


This is actually quite impure. In particular, this set is more impure the third set, which has an entropy of 0.65, because it's closer to a 50-50 mix.

![](2024-02-14-13-42-22.png)

Finally, one last example: if we have a set of all six dogs then $p_1$ is equal to 0 and the entropy of $p_1$ is equal to 0. So  there's zero impurity: a completely pure set of all "not cats" or all dogs.

![](2024-02-14-13-43-27.png)

Now, let's look at **the actual equation for the entropy function $H(p_1)$**. Recall that $p_1$ is the fraction of examples that are equal to cats so if we have a sample that is 2/3 cats then that sample must have 1/3 "not cats". Let's define $p_0$ to be equal to the fraction of examples that are "not cats" to be just equal to 1 minus $p_1$. 

![](2024-02-14-13-44-54.png)

The entropy function is then defined: 

$$ -p_1 log_2(p_1) - p_0 log_2(p_0) $$


and by convention when computing entropy we use log of base-2 rather than base-e,

Alternatively, this is also equal:

$$ -p_1 log_2(p_1) - (1-p_1) \space log_2(1-p_1) $$

**We take $log_2$ just to make the peak of this curve equal to one**. If we were to take $log_e$ or the base of natural logarithms, then that just vertically scales this function. It would still work but the numbers become a bit hard to interpret because the peak of the function isn't a round number like 1 anymore. 

One note on computing this function: if $p_1$ or $p_0$ is equal to 0 then one of the terms will look like $0 \space log(0)$, which is technically undefined (tends to negative infinity). But by convention, for the purposes of computing entropy, we'll take $0 \space log(0)$, to be equal to 0. That will correctly compute the entropy as zero.

If we're thinking that this definition of entropy looks a little bit like the definition of the logistic loss that we learned about in the last course, there is actually a mathematical rationale for why these two formulas look so similar. 

To summarize: the entropy function is a measure of the impurity of a set of data. **It starts from zero, goes up to one, and then comes back down to zero as a function of the fraction of positive examples in our sample.** 

There are other functions that look, they go from zero up to one and then back down. For example, if we look in open source packages we may also hear about something called the Gini criteria, which is another function that looks a lot like the entropy function, and that will work well as well for building decision trees. But for the sake of simplicity, in these sections we're going to focus on using the entropy criteria which will usually work just fine for most applications. Now that we have this definition of entropy, in the next section let's take a look at how we can actually use it to make decisions as to what feature to split on in the nodes of a decision tree. 

## Choosing a split: Informational gain

**When building a decision tree, the way we will decide what feature to split on at a node will be based on what choice of feature reduces entropy the most.** "Reduces entropy" or "reduces impurity" which also means "maximizes purity". 

In decision tree learning, **the reduction of entropy is called information gain.**

Let's take a look, in this section, at how to compute information gain and therefore choose what features to use to split on at each node in a decision tree. 

Let's use the example of deciding what feature to use at the root node of the decision tree we were building for recognizing cats versus "not cats".

If we had split **using their ear shape feature at the root node**, we would have gotten five examples on the left and five on the right:

![](2024-02-14-13-55-22.png)

On the left, we would have 4/5 cats, so $p_1$ would be equal to 4/5 or 0.8. On the right, 1/5 are cats, so $p_1$ is equal to 1/5 or 0.2. If we apply the entropy formula from the last section to this left subset of data and this right subset of data, we find that the degree of impurity on the left is $H(0.8)$, which is about 0.72, and on the right, the entropy of 0.2, $H(0.2)$ turns out also to be 0.72. 

![](2024-02-14-13-56-31.png)

One other option would be to **split on the face shape feature on the root node**. If we'd done so, then on the left, 4/7 examples would be cats, so $p_1$ is 4/7, and on the right, 1/3 are cats, so $p_1$ on the right is 1/3. 

The entropy of 4/7 and the entropy of 1/3 are 0.99 and 0.92, respectively. So the degree of impurity in the left and right nodes seems much higher, 0.99 and 0.92 compared to 0.72 and 0.72, from the previous choice of feature.

![](2024-02-14-14-06-29.png)

Finally, the third possible **choice of feature to use at the root node would be the whiskers feature**. In this case, $p_1$ on the left is 3/4, $p_1$ on the right is 2/6, and the entropy values are as seen below: 

![](2024-02-14-14-07-12.png)

The key question we need to answer is: g**iven these three options of a feature to use at the root node, which one works best?** Rather than looking at these entropy numbers and comparing them, it would be useful to **take a weighted average of them**:


If there's a node with a lot of examples in it, that has high entropy, that is "worse" than if there was a node with just a few examples in it, with also high entropy. **This is because entropy, as a measure of impurity, is worse if we have a very large and impure dataset compared to just a few examples and a branch of the tree that is very impure.** 

Of these three possible choices of features to use at the root node, which one do we want to use? 

So, associated with each of these splits is two numbers, the entropy on the left sub-branch and the entropy on the right sub-branch. In order to pick from these, we like to actually combine these two numbers into a single number by taking a weighted average.

In the first example we have that 5/10 examples went to the left sub-branch, so we can compute the weighted average as 5/10 times the entropy of 0.8, and then add to that 5/10 examples also went to the right sub-branch, plus 5/10 times the entropy of 0.2.  And the same formula applies for all cases:

![](2024-02-14-14-11-45.png)

**The way we will choose a split is by computing these three numbers and picking whichever one is lowest**, since that gives us the left and right sub-branches with the lowest average weighted entropy. 

Because of the way that decision trees are built, we're actually going to make one more change to these formulas to stick to the convention in decision tree building, but it won't actually change the outcome: **rather than computing this weighted average entropy, we're going to compute the reduction in entropy compared to the case in which we hadn't split at all.** 

So, if we go to the root node, where we have started off with all 10 examples, with five cats and dogs, there we had $p_1$ equals 5/10 or 0.5. The entropy of the root node was $H(0.5)$, which is 1. This was maximum impurity:

![](2024-02-14-14-14-36.png)

So, the formula that we're actually going to use for choosing a split is not this weighted entropy at the left and right sub-branches. Instead it's going to be: **the entropy at the root node, $H(0.5)$, minus the formula for the weighted average. **

![](2024-02-14-14-16-15.png)

In this example, if we work out the math, it turns out to be 0.28 for the face shape feature, 0.03 for the face shape feature, and 0.12 for the whiskers feature. **These numbers are called the information gain**, and **they measure the reduction in entropy that we get in our tree resulting from making a split**. 

![](2024-02-14-14-20-49.png)

**Why do we bother to compute reduction in entropy rather than just entropy at the left and right sub-branches?** 

Recall that one of the stopping criteria for deciding when **not to split any further is if the reduction in entropy is too small**. In such a case we could decide that we would be simply increasing the size of the tree unnecessarily and risking overfitting by splitting. So we just decide to not bother if the reduction in entropy is too small or below a threshold. 

In this particular example, **splitting on ear shape results in the biggest reduction in entropy,** since 0.28 is bigger than 0.03 or 0.12. We would therefore choose to split using the ear shape feature at the root node. 

One additional piece of notation that we'll introduce are: the fractions 5/10 and 5/10, 7/10 and 3/10, etc, we're going to call them $w_{\text{left}}$ because that's the fraction of examples that went to the left branch, and we're going to call the complement $w_{\text{right}}$ because that's the fraction of examples that went to the right branch. 

Let's now see **the general formula for how to compute information gain**:

Using the example of splitting on the ear shape feature, **let's define $p_1^{\text{left}}$ to be equal to the fraction of examples in the left subtree that have a positive label, i.e. that are cats**. In this example,$p_1^{\text{left}}$ will be equal to 4/5. Also, let's define $w_left$ to be the fraction of examples of all of the examples of the root node that went to the left sub-branch. So in this example, $w_left$ would be 5/10. And we can similarly define the same values for the right branch:

![](2024-02-14-14-29-29.png)

Let's also define $p_1^{\text{root}}$ to be the fraction of examples that are positive in the root node, which, in this case, would be 5/10 or 0.5. 

**Information gain** is then defined as:

$$ \text{Information gain} = H(p_1^{\text{root}}) - [w_{\text{left}} H(p_1^{\text{left}}) + w_{\text{right}} H(p_1^{\text{right}})] $$

With this definition of entropy, and we can calculate the information gain associated with choosing any particular feature to split on in the node. Then out of all the possible features we could choose to split on, **we can then pick the one that gives we the highest information gain**. 

That should result in, hopefully, increasing the purity of our subsets of data that we get on the left and right sub-branches of our decision tree.

Let's put all the things we've talked about together into the overall algorithm for building a decision tree given a training set. Let's go see that in the next section. 

## Putting it together

The information gain criteria lets us decide how to choose one feature to split a one-node. Let's take that and use that in multiple places through a decision tree in order to build a large decision tree with multiple nodes. 


First, the following is **the overall process of building a decision tree**:

1. Starts with all training examples at the root node of the tree.
2. Calculate the information gain for all possible features. Pick the feature to split on that gives the highest information gain.
3. Split the dataset into two subsets according to the selected feature, and create left and right branches of the tree. Send the training examples to either the left or the right branch, depending on the value of that feature for that example.
4. Keep on repeating the splitting process on both branches until the stopping criteria is met:
  -  when a node is 100% a single class (we have reached entropy of zero)
  - when further splitting a node will cause the tree to exceed the maximum depth that we had set
  - if information gain from an additional splits is less than the threshold
  - if the number of examples in a node is below a threshold

Let's look at an illustration of how this process will work: we started all of the examples at the root nodes and based on computing information gain for all three features, we decide that ear-shaped is the best feature to split on. Based on that, we create a left and right sub-branches and send the subsets of the data with pointy versus floppy ear to left and right sub-branches.

![](2024-02-14-14-39-19.png)

Let's for now just focus on the left sub-branch where we have five examples: if our splitting criteria is to keep splitting until everything in the node belongs to a single class, so either all cats or all dogs. We will look at this node and see if it meets the splitting criteria: it does not because there is a mix of cats and dogs here.

![](2024-02-14-14-40-48.png)

So, the next step is to then pick a feature to split on. We then go through the features one at a time and **compute the information gain of each of those features as if this node were the new root node of a decision tree that was trained using just five training examples**. We would compute the information gain for splitting on the whiskers feature, the information gain on splitting on the face shape feature and the ear shape feature.

The information gain for splitting on ear shape will be zero because all of these training example the same point ear shape. Between whiskers and face shape, face shape turns out to have a highest information gain. So, we're going to split on face shape and that allows us to build left and right sub branches as follows:

![](2024-02-14-14-46-30.png)

For the left sub-branch, we check for the criteria for whether or not we should stop splitting and we have all cats. So, **the stopping criteria is met**, and **we create a leaf node that makes a prediction of cat**. 

For the right sub-branch, we find that it is all dogs, so we also stop splitting since we've met the splitting criteria. We put a leaf node there, that predicts "not cat":

![](2024-02-14-14-48-03.png)

Having built out this left subtree, we can now turn our attention to building the right subtree. Let's' now again cover up the root node and the entire left subtree:

![](2024-02-14-14-48-32.png)

To build out the right subtree, we have five examples. Again, the first thing we do is check if the criteria to stop splitting has been met. Since the criteria is if all the examples are a single class, we know we have not met that criteria.

So we will decide to keep splitting in this right sub-branch as well. In fact, the procedure for building the right sub-branch will be a lot as if we were training a decision tree learning algorithm from scratch, where the dataset we have comprises just these five training examples. Again: computing information gain for all of the possible features to split on, we find that the whiskers feature has the highest information gain:

![](2024-02-14-14-50-40.png)

Now agaim, we check if the criteria to stop splitting are met, and we decide that they are. We end up with leaf nodes that predict cat and "not cat".

![](2024-02-14-14-51-26.png)

This is the overall process for building the decision tree. 

Notice that there's interesting aspects of what we've done: after we decided what to split on at the root node, the way we built the left subtree was by building a decision tree on a subset of five examples. The way we built the right subtree was by, again, building a decision tree on a subset of five examples.

![](2024-02-14-14-52-55.png)

In computer science, **this is an example of a recursive algorithm**: the way we build a decision tree at the root is by building other smaller decision trees in the left and the right sub-branches.

By the way, **how do we choose the maximum depth parameter?**

There are many different possible choices, but **some of the open-source libraries will have good default choices that we can use.**

One intuition is: **the larger the maximum depth, the bigger the decision tree we're willing to build.** This is a bit **like fitting a higher degree polynomial or training a larger neural network.** It lets the decision tree learn a more complex model, but it also increases the risk of overfitting if it is fitting a very complex function to our data. 

In theory, we could use cross-validation to pick parameters like the maximum depth, where we try out different values of the maximum depth and pick what works best on the cross-validation set. Although in practice, the open-source libraries have even somewhat better ways to choose this parameter for us. 

## Using one-hot encoding of categorical features

In the example we've seen so far each of the features could take on only one of two possible values. The ear shape was either pointy or floppy, the face shape was either round or not round and whiskers were either present or absent. But whether if we have features that can take on more than two discrete values, in this section we'll look at how we can use one-hot encoding to address features like that. 

Here's a new training set for our pet adoption center application where all the data is the same except for the ear shaped feature. Rather than ear shape only being pointy and floppy, it can now also take on an oval shape. And so the initial feature is still a categorical value feature but it can take on three possible values instead of just two possible values. 

And this means that when we split on this feature we end up creating three subsets of the data and end up building three sub branches for this tree. But in this section I'd like to describe a different way of addressing features that can take on more than two values, which is to use the one-hot encoding. In particular rather than using an ear shaped feature, they can take on any of three possible values. 

We're instead going to create three new features where one feature is, does this animal have pointy ears, a second is does their floppy ears and the third is does it have oval ears. And so for the first example whereas we previously had ear shape as pointy, we are now instead say that this animal has a value for the pointy ear feature of 1 and 0 for floppy and oval. Whereas previously for the second example, we previously said it had oval ears now we'll say that it has a value of 0 for pointy ears because it doesn't have pointy ears. 

It also doesn't have floppy ears but it does have oval ears which is why this value here is 1 and so on for the rest of the examples in the data set. And so instead of one feature taking on three possible values, we've now constructed three new features each of which can take on only one of two possible values, either 0 or 1. In a little bit more detail, if a categorical feature can take on k possible values, k was three in our example, then we will replace it by creating k binary features that can only take on the values 0 or 1. 

And we notice that among all of these three features, if we look at any role here, exactly 1 of the values is equal to 1. And that's what gives this method of future construction the name one-hot encoding. And because one of these features will always take on the value 1 that's the hot feature and hence the name one-hot encoding. 

And with this choice of features we're now back to the original setting of where each feature only takes on one of two possible values, and so the decision tree learning algorithm that we've seen previously will apply to this data with no further modifications. Just an aside, even though this week's material has been focused on training decision tree models the idea of using one-hot encodings to encode categorical features also works for training neural networks. In particular if we were to take the face shape feature and replace round and not round with 1 and 0 where round gets matter 1, not round gets matter 0 and so on. 

And for whiskers similarly replace presence with 1 and absence with 0. They noticed that we have taken all the categorical features we had where we had three possible values for ear shape, two for face shape and one for whiskers and encoded as a list of these five features. Three from the one-hot encoding of ear shape, one from face shape and from whiskers and now this list of five features can also be fed to a new network or to logistic regression to try to train a cat classifier. 

So one-hot encoding is a technique that works not just for decision tree learning but also lets we encode categorical features using ones and zeros, so that it can be fed as inputs to a neural network as well which expects numbers as inputs. So that's it, with a one-hot encoding we can get our decision tree to work on features that can take on more than two discrete values and we can also apply this to neural networks or linear regression or logistic regression training. But how about features that are numbers that can take on any value, not just a small number of discrete values. 

In the next section let's look at how we can get the decision tree to handle continuous value features that can be any number. 

## Continuous valued features

Let's look at how we can modify decision tree to work with features that aren't just discrete values but continuous values. That is features that can be any number. 

Let's start with an example, I have modified the cat adoption center of data set to add one more feature which is the weight of the animal. In pounds on average between cats and dogs, cats are a little bit lighter than dogs, although there are some cats are heavier than some dogs. But so the weight of an animal is a useful feature for deciding if it is a cat or not. 

So how do we get a decision tree to use a feature? The decision tree learning algorithm will proceed similarly as before except that rather than constraint splitting just on ear shape, face shape and whiskers. we have to consist splitting on ear shape, face shape whisker or weight. 

And if splitting on the weight feature gives better information gain than the other options. Then we will split on the weight feature. But how do we decide how to split on the weight feature? 

Let's take a look. Here's a plot of the data at the root. Not a plotted on the horizontal axis. 

The way to the animal and the vertical axis is cat on top and not cat below. So the vertical axis indicates the label, $y$ being 1 or 0. The way we were split on the weight feature would be if we were to split the data based on whether or not the weight is less than or equal to some value. 

Let's say 8 or some of the number. That will be the job of the learning algorithm to choose. And what we should do when constraints splitting on the weight feature is to consider many different values of this threshold and then to pick the one that is the best. 

And by the best I mean the one that results in the best information gain. So in particular, if we were considering splitting the examples based on whether the weight is less than or equal to 8, then we will be splitting this data set into two subsets. Where the subset on the left has two cats and the subset on the right has three cats and five dogs. 

So if we were to calculate our usual information gain calculation, we'll be computing the entropy at the root node N C p f 0.5 minus now 2/10 times entropy of the left split has two other two cats. So it should be 2/2 plus the right split has eight out of 10 examples and an entropy F. That's of the eight examples on the right three cats. 

To entry of 3/8 and this turns out to be 0.24. So this would be information gain if we were to split on whether the weight is less than equal to 8 but we should try other values as well. So what if we were to split on whether or not the weight is less than equal to 9 and that corresponds to this new line over here. 

And the information gain calculation becomes H (0.5) minus. So now we have four examples and left split all cats. So that's 4/10 times entropy of 4/4 plus six examples on the right of which we have one cat. 

So that's 6/10 times each of 1/6, which is equal to turns out 0.61. So the information gain here looks much better is 0.61 information gain which is much higher than 0.24. Or we could try another value say 13. 

And the calculation turns out to look, which is 0.40. In the more general case, we'll actually try not just three values, but multiple values along the X axis. And one convention would be to sort all of the examples according to the weight or according to the value of this feature and take all the values that are mid points between the sorted list of training. 

Examples as the values for consideration for this threshold over here. This way, if we have 10 training examples, we will test nine different possible values for this threshold and then try to pick the one that gives we the highest information gain. And finally, if the information gained from splitting on a given value of this threshold is better than the information gain from splitting on any other feature, then we will decide to split that node at that feature. 

And in this example an information gain of 0.61 turns out to be higher than that of any other feature. It turns out they're actually two thresholds. And so assuming the algorithm chooses this feature to split on, we will end up splitting the data set according to whether or not the weight of the animal is less than equal to £9. 

And so we end up with two subsets of the data and we can then build recursively, additional decision trees using these two subsets of the data to build out the rest of the tree. So to summarize to get the decision tree to work on continuous value features at every note. When consuming splits, we would just consider different values to split on, carry out the usual information gain calculation and decide to split on that continuous value feature if it gives the highest possible information gain. 

So that's how we get the decision tree to work with continuous value features. Try different thresholds, do the usual information gain calculation and split on the continuous value feature with the selected threshold if it gives we the best possible information gain out of all possible features to split on. And that's it for the required sections on the core decision tree algorithm After there's there is an optional section we can watch or not that generalizes the decision tree learning algorithm to regression trees. 

So far, we've only talked about using decision trees to make predictions that are classifications predicting a discrete category, such as cat or not cat. But what if we have a regression problem where we want to predict a number in the next section. I'll talk about a generalization of decision trees to handle that. 

## Regression Trees

So far we've only been talking about decision trees as classification algorithms. In this optional section, we'll generalize decision trees to be regression algorithms so that we can predict a number. Let's take a look. 

The example we're going to use for this section will be to use these three valued features that we had previously, that is, these features X, In order to predict the weight of the animal, Y. So just to be clear, the weight here, unlike the previous section is no longer an input feature. Instead, this is the target output, Y, that we want to predict rather than trying to predict whether or not an animal is or is not a cat. 

This is a regression problem because we want to predict a number, Y. Let's look at what a regression tree will look like. Here I've already constructed a tree for this regression problem where the root node splits on ear shape and then the left and right sub tree split on face shape and also face shape here on the right. 

And there's nothing wrong with a decision tree that chooses to split on the same feature in both the left and right side branches. It's perfectly fine if the splitting algorithm chooses to do that. If during training, we had decided on these splits, then this node down here would have these four animals with weights 7.2, 7.6 and 10.2. 

This node would have this one animal with weight 9.2 and so on for these remaining two nodes. So, the last thing we need to fill in for this decision tree is if there's a test example that comes down to this node, what is there weights that we should predict for an animal with pointy ears and a round face shape? The decision tree is going to make a prediction based on taking the average of the weights in the training examples down here. 

And by averaging these four numbers, it turns out we get 8.35. If on the other hand, an animal has pointy ears and a not round face shape, then it will predict 9.2 or 9.2 pounds because that's the weight of this one animal down here. And similarly, this will be 17.70 and 9.90. 

So, what this model will do is given a new test example, follow the decision nodes down as usual until it gets to a leaf node and then predict that value at the leaf node which I had just computed by taking an average of the weights of the animals that during training had gotten down to that same leaf node. So, if we were constructing a decision tree from scratch using this data set in order to predict the weight. The key decision as we've seen earlier this week will be, how do we choose which feature to split on? 

Let me illustrate how to make that decision with an example. At the root node, one thing we could do is split on the ear shape and if we do that, we end up with left and right branches of the tree with five animals on the left and right with the following weights. If we were to choose the split on the face shape, we end up with these animals on the left and right with the corresponding weights that are written below. 

And if we were to choose to split on whiskers being present or absent, we end up with this. So, the question is, given these three possible features to split on at the root node, which one do we want to pick that gives the best predictions for the weight of the animal? When building a regression tree, rather than trying to reduce entropy, which was that measure of impurity that we had for a classification problem, we instead try to reduce the variance of the weight of the values Y at each of these subsets of the data. 

So, if we've seen the notion of variants in other contexts, that's great. This is the statistical mathematical notion of variants that we'll used in a minute. But if we've not seen how to compute the variance of a set of numbers before, don't worry about it. 

All we need to know for this slide is that variants informally computes how widely a set of numbers varies. So for this set of numbers 7.2, 9.2 and so on, up to 10.2, it turns out the variance is 1.47, so it doesn't vary that much. Whereas, here 8.8, 15, 11, 18 and 20, these numbers go all the way from 8.8 all the way up to 20. 

And so the variance is much larger, turns out to the variance of 21.87. And so the way we'll evaluate the quality of the split is, we'll compute same as before, W left and W right as the fraction of examples that went to the left and right branches. And the average variance after the split is going to be 5/10, which is W left times 1.47, which is the variance on the left and then plus 5/10 times the variance on the right, which is 21.87. 

So, this weighted average variance plays a very similar role to the weighted average entropy that we had used when deciding what split to use for a classification problem. And we can then repeat this calculation for the other possible choices of features to split on. Here in the tree in the middle, the variance of these numbers here turns out to be 27.80. 

The variance here is 1.37. And so with W left equals seven-tenths and W right as three-tenths, and so with these values, we can compute the weighted variance as follows. Finally, for the last example, if we were to split on the whiskers feature, this is the variance on the left and right, there's W left and W right. 

And so the weight of variance is this. A good way to choose a split would be to just choose the value of the weighted variance that is lowest. Similar to when we're computing information gain, we're going to make just one more modification to this equation. 

Just as for the classification problem, we didn't just measure the average weighted entropy, we measured the reduction in entropy and that was information gain. For a regression tree we'll also similarly measure the reduction in variance. Turns out, if we look at all of the examples in the training set, all ten examples and compute the variance of all of them, the variance of all the examples turns out to be 20.51. 

And that's the same value for the roots node in all of these, of course, because it's the same ten examples at the roots node. And so what we'll actually compute is the variance of the roots node, which is 20.51 minus this expression down here, which turns out to be equal to 8.84. And so at the roots node, the variance was 20.51 and after splitting on ear shape, the average weighted variance at these two nodes is 8.84 lower. 

So, the reduction in variance is 8.84. And similarly, if we compute the expression for reduction in variance for this example in the middle, it's 20.51 minus this expression that we had before, which turns out to be equal to 0.64. So, this is a very small reduction in variance. 

And for the whiskers feature we end up with this which is 6.22. So, between all three of these examples, 8.84 gives we the largest reduction in variance. So, just as previously we would choose the feature that gives we the largest information gain for a regression tree, we will choose the feature that gives we the largest reduction in variance, which is why we choose ear shape as the feature to split on. 

Having chosen the ear shape feature to split on, we now have two subsets of five examples in the left and right side branches and we would then, again, we say recursively, where we take these five examples and do a new decision tree focusing on just these five examples, again, evaluating different options of features to split on and picking the one that gives we the biggest variance reduction. And similarly on the right. And we keep on splitting until we meet the criteria for not splitting any further. 

And so that's it. With this technique, we can get our decision treat to not just carry out classification problems, but also regression problems. So far, we've talked about how to train a single decision tree. 

It turns out if we train a lot of decision trees, we call this an ensemble of decision trees, we can get a much better result. Let's take a look at why and how to do so in the next section.