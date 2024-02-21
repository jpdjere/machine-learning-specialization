# Welcome

![](2024-02-18-22-05-54.png)

- Week 1: Unsupervised Learning
  - Clustering
  - Anomaly detection
- Week 2: Recommender Systems
- Week 3: Reinforcement Learning

# Clustering

## What is clustering? 

What is clustering? **A clustering algorithm looks at a number of data points and automatically finds data points that are related or similar to each other.** What does that mean?

Let's contrast **clustering**, which is an **unsupervised learning algorithm**, with what we had previously seen with supervised learning for binary classification:

Given a dataset like this with features $x_1$ and $x_2$:

![](2024-02-19-22-14-18.png)

With supervised learning, we had a training set with both the input features $x$ as well as the labels $y$. We could therefore plot a dataset like the above and fit, say, a logistic regression algorithm or a neural network to learn a decision boundary like so:

![](2024-02-19-22-15-05.png)

And, in supervised learning, the dataset included both the inputs $x$ as well as the target outputs $y$. 

In contrast, in unsupervised learning, we are given a dataset like this:

![](2024-02-19-22-15-46.png)

Notice that we have with just $x$, but not the labels or the target labels $y$. That's why when I plot a dataset, it looks like the above, with just dots rather than two classes denoted by the x's and the o's. 

**Because we don't have target labels $y$, we're not able to tell the algorithm what is the "right answer", $y$ that we wanted to predict.** 

Instead, we're going to ask the algorithm to find something interesting about the data, that is, to **find some interesting structure about this data.** 

The first unsupervised learning algorithm that we will learn about is called a **clustering algorithm**, which looks for one particular type of structure in the data:

![](2024-02-19-22-18-32.png)

Namely, the clustering algorithm looks at a dataset like this and tries to see **if the data can be grouped into clusters, meaning groups of points that are similar to each other**. A clustering algorithm, in this case, might find that this dataset comprises of data from two clusters shown here: 

![](2024-02-19-22-19-41.png)

Here are some applications of clustering:

- grouping similar news articles together
- market segmentation
- analyze DNA data, where we will look at the genetic expression data from different individuals and try to group them into people that exhibit similar traits. 
- astronomical data analysis: group bodies in space together to figure out which ones form one galaxy or which one form coherent structures in space

## K-means intuition

Let's see what the **K-means clustering algorithm** does. 

Let's start with the example of a dataset with 30 unlabeled training examples"

![](2024-02-19-22-23-09.png)

The first thing that the K-means algorithm does is: it will take a random guess at where the centers of the two clusters that we might ask it to find might be.

In this example we're going to ask it to try to find two clusters. (Later we'll talk about how we might decide how many clusters to find) 

But the very first step is it will randomly pick two points, which we see here as a red cross and the blue cross, at where might be the centers of two different clusters. Note that this is just a random initial guess and they're not particularly good guesses: 

![](2024-02-19-22-28-30.png)

**K-means will repeatedly do two different things:**

**First, it will assign points to cluster centroids and, secondly, it will move cluster centroids to the average of the new clusters.**

Let's take a look at each step. The first of the two steps is: the algorithm will go through each of these points and look at whether it is closer to centers of the clusters, called cluster centroids, which were provided by initial guesses.

![](2024-02-19-22-33-15.png)

For each of them it will check if it is closer to the red cluster centroid, or if it's closer to the blue cluster centroid. **And it will assign each of these points to whichever of the cluster centroids tt is closer to**:

![](2024-02-19-22-33-33.png)

The **second** of the two steps that K-means does is: it will take all the data points of one cluster and take an average of them. And it will move the cluster centroid to whatever the average location of the dots in that cluster is:

![](2024-02-19-22-35-35.png)
![](2024-02-19-22-35-42.png)

But now that we have these new and hopefully slightly improved guesses for the locations of the to cluster centroids, the algorithm will go through all of the 30 training examples again, and check for every one of them, whether it's closer to the new red or the blue cluster centroids. And we will reassign them to a cluster based on which cluster centroid is the nearset.

![](2024-02-19-22-39-22.png)

So if we go through and associate each point with the closer cluster centroids, we end up with this:

![](2024-02-19-22-39-44.png)

And then we just repeat the second part of K-means again, which is look at all of the data points in a cluster, compute the average and move the cluster centroids:

![](2024-02-19-22-40-48.png)

If we were to keep on repeating these two steps, we find that we reach a point in which there are no more changes to the colors of the points or to the locations of the clusters centroids. This means that at this point **the K-means clustering algorithm has converged:**

![](2024-02-19-22-42-04.png)

In this example, it looks like K-means has done a pretty good job and it has found two clusters.

## K-means algorithm

In the last section, we saw an illustration of the k-means algorithm running. Now let's write out the K-means algorithm in detail.

Here's the K-means algorithm: 

1. Randomly initialize $K$ cluster centroids, $\mu_1, \mu_2, ..., \mu_k$

![](2024-02-19-22-44-24.png)

In the example that we had, $K$ was equal to 2and the red cross would be the location of $\mu_1$ and the blue cross would be the location of $\mu_2$. 

**$\mu_1$ and $\mu_2$ are vectors which have the same dimension as our training examples, $x_1$ through say $x_{30}$, in our example.**

![](2024-02-19-22-46-34.png)

Having randomly initialized the K cluster centroids, K-means will then repeatedly carry out the two steps that we saw in the last section:

**2. Repeat:**
  
- **2.1: assign points to cluster centroids**

The first substep is to assign points to clusters centroids: that is, to color each of the points either red or blue, corresponding to assigning them to cluster centroids 1 or 2 when $K$ is equal to 2. 

That means that we're going to:


$$
\begin{align*}
    \text{for } i &= 1 \text{ to } m: \\
    c^{(i)} &:= \text{index (from 1 to K of cluster centroid closest to } x^{(i)}) \\
\end{align*}
$$

This means that, for each datapoint, we're going to set $c^i$ to be equal to the index of the cluster closest to the training example $x^i$, which can be anything from one to $K$ of the cluster centroid. 

Mathematically we can write this out as computing the distance between $x^i$ and $\mu_k$. 

In math, the distance between two points is often written with the **L2 norm:**

$$ \min {|| x^{(i)} - \mu_k ||}^2 $$

What we want to find is the value of $k$ that minimizes this, because that corresponds to the cluster centroid $\mu_k$ that is closest to the training example $x^{(i)}$. 

Then the value of $k$ that minimizes this is what gets set to $c^i$. 

When we implement this algorithm, we find that it's actually  more convenient to minimize the squared distance, because the cluster centroid with the smallest square distance should be the same as the cluster centroid with the smallest distance. 

![](2024-02-19-23-02-13.png)

As a concrete example, this point up here is closer to the red or two cluster centroids 1. If this was training example x^1, we will set c^1 to be equal to 1. 

Whereas this point over here, if this was the 12th training example, this is closer to the second cluster centroids the blue one. We will set this, the corresponding cluster assignment variable to two because it's closer to cluster centroid 2. 

![](2024-02-19-23-03-04.png)

That's the first step of the K-means algorithm, assign points to cluster centroids. 

- **2.1: move cluster centroids**

The second step is to move the cluster centroids. What that means is:

$$
\begin{align*}
    \text{for } k &= 1 \text{ to } K: \\
    \mu_k &:= \text{average (mean) of points assigned to cluster k} \\
\end{align*}
$$


This means: for each of the cluster, we're going to set the cluster centroid location to be updated to be the average (or the mean) of the points assigned to that cluster $k$.

![](2024-02-19-23-20-39.png)

Concretely, what that means is: we'll look at all of the red points, look at their position on the horizontal axis -the value of the first feature $x^1$- and average that out. The, compute the average value on the vertical axis as well, the featrue $x_2$. Those two values gives us the coordinates of the new centroid of the cluster in vector form:

![](2024-02-19-23-22-24.png)

And the same for the blue points:

![](2024-02-19-23-22-59.png)

The mathematical formula would be:

![](2024-02-19-23-24-04.png)

Again, each of these $x$ values are vectors with two numbers in them, or $n$ numbers in them if we have $n$ features. So, $\mu_k$ will also have two numbers in it, or $n$ numbers if we have $n$ features instead of two. 

Now, there is one **corner case** of this algorithm: **what happens if a cluster has zero training examples assigned to it?**. In that case, the second step, we would be trying to compute the average of zero points. 

If that ever happens, the most common thing to do is to just eliminate that cluster. we end up with K minus 1 clusters. 

Or if we really, really need K clusters an alternative would be to just randomly reinitialize that cluster centroid and hope that it gets assigned at least some points next time round. But it's actually more common when running K-means to just eliminate a cluster if no points are assigned to it. Even though I've mainly been describing K-means for clusters that are well separated. 

Clusters that may look like this. Where if we asked her to find three clusters, hopefully they will find these three distinct clusters. It turns out that K-means is also frequently applied to data sets where the clusters are not that well separated. 

For example, if we are a designer and manufacturer of cool t-shirts, and we want to decide, how do I size my small, medium, and large t-shirts. How small should a small be, how large should a large be, and what should a medium-size t-shirt really be? One thing we might do is collect data of people likely to buy our t-shirts based on their heights and weights. 

we find that the height and weight of people tend to vary continuously on the spectrum without some very clear clusters. Nonetheless, if we were to run K-means with say, three clusters centroids, we might find that K-means would group these points into one cluster, these points into a second cluster, and these points into a third cluster. If we're trying to decide exactly how to size our small, medium, and large t-shirts, we might then choose the dimensions of our small t-shirt to try to make it fit these individuals well. 

The medium-size t-shirt to try to fit these individuals well, and the large t-shirt to try to fit these individuals well with potentially the cluster centroids giving we a sense of what is the most representative height and weight that we will want our three t-shirt sizes to fit. This is an example of K-means working just fine and giving a useful results even if the data does not lie in well-separated groups or clusters. That was the K-means clustering algorithm. 

Assign cluster centroids randomly and then repeatedly assign points to cluster centroids and move the cluster centroids. But what this algorithm really doing and do we think this algorithm will converge or they just keep on running forever and never converge. To gain deeper intuition about the K-means algorithm and also see why we might hope this algorithm does converge, let's go on to the next section where we see that K-means is actually trying to optimize a specific cost function. 

Let's take a look at that in the next section. 

## Optimization objective

In the earlier courses, courses one and two of the specialization, we saw a lot of supervised learning algorithms as taking training set posing a cost function. And then using grading descent or some other algorithms to optimize that cost function. 

It turns out that the K-means algorithm that we saw in the last section is also optimizing a specific cost function. Although the optimization algorithm that it uses to optimize that is not gradient descent is actually the algorithm that we already saw in the last section. Let's take a look at what all this means. 

Let's take a look at what is the cost function for K-means, to get started as a reminder this is a notation we've been using whereas CI is the index of the cluster. So CI is some number from one Su K of the index of the cluster to which training example XI is currently assigned and new K is the location of cluster centroid k. Let me introduce one more piece of notation, which is when lower case K equals CI. 

So mu subscript CI is the cluster centroid of the cluster to which example XI has been assigned. So for example, if I were to look at some training example C train example 10 and I were to ask What's the location of the clustering centroids to which the 10th training example has been assigned? Well, I would then look up C10. 

This will give me a number from one to K. That tells me was example 10 assigned to the red or the blue or some other cluster centroid, and then mu subscript C- 10 is the location of the cluster centroid to which extent has been assigned. So armed with this notation, let me now write out the cost function that K means turns out to be minimizing. 

The cost function J, which is a function of C1 through CM. These are all the assignments of points to clusters Androids as well as new one through mu capsule K. These are the locations of all the clusters centroid is defined as this expression on the right. 

It is the average, so one over M some from i equals to m of the squared distance between every training example XI as I goes from one through M it is a square distance between X I. And Nu subscript C high. So this quantity up here, in other words, the cost function good for K is the average squared distance between every training example XI. 

And the location of the cluster centroid to which the training example exile has been assigned. So for this example up here we've been measuring the distance between X10 and mu subscript C10. The cluster centroid to which extent has been assigned and taking the square of that distance and that would be one of the terms over here that we're averaging over. 

And it turns out that what the K means algorithm is doing is trying to find assignments of points of clusters centroid as well as find locations of clusters centroid that minimizes the squared distance. Visually, here's what we saw part way into the run of K means in the earlier section. And at this step the cost function. 

If we were to computer it would be to look at everyone at the blue points and measure these distances and computer square. And then also similarly look at every one of the red points and compute these distances and compute the square. And then the average of the squares of all of these differences for the red and the blue points is the value of the cost function J, at this particular configuration of the parameters for K-means. 

And what they will do on every step is try to update the cluster assignments C1 through C30 in this example. Or update the positions of the cluster centralism, U1 and U2. In order to keep on reducing this cost function J. 

By the way, this cost function J also has a name in the literature is called the distortion function. I don't know that this is a great name. But if we hear someone talk about the key news algorithm and the distortion or the distortion cost function, that's just what this formula J is computing. 

Let's now take a deeper look at the algorithm and why the algorithm is trying to minimize this cost function J. Or why is trying to minimize the distortion here on top of copied over the cost function from the previous slide. It turns out that the first part of K means where we assign points to cluster centroid. 

That turns out to be trying to update C1 through CM. To try to minimize the cost function J as much as possible while holding mu one through mu K fix. And the second step, in contrast where we move the custom centroid, it turns out that is trying to leave C1 through CM fix. 

But to update new one through mu K to try to minimize the cost function or the distortion as much as possible. Let's take a look at why this is the case. During the first step, if we want to choose the values of C1 through CM or save a particular value of Ci to try to minimize this. 

Well, what would make Xi minus mu CI as small as possible? This is the distance or the square distance between a training example XI. And the location of the class is central to which has been assigned. 

So if we want to minimize this distance or the square distance, what we should do is assign XI to the closest cluster centroid. So to take a simplified example, if we have two clusters centroid say close to central is one and two and just a single training example, XI. If we were to sign it to cluster centroid one, this square distance here would be this large distance, well squared. 

And if we were to assign it to cluster centroid 2 then this square distance would be the square of this much smaller distance. So if we want to minimize this term, we will take X I and assign it to the closer centroid, which is exactly what the algorithm is doing up here. So that's why the step where we assign points to a cluster centroid is choosing the values for CI to try to minimize J. 

Without changing, we went through the mu K for now, but just choosing the values of C1 through CM to try to make these terms as small as possible. How about the second step of the K-means algorithm that is to move to clusters centroids? It turns out that choosing mu K to be average and the mean of the points assigned is the choice of these terms mu that will minimize this expression. 

To take a simplified example, say we have a cluster with just two points assigned to it shown as follows. And so with the cluster centroid here, the average of the square distances would be a distance of one here squared plus this distance here, which is 9 squared. And then we take the average of these two numbers. 

And so that turns out to be one half of 1 plus 81, which turns out to be 41. But if we were to take the average of these two points, so 1+ 11/2, that's equal to 6. And if we were to move the cluster centroid over here to middle than the average of these two square distances, turns out to be a distance of five and five here. 

So we end up with one half of 5 squared plus 5 squared, which is equal to 25. And this is a much smaller average squared distance than 41. And in fact, we can play around with the location of this cluster centroid and maybe convince yourself that taking this mean location. 

This average location in the middle of these two training examples, that is really the value that minimizes the square distance. So the fact that the K-means algorithm is optimizing a cost function J means that it is guaranteed to converge, that is on every single iteration. The distortion cost function should go down or stay the same, but if it ever fails to go down or stay the same, in the worst case, if it ever goes up. 

That means there's a bug in the code, it should never go up because every single step of K means is setting the value CI and mu K to try to reduce the cost function. Also, if the cost function ever stops going down, that also gives we one way to test if K means has converged. Once there's a single iteration where it stays the same. 

That usually means K means has converged and we should just stop running the algorithm even further or in some rare cases we will run K means for a long time. And the cost function of the distortion is just going down very, very slowly, and that's a bit like gradient descent where maybe running even longer might help a bit. But if the rate at which the cost function is going down has become very, very slow. 

we might also just say this is good enough. we're just going to say it's close enough to convergence and not spend even more compute cycles running the algorithm for even longer. So these are some of the ways that computing the cost function is helpful helps we figure out if the algorithm has converged. 

It turns out that there's one other very useful way to take advantage of the cost function, which is to use multiple different random initialization of the cluster centroid. It turns out if we do this, we can often find much better clusters using K means, let's take a look at the next section of how to do that. 

## Initilizing K-means

The very first step of the K means clustering algorithm, was to choose random locations as the initial guesses for the cluster centroids mu one through mu K. 

But how do we actually take that random guess. Let's take a look at that in this section, as well as how we can take multiple attempts at the initial guesses with mu one through mu K. That will result in our finding a better set of clusters. 

Let's take a look, here again is the K means algorithm and in this section let's take a look at how we can implement this first step. When running K means we should pretty much always choose the number of cluster centroids K to be lessened to training examples m. It doesn't really make sense to have K greater than m because then there won't even be enough training examples to have at least one training example per cluster centroids. 

So in our earlier example we had K equals two and m equals 30. In order to choose the cluster centroids, the most common way is to randomly pick K training examples. So here is a training set where if I were to randomly pick two training examples, maybe I end up picking this one and this one. 

And then we would set new one through mu K equal to these K training examples. So I might initialize my red cluster centroid here, and initialize my blue cluster centroids over here, in the example where K was equal to two. And it turns out that if this was our random initialization and we were to run K means we pray end up with K means deciding that these are the two classes in the data set. 

Notes that this method of initializing the cost of central is a little bit different than what I had used in the illustration in the earlier sections. Where I was initializing the cluster centroids mu one and mu two to be just random points rather than sitting on top of specific training examples. I've done that to make the illustrations clearer in the earlier sections. 

But what we're showing in this slide is actually a much more commonly used way of initializing the clusters centroids. Now with this method there is a chance that we end up with an initialization of the cluster centroids where the red cross is here and maybe the blue cross is here. And depending on how we choose the random initial central centroids K-means will end up picking a difference set of causes for our data set. 

Let's look at a slightly more complex example, where we're going to look at this data set and try to find three clusters so k equals three in this data. If we were to run K means with one random initialization of the cluster centroid, we may get this result up here and this looks like a pretty good choice. Pretty good clustering of the data into three different clusters. 

But with a different initialization, say we had happened to initialize two of the cluster centroids within this group of points. And one within this group of points, after running k means we might end up with this clustering, which doesn't look as good. And this turns out to be a local optima, in which K-means is trying to minimize the distortion cost function, that cost function J of C one through CM and mu one through mu K that we saw in the last section. 

But with this less fortunate choice of random initialization, it had just happened to get stuck in a local minimum. And here's another example of a local minimum, where a different random initialization course came in to find this clustering of the data into three clusters, which again doesn't seem as good as the one that we saw up here on top. So if we want to give k means multiple shots at finding the best local optimum. 

If we want to try multiple random initialization, so give it a better chance of finding this good clustering up on top. One other thing we could do with the K-means algorithm is to run it multiple times and then to try to find the best local optima. And it turns out that if we were to run k means three times say, and end up with these three distinct clusterings. 

Then one way to choose between these three solutions, is to compute the cost function J for all three of these solutions, all three of these choices of clusters found by k means. And then to pick one of these three according to which one of them gives we the lowest value for the cost function J. And in fact, if we look at this grouping of clusters up here, this green cross has relatively small square distances, all the green dots. 

The red cross is relatively small distance and red dots and similarly the blue cross. And so the cost function J will be relatively small for this example on top. But here, the blue cross has larger distances to all of the blue dots. 

And here the red cross has larger distances to all of the red dots, which is why the cost function J, for these examples down below would be larger. Which is why if we pick from these three options, the one with the smallest distortion of the smallest cost function J. we end up selecting this choice of the cluster centroids. 

So let me write this out more formally into an algorithm, and wish we would run K-means multiple times using different random initialization. Here's the algorithm, if we want to use 100 random initialization for K-means, then we would run 100 times randomly initialized K-means using the method that we saw earlier in this section. Pick K training examples and let the cluster centroids initially be the locations of those K training examples. 

Using that random initialization, run the K-means algorithm to convergence. And that will give we a choice of cluster assignments and cluster centroids. And then finally, we would compute the distortion compute the cost function as follows. 

After doing this, say 100 times, we would finally pick the set of clusters, that gave the lowest cost. And it turns out that if we do this will often give we a much better set of clusters, with a much lower distortion function than if we were to run K means only a single time. I plugged in the number up here as 100. 

When we're using this method, doing this somewhere between say 50 to 1000 times would be pretty common. Where, if we run this procedure a lot more than 1000 times, it tends to get computational expensive. And we tend to have diminishing returns when we run it a lot of times. 

Whereas trying at least maybe 50 or 100 random initializations, will often give we a much better result than if we only had one shot at picking a good random initialization. But with this technique we are much more likely to end up with this good choice of clusters on top. And these less superior local minima down at the bottom. 

So that's it, when we're using the K means algorithm myself, I will almost always use more than one random initialization. Because it just causes K means to do a much better job minimizing the distortion cost function and finding a much better choice for the cluster centroids. Before we wrap up our discussion of K means, there's just one more section in which I hope to discuss with we. 

The question of how do we choose the number of clusters centroids? How do we choose the value of K? Let's go on to the next section to take a look at that. 

## Choosing the number of clusters

The k-means algorithm requires as one of its inputs, k, the number of clusters we want it to find, but how do we decide how many clusters to used. Do we want two clusters or three clusters of five clusters or 10 clusters? Let's take a look. 

For a lot of clustering problems, the right value of K is truly ambiguous. If I were to show different people the same data set and ask, how many clusters do we see? There will definitely be people that will say, it looks like there are two distinct clusters and they will be right. 

There would also be others that will see actually four distinct clusters. They would also be right. Because clustering is unsupervised learning algorithm we're not given the quote right answers in the form of specific labels to try to replicate. 

There are lots of applications where the data itself does not give a clear indicator for how many clusters there are in it. I think it truly is ambiguous if this data has two or four, or maybe three clusters. If we take say, the red one here and the two blue ones here say. 

If we look at the academic literature on K-means, there are a few techniques to try to automatically choose the number of clusters to use for a certain application. I'll briefly mention one here that we may see others refer to, although I had to say, I personally do not use this method myself. But one way to try to choose the value of K is called the elbow method and what that does is we would run K-means with a variety of values of K and plot the cost function or the distortion function J as a function of the number of clusters. 

What we find is that when we have very few clusters, say one cluster, the distortion function or the cost function J will be high and as we increase the number of clusters, it will go down, maybe as follows. and if the curve looks like this, we say, well, it looks like the cost function is decreasing rapidly until we get to three clusters but the decrease is more slowly after that. Let's choose K equals 3 and this is called an elbow, by the way, because think of it as analogous to that's our hand and that's our elbow over here. 

Plotting the cost function as a function of K could help, it could help we gain some insight. I personally hardly ever use the the elbow method myself to choose the right number of clusters because I think for a lot of applications, the right number of clusters is truly ambiguous and we find that a lot of cost functions look like this with just decreases smoothly and it doesn't have a clear elbow by wish we could use to pick the value of K. By the way, one technique that does not work is to choose K so as to minimize the cost function J because doing so would cause we to almost always just choose the largest possible value of K because having more clusters will pretty much always reduce the cost function J. 

Choosing K to minimize the cost function J is not a good technique. How do we choose the value of K and practice? Often we're running K-means in order to get clusters to use for some later or some downstream purpose. 

That is, we're going to take the clusters and do something with those clusters. What I usually do and what I recommend we do is to evaluate K-means based on how well it performs for that later downstream purpose. Let me illustrate to the example of t-shirt sizing. 

One thing we could do is run K-means on this data set to find the clusters, in which case we may find clusters like that and this would be how we size our small, medium, and large t-shirts, but how many t-shirt sizes should there be? Well, it's ambiguous. If we were to also run K-means with five clusters, we might get clusters that look like this. 

This will let shoe size t-shirts according to extra small, small, medium, large, and extra large. Both of these are completely valid and completely fine groupings of the data into clusters, but whether we want to use three clusters or five clusters can now be decided based on what makes sense for our t-shirt business. Does a trade-off between how well the t-shirts will fit, depending on whether we have three sizes or five sizes, but there will be extra costs as well associated with manufacturing and shipping five types of t-shirts instead of three different types of t-shirts. 

What I would do in this case is to run K-means with K equals 3 and K equals 5 and then look at these two solutions to see based on the trade-off between fits of t-shirts with more sizes, results in better fit versus the extra cost of making more t-shirts where making fewer t-shirts is simpler and less expensive to try to decide what makes sense for the t-shirt business. When we get to the programming exercise, we also see there an application of K-means to image compression. This is actually one of the most fun visual examples of K-means and there we see that there'll be a trade-off between the quality of the compressed image,

that is, how good the image looks versus how much we can compress the image to save the space. 

In that program exercise, we see that we can use that trade-off to maybe manually decide what's the best value of K based on how good do we want the image to look versus how large we want the compress image size to be. That's it for the K-means clustering algorithm. Congrats on learning our first unsupervised learning algorithm. 

we now know not just how to do supervised learning, but also unsupervised learning. I hope we also have fun with the practice lab, is actually one of the most fun exercises I know of the K-means. With that, we're ready to move on to our second unsupervised learning algorithm, which is anomaly detection. 

How do we look at the data set and find unusual or anomalous things in it. This turns out to be another, one of the most commercially important applications of unsupervised learning. I've used this myself many times in many different applications. 

Let's go on to the next section to talk about anomaly detection.