# Welcome

![](./img/2024-02-18-22-05-54.png)

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

![](./img/2024-02-19-22-14-18.png)

With supervised learning, we had a training set with both the input features $x$ as well as the labels $y$. We could therefore plot a dataset like the above and fit, say, a logistic regression algorithm or a neural network to learn a decision boundary like so:

![](./img/2024-02-19-22-15-05.png)

And, in supervised learning, the dataset included both the inputs $x$ as well as the target outputs $y$. 

In contrast, in unsupervised learning, we are given a dataset like this:

![](./img/2024-02-19-22-15-46.png)

Notice that we have with just $x$, but not the labels or the target labels $y$. That's why when I plot a dataset, it looks like the above, with just dots rather than two classes denoted by the x's and the o's. 

**Because we don't have target labels $y$, we're not able to tell the algorithm what is the "right answer", $y$ that we wanted to predict.** 

Instead, we're going to ask the algorithm to find something interesting about the data, that is, to **find some interesting structure about this data.** 

The first unsupervised learning algorithm that we will learn about is called a **clustering algorithm**, which looks for one particular type of structure in the data:

![](./img/2024-02-19-22-18-32.png)

Namely, the clustering algorithm looks at a dataset like this and tries to see **if the data can be grouped into clusters, meaning groups of points that are similar to each other**. A clustering algorithm, in this case, might find that this dataset comprises of data from two clusters shown here: 

![](./img/2024-02-19-22-19-41.png)

Here are some applications of clustering:

- grouping similar news articles together
- market segmentation
- analyze DNA data, where we will look at the genetic expression data from different individuals and try to group them into people that exhibit similar traits. 
- astronomical data analysis: group bodies in space together to figure out which ones form one galaxy or which one form coherent structures in space

## K-means intuition

Let's see what the **K-means clustering algorithm** does. 

Let's start with the example of a dataset with 30 unlabeled training examples"

![](./img/2024-02-19-22-23-09.png)

The first thing that the K-means algorithm does is: it will take a random guess at where the centers of the two clusters that we might ask it to find might be.

In this example we're going to ask it to try to find two clusters. (Later we'll talk about how we might decide how many clusters to find) 

But the very first step is it will randomly pick two points, which we see here as a red cross and the blue cross, at where might be the centers of two different clusters. Note that this is just a random initial guess and they're not particularly good guesses: 

![](./img/2024-02-19-22-28-30.png)

**K-means will repeatedly do two different things:**

**First, it will assign points to cluster centroids and, secondly, it will move cluster centroids to the average of the new clusters.**

Let's take a look at each step. The first of the two steps is: the algorithm will go through each of these points and look at whether it is closer to centers of the clusters, called cluster centroids, which were provided by initial guesses.

![](./img/2024-02-19-22-33-15.png)

For each of them it will check if it is closer to the red cluster centroid, or if it's closer to the blue cluster centroid. **And it will assign each of these points to whichever of the cluster centroids tt is closer to**:

![](./img/2024-02-19-22-33-33.png)

The **second** of the two steps that K-means does is: it will take all the data points of one cluster and take an average of them. And it will move the cluster centroid to whatever the average location of the dots in that cluster is:

![](./img/2024-02-19-22-35-35.png)
![](./img/2024-02-19-22-35-42.png)

But now that we have these new and hopefully slightly improved guesses for the locations of the to cluster centroids, the algorithm will go through all of the 30 training examples again, and check for every one of them, whether it's closer to the new red or the blue cluster centroids. And we will reassign them to a cluster based on which cluster centroid is the nearset.

![](./img/2024-02-19-22-39-22.png)

So if we go through and associate each point with the closer cluster centroids, we end up with this:

![](./img/2024-02-19-22-39-44.png)

And then we just repeat the second part of K-means again, which is look at all of the data points in a cluster, compute the average and move the cluster centroids:

![](./img/2024-02-19-22-40-48.png)

If we were to keep on repeating these two steps, we find that we reach a point in which there are no more changes to the colors of the points or to the locations of the clusters centroids. This means that at this point **the K-means clustering algorithm has converged:**

![](./img/2024-02-19-22-42-04.png)

In this example, it looks like K-means has done a pretty good job and it has found two clusters.

## K-means algorithm

In the last section, we saw an illustration of the k-means algorithm running. Now let's write out the K-means algorithm in detail.

Here's the K-means algorithm: 

1. Randomly initialize $K$ cluster centroids, $\mu_1, \mu_2, ..., \mu_k$

![](./img/2024-02-19-22-44-24.png)

In the example that we had, $K$ was equal to 2and the red cross would be the location of $\mu_1$ and the blue cross would be the location of $\mu_2$. 

**$\mu_1$ and $\mu_2$ are vectors which have the same dimension as our training examples, $x_1$ through say $x_{30}$, in our example.**

![](./img/2024-02-19-22-46-34.png)

Having randomly initialized the $K$ cluster centroids, K-means will then repeatedly carry out the two steps that we saw in the last section:

**2. Repeat:**
  
- **2.1: assign points to cluster centroids**

The first substep is to assign points to clusters centroids: that is, to color each of the points either red or blue, corresponding to assigning them to cluster centroids 1 or 2 when $K$ is equal to 2. 

That means that we're going to:


$$
\begin{align*}
    \text{for } i &= 1 \text{ to } m: \\
    c^{(i)} &:= \text{index (from 1 to $K$ of cluster centroid closest to } x^{(i)}) \\
\end{align*}
$$

This means that, for each datapoint, we're going to set $c^i$ to be equal to the index of the cluster closest to the training example $x^i$, which can be anything from one to $K$ of the cluster centroid. 

Mathematically we can write this out as computing the distance between $x^i$ and $\mu_k$. 

In math, the distance between two points is often written with the **L2 norm:**

$$ \min {|| x^{(i)} - \mu_k ||}^2 $$

What we want to find is the value of $k$ that minimizes this, because that corresponds to the cluster centroid $\mu_k$ that is closest to the training example $x^{(i)}$. 

Then the value of $k$ that minimizes this is what gets set to $c^i$. 

When we implement this algorithm, we find that it's actually  more convenient to minimize the squared distance, because the cluster centroid with the smallest square distance should be the same as the cluster centroid with the smallest distance. 

![](./img/2024-02-19-23-02-13.png)

As a concrete example, this point up here is closer to the red or two cluster centroids 1. If this was training example x^1, we will set c^1 to be equal to 1. 

Whereas this point over here, if this was the 12th training example, this is closer to the second cluster centroids the blue one. We will set this, the corresponding cluster assignment variable to two because it's closer to cluster centroid 2. 

![](./img/2024-02-19-23-03-04.png)

That's the first step of the K-means algorithm, assign points to cluster centroids. 

- **2.1: move cluster centroids**

The second step is to move the cluster centroids. What that means is:

$$
\begin{align*}
    \text{for } $K$ &= 1 \text{ to } K: \\
    \mu_k &:= \text{average (mean) of points assigned to cluster k} \\
\end{align*}
$$


This means: for each of the cluster, we're going to set the cluster centroid location to be updated to be the average (or the mean) of the points assigned to that cluster $k$.

![](./img/2024-02-19-23-20-39.png)

Concretely, what that means is: we'll look at all of the red points, look at their position on the horizontal axis -the value of the first feature $x^1$- and average that out. The, compute the average value on the vertical axis as well, the featrue $x_2$. Those two values gives us the coordinates of the new centroid of the cluster in vector form:

![](./img/2024-02-19-23-22-24.png)

And the same for the blue points:

![](./img/2024-02-19-23-22-59.png)

The mathematical formula would be:

![](./img/2024-02-19-23-24-04.png)

Again, each of these $x$ values are vectors with two numbers in them, or $n$ numbers in them if we have $n$ features. So, $\mu_k$ will also have two numbers in it, or $n$ numbers if we have $n$ features instead of two. 

Now, there is one **corner case** of this algorithm: **what happens if a cluster has zero training examples assigned to it?**. In that case, the second step, we would be trying to compute the average of zero points. 

If that ever happens, **the most common thing to do is to just eliminate that cluster. We end up with $K - 1$ clusters.**

**If really need $K$ clusters, an alternative is to just randomly reinitialize that cluster centroid and hope that it gets assigned at least some points next time round.** 

---
K-means is also frequently applied to data sets where the clusters are not that well separated, like the plot on the right:

![](./img/2024-02-19-23-27-58.png)

For example, we are a designer and manufacturer of T-shirts, and we want to decide how to size my small, medium, and large t-shirts. How small should a small be, how large should a large be, and what should a medium-size t-shirt really be? 

One thing we might do is collect data of people likely to buy our t-shirts based on their heights and weights. We find that the height and weight of people tend to vary continuously on the spectrum without some very clear clusters. 

Nonetheless, if we were to run K-means with three clusters centroids, we might find that K-means would group the points as follows:

![](./img/2024-02-19-23-29-52.png)

If we're trying to decide exactly how to size our small, medium, and large t-shirts, we might then choose the dimensions of each size to try to fit the datapoints in each cluster. **Potentially, the cluster centroids might provide a sense of what is the most representative height and weight that we will want our three t-shirt sizes to fit.** 

This is an example of K-means working just fine and giving a useful results even if the data does not lie in well-separated groups or clusters.

## Optimization objective

We have seen a lot of supervised learning algorithms that take atraining set, present a cost function, and then, using grading descent or some other algorithm, they attempt to optimize that cost function. 

It turns out that **the K-means algorithm is also optimizing a specific cost function**, although the optimization algorithm that it uses to optimize that is not gradient descent; it is actually the algorithm that we saw in the last section. Let's take a look at what all this means. 

Let's take a look at what is the cost function for K-means: to get started, a reminder of the notation until now:

$$c^{(i)} = \text{index of cluster (1, 2, ... K) to which example} \space x^{(i)}\space \text{is currently assigned} $$ $$ \mu_k = \text{cluster centroid} \space $K$ $$

And let's introduce one more piece of notation:

$$ \mu_{c^{(i)}} = \text{cluster centroid of cluster to which example }x^{(i)}\space \text{has benn assigned}$$

So for example, if I were to look at some training example $x^{(10)}$ and I were to ask: what's the location of the clustering centroidsto which that training example has been assigned? 

Well, I would then look up $c^{(10)}$ and that would give me a number from $1$ to $K$, which tells me to which cluster centroid my data point was assigned to. And then $\mu_{c^{10}}$ would be the location of the cluster centroid to which $x^{(10)}$ was assigned to. 

Armed with this notation, let's now write out **the cost function that K-means minimizes:**

$$ J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_{K}) = \frac{1}{m} \sum_{i = 1}^m {|| x^{(i)} - \mu_{c^{(i)}} ||}^2 $$

- $c^{(1)}$ to $c^{(m)}$: all the assignments of points to clusters centroids
- $\mu_{c^{(1)}}$ to $\mu_{c^{(k)}}$ : the locations of all the clusters centroids

So, the cost function is the average of the squared distance between every training example $x^{(i)}$ (as $i$ goes from $1$ through $m$) and $\mu_{c^{(i)}}$. 

In other words, **the cost function for K-means is the average squared distance between every training example andd the location of the cluster centroid to which the training example has been assigned.**

What the K-means algorithm is doing is trying to find are two things:

**1. assignments of points of clusters centroid** 
**2. locations of clusters centroid**

which in combination that minimize the squared distance. 

Visually, here's what the cost function is trying to minimize. At each step:

![](./img/2024-02-21-23-58-00.png)

the algorithm is taking the distances of the red points to the red cluster centroid and calculating its squares, doing the same with the blue points and then averaging all of them, which is the value of the cost function $J$, at this particular configuration of the parameters for K-means. 

And the, what the algorithm they will do on every step is try to update the cluster assignments $c^{(i)}$, or update the positions of the cluster centroids $\mu_{c^{(i)}}$, in order to keep on reducing this cost function J. 

**This cost function $J$ also has the name of distortion function**.

Let's now take a deeper look at the algorithm and tru to understand why the algorithm is trying to minimize this cost function $J$:

![](./img/2024-02-22-00-03-48.png)

- In the **first part of K-means**, where we assign points to cluster centroids, that is actually trying to update $c^{(1)}$ through $c^{(im)}$ in order to try to minimize the cost function $J$ while holding $\mu_{1}$ through $\mu_{K}$ fixed. 

- **And the second step**, where we move the custom centroids, that is trying to update $\mu_{1}$ through $\mu_{K}$, while leaving $c^{(1)}$ through $c^{(im)}$ fixed.

Going back to the first step, if we want to minimize this distance or the square distance, what we should do is assign $x^{(i)}$ to the closest cluster centroid. So to take a simplified example:
![](./img/2024-02-22-00-09-26.png)

We have the two clusters centroids above and just a single training example, $x^{(i)}$. If we were to assign that data point to the blue cluster, this square distance betwen the data point and the blue cluster centroid would be much larger that between the data point and the red cluster centroid.

So if we want to minimize the term, we will take $x^{(i)}$ and assign it to the closer centroid, the blue centroid. And that's what the algorithm is doing, without changing the values for $\mu$ which are the location of the cluster centroids themselves. 

How about the second step of the K-means algorithm, where we move the clusters centroids? It turns out that **choosing $\mu_k$, the location of the centroid, to be average of the points assigned to that cluster is the best choice, that will minimize this expression.**

To take a simplified example, say we have a cluster with just two points assigned to it shown as follows: 
![](./img/2024-02-22-00-15-30.png)

With the cluster centroid on the left, the average of the square distances would turn out to be 41. 

But if we were to take the average of these two points, so $(1 + 11) / 2$, that's equal to 6. And if we were to move the cluster centroid to the average of these two square distances, a distance of 5 and 5, we end up with a squared distance of 25. And this is a much smaller average squared distance than 41.

**The fact that the K-means algorithm is optimizing a cost function $J$ means that it is guaranteed to converge**, that is, on every single iteration, the distortion cost function should go down or stay the same.

If the cost function stops going down, once there's a single iteration where it stays the same,that gives an indications that Komeans has converged and we should just stop running the algorithm.

In some rare cases we will run K-means for a long time and the cost function of the distortion will go down very, very slowly: that's similar to might what happen with gradient descent, where maybe running even longer might help, but if the rate at which the cost function is going down has become very slow, we might also decide we have reached a good-enough state, close enough to the point of convergence.

## Initilizing K-means

The first step of the K-means clustering algorithm **was to choose random locations as the initial guesses for the cluster centroids**, $\mu_{1}$ through $\mu_{K}$. 

But how do we actually take that random gues?. Let's take a look at that, as well as how we can **take multiple attempts at the initial guesses** for $\mu_{1}$ through $\mu_{K}$, which **will result in our finding a better set of clusters.**

Here again is the K-means algorithm: let's take a look at how we can implement this first step, step 0: 

![](./img/2024-02-22-10-07-57.png)

When running K-means we should **always choose the number of cluster centroids $K$ to be less than the amount of training examples $m$**: it doesn't really make sense to have $K$ greater than $m$ because then there won't even be enough training examples to have at least one training example per cluster centroid:

![](./img/2024-02-22-10-09-58.png)

In our earlier example we had $K$ equals 2 and *m* equals 30. In order to choose the cluster centroids, the most common way is to **randomly pick $K$ training examples and set $\mu_{1}$ through $\mu_{K}$ in those locations**. For example, in the following training, I would randomly pick two training examples and initialize my red cluster centroid and my blue cluster centroids  here, in the example where $K$ was equal to 2:

![](./img/2024-02-22-10-13-39.png)

If this was our random initialization and we were to run K-means we pray end up with K-means deciding that these are the two classes in the data set:

![](./img/2024-02-22-10-13-56.png)

Notes that this method of initializing the cost of central is a little bit different than what I had used in the illustration in the earlier sections. Where I was initializing the cluster centroids mu one and mu two to be just random points rather than sitting on top of specific training examples. That was only to make the illustrations clearer in the earlier sections.But ,initialization by choosing random datapoints is actually a much more commonly used way of initializing the clusters centroids. 

However, with this method **there is a chance that, depending on how we choose the random initial central centroids, K-means will end up picking a difference set of clusters for our data set.**

Let's look at a slightly more complex example, where we're going to look at this data set and try to find three clusters, so $K$ equals 3:

![](./img/2024-02-22-10-17-18.png)

If we were to run K-means with random initialization of the cluster centroid, we may get the result in the top, which looks like a pretty good choice. 

But with a different initialization, we might end up with the other two alternatives seen below, which **are not fully optimized: with this less fortunate choice of random initialization, ithe algorithm gets stuck in a local minimum.**

So, **if we want to give K-means multiple shots at finding the best local optimum**, by trying  multiple random initialization, **we can run the algorithm multiple times and then to try to find the best local optima**. 

If we were to run K-means 3 times, for example, and end up with these three distinct clusterings, one way to choose between these three solutions is to **compute the cost function $J$ for all three solutions**, all three of these choices of clusters found by K-means, and get $J_1$, $J_2$ and $J_3$. **And then to pick one of these three according to which one of them gives we the lowest value for the cost function**.

You can see this graphically, by the distance of each of the dots assigned to each cluster to each cluster centroid, and how it would end up being much less on the top graph:

![](./img/2024-02-22-10-24-42.png)

Let's more formally into an algorithm: 

![](./img/2024-02-22-10-27-08.png)

- If we want to use 100 random initialization for K-means, then we would run 100 times randomly initialized K-means using the method that we saw earlier in this section. 

- Pick $K$ training examples and let the cluster centroids initially be the locations of those $K$ training examples. 

- Using that random initialization, run the K-means algorithm to convergence. And that will give us a choice of cluster assignments and cluster centroids.

- Compute the distortion compute the cost function $J$.

- After doing this, say 100 times, we would finally pick the set of clusters that gave the lowest cost. 

When we're using this method, **repeating the process somewhere between 50 to 1000 times would be pretty common**. After 1000 times, it tends to get computational expensive and have diminishing returns.

## Choosing the number of clusters

The k-means algorithm requires as one of its inputs, $K$, **the number of clusters we want it to find**. But **how do we decide how many clusters to use?**

**For a lot of clustering problems, the right value of $K$ is truly ambiguous.** If we show different people the same data set and ask, how many clusters do you see? There will definitely be people that will say there are two distinct clusters:

![](./img/2024-02-22-23-58-25.png)

There would also be others that will see actually four distinct clusters:

![](./img/2024-02-22-23-58-44.png)

Both would be right. **Because clustering is an unsupervised learning algorithm, we're not given the "right" answers in the form of specific labels to try to replicate.** 

There are lots of applications where the data itself does not give a clear indicator for how many clusters there are in it.

If we look at the academic literature on K-means, there are a few techniques to try to automatically choose the number of clusters to use for a certain application. 

We'll briefly mention one here that we may see others refer to, although I have to say, I personally do not use this method myself. But **one way to try to choose the value of $K$ is called the "elbow method": run K-means with a variety of values of $K$ and plot the cost function or the distortion function $J$ as a function of the number of clusters.** 

![](./img/2024-02-23-00-02-00.png)

For very few clusters, the distortion function or the cost function $J$ will be high, and as we increase the number of clusters, it will go down, maybe as follows. The cost function decreases rapidly until we get to three clusters but the decrease is more slowly after that. So we choose $K$ equals 3. This is called an elbow, because of the shape of the curve.

I personally hardly ever use the the elbow method myself to choose the right number of clusters because I think for a lot of applications, the right number of clusters is truly ambiguous and we find that a lot of cost functions look just decrease smoothly and don't have a clear elbow by which we can pick the value of K. 

Also: one technique that does not work is to choose $K$ so as to minimize the cost function $J$ because doing so would cause we to almost always just choose the largest possible value of $K$,since having more clusters will pretty much always reduce the cost function $J$. 

**So, how do we choose the value of $K$ in practice?** 

Often we're running K-means in order to get clusters to use for some later or some downstream purpose. That is, we're going to take the clusters and do something with those clusters. 

So, **what we do is to evaluate K-means based on how well it performs for that later downstream purpose**. Let's illustrate that with the example of t-shirt sizing:

![](./img/2024-02-23-00-07-10.png)

One thing we could do is run K-means on this data set to find the clusters, in which case we may find clusters for sizes small, medium, and large t-shirts.

But how many t-shirt sizes should there actually be? It's ambiguous: if we were to also run K-means with five clusters, we might get clusters for sizes extra small, small, medium, large, and extra large. 

**Both of these are completely valid and completely fine groupings of the data into clusters**, but whether we want to use three clusters or five clusters can now be decided based on what makes sense for our t-shirt business. There's a trade-off between how well the t-shirts will fit, between having three sizes or five sizes, with the extra costs associated with manufacturing and shipping five types of t-shirts instead of three.

So, what we can do in this case is to run K-means with $K$ = 3 and $K$ = 5 and then look at these two solutions to see, if the tradeoff is worth it, and what makes more sense from the business point of view.

## Lab: K-means Clustering


In this exercise, you will implement the K-means algorithm and use it for image compression. 

* You will start with a sample dataset that will help you gain an intuition of how the K-means algorithm works. 
* After that, you will use the K-means algorithm for image compression by reducing the number of colors that occur in an image to only those that are most common in that image.

### Outline
- [ 1 - Implementing K-means](#1)
  - [ 1.1 Finding closest centroids](#1.1)
    - [ Exercise 1](#ex01)
  - [ 1.2 Computing centroid means](#1.2)
    - [ Exercise 2](#ex02)
- [ 2 - K-means on a sample dataset ](#2)
- [ 3 - Random initialization](#3)
- [ 4 - Image compression with K-means](#4)
  - [ 4.1 Dataset](#4.1)
  - [ 4.2 K-Means on image pixels](#4.2)
  - [ 4.3 Compress the image](#4.3)

First, run the cell below to import the packages needed in this assignment:

- [numpy](https://numpy.org/) is the fundamental package for scientific computing with Python.
- [matplotlib](http://matplotlib.org) is a popular library to plot graphs in Python.
- `utils.py` contains helper functions for this assignment. You do not need to modify code in this file.

```py
import numpy as np
import matplotlib.pyplot as plt
from utils import *

%matplotlib inline
```

<a name="1"></a>
### 1 - Implementing K-means

The K-means algorithm is a method to automatically cluster similar
data points together. 

* Concretely, you are given a training set $\{x^{(1)}, ..., x^{(m)}\}$, and you want
to group the data into a few cohesive “clusters”. 


* K-means is an iterative procedure that
     * Starts by guessing the initial centroids, and then 
     * Refines this guess by 
         * Repeatedly assigning examples to their closest centroids, and then 
         * Recomputing the centroids based on the assignments.
         

* In pseudocode, the K-means algorithm is as follows:

    ``` python
    # Initialize centroids
    # K is the number of clusters
    centroids = kMeans_init_centroids(X, K)
    
    for iter in range(iterations):
        # Cluster assignment step: 
        # Assign each data point to the closest centroid. 
        # idx[i] corresponds to the index of the centroid 
        # assigned to example i
        idx = find_closest_centroids(X, centroids)

        # Move centroid step: 
        # Compute means based on centroid assignments
        centroids = compute_centroids(X, idx, K)
    ```


* The inner-loop of the algorithm repeatedly carries out two steps: 
    1. Assigning each training example $x^{(i)}$ to its closest centroid, and
    2. Recomputing the mean of each centroid using the points assigned to it. 
    
    
* The $K$-means algorithm will always converge to some final set of means for the centroids. 

* However, the converged solution may not always be ideal and depends on the initial setting of the centroids.
    * Therefore, in practice the K-means algorithm is usually run a few times with different random initializations. 
    * One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion).

You will implement the two phases of the K-means algorithm separately
in the next sections. 
* You will start by completing `find_closest_centroid` and then proceed to complete `compute_centroids`.

<a name="1.1"></a>
### 1.1 Finding closest centroids

In the “cluster assignment” phase of the K-means algorithm, the
algorithm assigns every training example $x^{(i)}$ to its closest
centroid, given the current positions of centroids. 

<a name="ex01"></a>
#### Exercise 1

Your task is to complete the code in `find_closest_centroids`. 
* This function takes the data matrix `X` and the locations of all
centroids inside `centroids` 
* It should output a one-dimensional array `idx` (which has the same number of elements as `X`) that holds the index  of the closest centroid (a value in $\{0,...,K-1\}$, where $K$ is total number of centroids) to every training example . *(Note: The index range 0 to K-1 varies slightly from what is shown in the lectures (i.e. 1 to K) because Python list indices start at 0 instead of 1)*
* Specifically, for every example $x^{(i)}$ we set
$$c^{(i)} := j \quad \mathrm{that \; minimizes} \quad ||x^{(i)} - \mu_j||^2,$$
where 
 * $c^{(i)}$ is the index of the centroid that is closest to $x^{(i)}$ (corresponds to `idx[i]` in the starter code), and 
 * $\mu_j$ is the position (value) of the $j$’th centroid. (stored in `centroids` in the starter code)
 * $||x^{(i)} - \mu_j||$ is the L2-norm
 
If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

```py
# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    
    # Loop through every datapoint
    for i in range(X.shape[0]):
        # Create an array where to store the distance of the current
        # datapoint to all the possible centroids
        distances = []
        
        # Loop thorugh all the centroids
        for centroid in centroids:
            # Calculate the L2 norm from the datapoint to the centroid
            # and append it to the distances array
            dist = np.linalg.norm(X[i]-centroid)**2
            distances.append(dist)
            
        # Now in distances we have all the distances, and the index of
        # each represents also K, the index of the cluster. The value
        # of each element is the distance. So we get the index of the
        # lowest element and assign it to idx.
        idx[i] = np.argmin(distances)
        
     ### END CODE HERE ###
    
    return idx
```

Now let's check your implementation using an example dataset.

```py
# Load an example dataset that we will be using
X = load_data()
```

The code below prints the first five elements in the variable `X` and the dimensions of the variable.

```py
print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)

# First five elements of X are:
#  [[1.84207953 4.6075716 ]
#  [5.65858312 4.79996405]
#  [6.35257892 3.2908545 ]
#  [2.90401653 4.61220411]
#  [3.23197916 4.93989405]]
# The shape of X is: (300, 2)
```

```py
# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])

# UNIT TEST
from public_tests import *

find_closest_centroids_test(find_closest_centroids)

# First three elements in idx are: [0 2 1]
# All tests passed!
```

<a name="1.2"></a>
### 1.2 Computing centroid means

Given assignments of every point to a centroid, the second phase of the
algorithm recomputes, for each centroid, the mean of the points that
were assigned to it.


<a name="ex02"></a>
### Exercise 2

Please complete the `compute_centroids` below to recompute the value for each centroid

* Specifically, for every centroid $\mu_k$ we set
$$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$$ 

    where 
    * $C_k$ is the set of examples that are assigned to centroid $k$
    * $|C_k|$ is the number of examples in the set $C_k$


* Concretely, if two examples say $x^{(3)}$ and $x^{(5)}$ are assigned to centroid $k=2$,
then you should update $\mu_2 = \frac{1}{2}(x^{(3)}+x^{(5)})$.

If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.


**My approach**
```py
# UNQ_C2
# GRADED FUNCTION: compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    num_elements_per_cluster = np.zeros((K))
    
    
    ### START CODE HERE ###
    for index_example, index_centroid in enumerate(idx):
        centroids[index_centroid] += X[index_example]
        num_elements_per_cluster[index_centroid] += 1
        
    for i in range(len(centroids)):
        centroids[i] = centroids[i]/num_elements_per_cluster[i]
        
    ### END CODE HERE ## 
    
    return centroids
```

**Best approach**

```py
# UNQ_C2
# GRADED FUNCTION: compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    num_elements_per_cluster = np.zeros((K))
    
    
    ### START CODE HERE ###
    for k in range(K):
      points = X[idx == k]
      centroids[k] = np.mean(points, axis=0)
        
    ### END CODE HERE ## 
    
    return centroids
```

Now check your implementation by running the cell below:

```py
K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)

# UNIT TEST
compute_centroids_test(compute_centroids)

# The centroids are: [[2.42830111 3.15792418]
#  [5.81350331 2.63365645]
#  [7.11938687 3.6166844 ]]
# All tests passed!
```

<a name="2"></a>
## 2 - K-means on a sample dataset 

After you have completed the two functions (`find_closest_centroids`
and `compute_centroids`) above, the next step is to run the
K-means algorithm on a toy 2D dataset to help you understand how
K-means works. 
* We encourage you to take a look at the function (`run_kMeans`) below to understand how it works. 
* Notice that the code calls the two functions you implemented in a loop.

When you run the code below, it will produce a
visualization that steps through the progress of the algorithm at
each iteration. 
* At the end, your figure should look like the one displayed in Figure 1.
* The final centroids are the black X-marks in the middle of the colored clusters.
* You can see how these centroids got to their final location by looking at the other X-marks connected to it.

![](./img/2024-02-28-14-34-42.png)


**Note**: You do not need to implement anything for this part. Simply run the code provided below

```py
# You do not need to implement anything for this part

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx
```

```py
# Load an example dataset
X = load_data()

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])

# Number of iterations
max_iters = 10

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
```

```py
# K-Means iteration 0/9
# K-Means iteration 1/9
# K-Means iteration 2/9
# K-Means iteration 3/9
# K-Means iteration 4/9
# K-Means iteration 5/9
# K-Means iteration 6/9
# K-Means iteration 7/9
# K-Means iteration 8/9
# K-Means iteration 9/9
```

![](./img/2024-02-28-14-36-14.png)

<a name="3"></a>
## 3 - Random initialization

The initial assignments of centroids for the example dataset was designed so that you will see the same figure as in Figure 1. In practice, a good strategy for initializing the centroids is to select random examples from the
training set.

In this part of the exercise, you should understand how the function `kMeans_init_centroids` is implemented.
* The code first randomly shuffles the indices of the examples (using `np.random.permutation()`). 
* Then, it selects the first $K$ examples based on the random permutation of the indices. 
* This allows the examples to be selected at random without the risk of selecting the same example twice.

**Note**: You do not need to implement anything for this part of the exercise.

```py
# You do not need to modify this part

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids
```

You can run K-Means again but this time with random initial centroids. Run the cell below several times and observe how different clusters are created based on the initial points chosen.

```py
# Run this cell repeatedly to see different outcomes.

# Set number of centroids and max number of iterations
K = 3
max_iters = 10

# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
```

![](./img/2024-02-28-14-38-19.png)

![](./img/2024-02-28-14-38-32.png)

![](./img/2024-02-28-14-38-53.png)

![](./img/2024-02-28-14-39-16.png)

Note that the last initialization resulted in "wrong" clusters.

<a name="4"></a>
## 4 - Image compression with K-means

In this exercise, you will apply K-means to image compression. 

* In a straightforward 24-bit color representation of an image$^{2}$, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often refered to as the RGB encoding.
* Our image contains thousands of colors, and in this part of the exercise, you will reduce the number of
colors to 16 colors.
* By making this reduction, it is possible to represent (compress) the photo in an efficient way. 
* Specifically, you only need to store the RGB values of the 16 selected colors, and for each pixel in the image you now need to only store the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities).

In this part, you will use the K-means algorithm to select the 16 colors that will be used to represent the compressed image.
* Concretely, you will treat every pixel in the original image as a data example and use the K-means algorithm to find the 16 colors that best group (cluster) the pixels in the 3- dimensional RGB space.
* Once you have computed the cluster centroids on the image, you will then use the 16 colors to replace the pixels in the original image.

<img src="images/figure 2.png" width="500" height="500">

$^{2}$<sub>The provided photo used in this exercise belongs to Frank Wouters and is used with his permission.</sub>

<a name="4.1"></a>
### 4.1 Dataset

**Load image**

First, you will use `matplotlib` to read in the original image, as shown below.

```py
# Load an image of a bird
original_img = plt.imread('bird_small.png')
```

**Visualize image**

You can visualize the image that was just loaded using the code below.

```py
# Visualizing the image
plt.imshow(original_img)
```

![](./img/2024-03-02-14-26-51.png)

**Check the dimension of the variable**

As always, you will print out the shape of your variable to get more familiar with the data.

```py
print("Shape of original_img is:", original_img.shape)

# Shape of original_img is: (128, 128, 3)
```

As you can see, this creates a three-dimensional matrix `original_img` where 
* the first two indices identify a pixel position, and
* the third index represents red, green, or blue. 

For example, `original_img[50, 33, 2]` gives the blue intensity of the pixel at row 50 and column 33.

![](./img/2024-03-02-14-30-24.png)

#### Processing data

To call the `run_kMeans`, you need to first transform the matrix `original_img` into a two-dimensional matrix.

* The code below reshapes the matrix `original_img` to create an $m \times 3$ matrix of pixel colors (where
$m=16384 = 128\times128$)

*Note: If you'll try this exercise later on a JPG file, you first need to divide the pixel values by 255 so it will be in the range 0 to 1. This is not necessary for PNG files (e.g. `bird_small.png`) because it is already loaded in the required range (as mentioned in the [plt.imread() documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html)). We commented a line below for this so you can just uncomment it later in case you want to try a different file.* 

```py
# Divide by 255 so that all values are in the range 0 - 1 (not needed for PNG files)
# original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
```

<a name="4.2"></a>
### 4.2 K-Means on image pixels

Now, run the cell below to run K-Means on the pre-processed image.

```py
# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-Means - this can take a couple of minutes depending on K and max_iters
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

# K-Means iteration 0/9
# K-Means iteration 1/9
# K-Means iteration 2/9
# K-Means iteration 3/9
# K-Means iteration 4/9
# K-Means iteration 5/9
# K-Means iteration 6/9
# K-Means iteration 7/9
# K-Means iteration 8/9
# K-Means iteration 9/9
```

```py
print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

# Shape of idx: (16384,)
# Closest centroid for the first five elements: [10 10 10 10 10]
```

The code below will plot all the colors found in the original image. As mentioned earlier, the color of each pixel is represented by RGB values so the plot should have 3 axes -- R, G, and B. You'll notice a lot of dots below representing thousands of colors in the original image. The red markers represent the centroids after running K-means. These will be the 16 colors that you will use to compress the image.

```py
# Plot the colors of the image and mark the centroids
plot_kMeans_RGB(X_img, centroids, idx, K)
```

![](./img/2024-03-02-14-34-58.png)

You can visualize the colors at each of the red markers (i.e. the centroids) above with the function below. You will only see these colors when you generate the new image in the next section. The number below each color is its index and these are the numbers you see in the `idx` array.

```py
# Visualize the 16 colors selected
show_centroid_colors(centroids)
```

![](./img/2024-03-02-14-35-37.png)

<a name="4.3"></a>
### 4.3 Compress the image

After finding the top $K=16$ colors to represent the image, you can now
assign each pixel position to its closest centroid using the
`find_closest_centroids` function. 
* This allows you to represent the original image using the centroid assignments of each pixel. 
* Notice that you have significantly reduced the number of bits that are required to describe the image. 
    * The original image required 24 bits (i.e. 8 bits x 3 channels in RGB encoding) for each one of the $128\times128$ pixel locations, resulting in total size of $128 \times 128 \times 24 = 393,216$ bits. 
    * The new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location. 
    * The final number of bits used is therefore $16 \times 24 + 128 \times 128 \times 4 = 65,920$ bits, which corresponds to compressing the original image by about a factor of 6.

```py
# Find the closest centroid of each pixel
idx = find_closest_centroids(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :] 

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 
```

Finally, you can view the effects of the compression by reconstructing
the image based only on the centroid assignments. 
* Specifically, you replaced each pixel with the value of the centroid assigned to
it. 
* Figure 3 shows a sample reconstruction. Even though the resulting image retains most of the characteristics of the original, you will also see some compression artifacts because of the fewer colors used.

![](./img/2024-03-02-14-37-21.png)

* Run the code below to see how the image is reconstructed using the 16 colors selected earlier.


```py
# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
```

![](./img/2024-03-02-14-38-08.png)