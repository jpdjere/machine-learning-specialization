# Collaborative filtering

## Making recommendations

Every time we go to an online shopping website like Amazon or a movie streaming sites like Netflix or go to one of the apps or sites that do food delivery, these sites will recommend things to us that they think we may want to buy or movies they think we may want to watch or restaurants that they think we may want to try out. 

For many companies, a large fraction of sales is driven by their recommended systems. So today for many companies, the economics or the value driven by recommended systems is very large/

Let's dive in, using as a running example the application of predicting movie ratings. 

In our example, we run a large movie streaming website, where our users can rate movies using 0 to 5 stars.

![](2024-03-10-12-57-19.png)

Notice that we denote via a question mark when a user has not rated a film.

The notation we're going to use is we're going to use is:

- $n_u$: number of users
- $n_m$: number of films
- $r(i,j) = 1$: if user $j$ has rated movie $i$, otherwise $0$
- $y(i,j)$: rating given by user $j$ to movie $i$

![](2024-03-10-13-00-01.png)

So with this framework for recommended systems **one possible way to approach the problem is to look at the movies that users have not rated, and to try to predict how users would rate those movies**. Based on this we can then try to recommend to users things that they are more likely to rate as five stars. 

We'll start to develop an algorithm for doing exactly that, but making one very special assumption: we're going to assume temporarily that we have access to features or extra information about the movies such as which movies are romance movies, which movies are action movies. 

However, later we will actually ask what if we don't have these features, how can we still get the algorithm to work then?

## Using per-item features

So let's take a look at how we can develop a recommender system if we have features of each item (movie). 

Here's the same data set that we had previously with the four users having rated some but not all of the five movies. What if we additionally have features of the movies?

![](2024-03-12-17-21-52.png)

The two features $x_1$ and $x_2$, that tell us how much each of these is a romance movie, and how much each of these is an action movie.

Rcall that we had used the notation $n_u$ to denote the number of users, which is 4 and $m$ to denote the number of movies, which is 5. We'll also introduce $n$ to denote the number of features we have here. And so $n = 2$, because we have two features $x_1$ and $x_2$ for each movie.

![](2024-03-12-17-24-28.png)

With these features we have for example that the features for movie one, "Love at Last", would be $[0.9 \space\space 0]$. And the features for the third movie "Cute Puppies of Love" would be $[0.99 \space\space 0]$. 

And let's start by taking a look at how we might make predictions for Alice's movie ratings. 

So for user one, that is Alice, let's say we predict the rating for movie $i$ as $w \cdot x^{(i)} + b$, which is just like linear regression. For example, if we had a parameter $w = [5 \space\space 0]$ (this is invented), then: 

![](2024-03-12-17-32-57.png)

So for the first movie we would have a rating of 4.95. And this rating seems pretty plausible: Alice has given high ratings to "Love at Last" and Romance Forever, two highly romantic movies, but given low ratings to the action movies, "Nonstop Car Chases" and "Swords vs Karate". So if we look at "Cute Puppies of Love" predicting that she might rate that movie 4.95 seems quite plausible. 

So these parameters $w$ and $b$ for Alice seem like a reasonable model for predicting her movie ratings. 

Just adding a little the notation: because we have not just 1 user but multiple users, or really $n_u$ equals 4 users, we're going to add a superscript 1 to denote that this is the parameter $w^{(1)}$ for user 1 and add a superscript 1 for $b$ as well.

More generally for this model we have user $j$ and we can predict his rating for movie $i$ as: 

$$ \text{Rating for movie} \space i = w^{(j)} \cdot x^{(i)} + b^{(j)} $$


This is a lot like linear regression, except that we're fitting a different linear regression model for each of the 4 users in the dataset. 

Let's take a look at **how we can formulate the cost function for this algorithm**:

As a reminder, our notation is: 

![](2024-03-13-16-19-20.png)

Notice that we introduce a new $m^{(j)}$ to denote the number of movies rated by user $j$.

Therefore given the ratings that a user $j$ has given to the movies they have seen, to learn the parameters $w^{(j)}$ and $b^{(j)}$, we have the cost function:

$$\text{min } J(w^{(j)},b^{(j)}) = \frac{1}{2m^{(j)}} \sum_{i:r(j, i) = 1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 $$

The avobe is using the mean squared error criteria. We're trying to choose parameters $w$ and $b$ to minimize the squared error between the predicted rating and the actual rating that was observed. **But the user hasn't rated all the movies, so if we're going to sum over this, we're going to sum over only over the values of $i$ where $r(i,j)=1$.** 

And we also do the usual normalization $\frac{1}{2m^{(j)}}$. And this is very much like the cost function we have for linear regression with $m$ or really $m^{(j)}$ training examples, where we're summing over the $m^{(j)}$ movies for which we have a rating, taking a squared error and then normalizing over $\frac{1}{2m^{(j)}}$.

If we minimize this function, then we should come up with a pretty good choice of parameters $w^{(j)}$ and $b^{(j)}$, for making predictions for user $j$'s ratings. 

Let's add one more term to this cost function, which is the **regularization term to prevent overfitting**:

$$\text{min } J(w^{(j)},b^{(j)}) = \frac{1}{2m^{(j)}} \sum_{i:r(j, i) = 1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2m^{(j)}} \sum_{k=1}^{n} (w_k^{(j)})^2$$

Its our usual regularization parameter: lambda divided by $2m^{(j)}$ and then times the sum of the squared values of the parameters $w$. And so $n$ is the number of features in $x^{(i)}$ and that's the same as a number of features in $w^{(j)}$.

Before moving on, it turns out that for recommender systems it is actually convenient to eliminate this division by $m^{(j)}$ term: it is just a constant in this expression. And so, even if we take it out, we should end up with the same value of $w$ and $b$:

$$\text{min } J(w^{(j)},b^{(j)}) = \frac{1}{2} \sum_{i:r(j, i) = 1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (w_k^{(j)})^2$$

So for a single user, we have the cost function above.

But instead of focusing on a single user, let's look at how we learn the parameters for all of the users. To learn the parameters $w^{(1)}$, $b^{(1)}$, $w^{(2)}$, $b^{(2)}$,...,$w^{(n_u)}$, $b^{(n_u)}$, we would take this cost function on top and sum it over all the $n_u$ users. So we would have a sum from $j = 1$ to $n_u$ of the same cost function that we had written up above.:

$$\text{min } J(w^{(j)},b^{(j)}) = \frac{1}{2} \sum_{j = 1}^{n_u} \sum_{i:r(j, i) = 1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2 + \frac{\lambda}{2}  \sum_{j = 1}^{n_u}  \sum_{k=1}^{n} (w_k^{(j)})^2$$

And this becomes the cost for learning all the parameters for all of the users. 

If we use gradient descent or any other optimization algorithm to minimize then we get a pretty good set of parameters for predicting movie ratings for all the users. And notice that this algorithm is a lot like linear regression, only now we're training a different linear regression model for each of the $n_u$ users. 

---

But where do these features come from? And what if we don't have access to such features that give we enough detail about the movies with which to make these predictions? 

## Collaborative filtering algorithm

In the last section, we saw how if we have features for each movie, such as features $x_1$ and $x_2$ that tell we how much is this a romance movie and how much is this an action movie. 

Then we can use basically linear regression to learn to predict movie ratings. But what if we don't have those features, $x_1$ and $x_2$? Let's take a look at how we can learn or come up with those features $x_1$ and $x_2$ from the data. 

Here's the data that we had before. But what if instead of having these numbers for $x_1$ and $x_2$, we didn't know in advance what the values of the features $x_1$ and $x_2$ were? we're going to replace them with question marks over here. 

Now, just for the purposes of illustration, let's say we had somehow already learned parameters for the four users. Let's say that we learned parameters $w^1$ equals 5 and 0 and $b^1$ equals 0, for user one. $w^2$ is also 5, 0 $b^2$, 0. 

$w^3$ is 0, 5 $b^3$ is 0, and for user four $w^4$ is also 0, 5 and $b^4$ 0, 0. We'll worry later about how we might have come up with these parameters, $w$ and b. But let's say we have them already. 

As a reminder, to predict user j's rating on movie i, we're going to use w^j dot product, the features of $x_i$ plus b^j. To simplify this example, all the values of $b$ are actually equal to 0. Just to reduce a little bit of writing, we're going to ignore $b$ for the rest of this example. 

Let's take a look at how we can try to guess what might be reasonable features for movie one. If these are the parameters we have on the left, then given that Alice rated movie one, 5, we should have that $w^1$.x^1 should be about equal to 5 and $w^2$.x^2 should also be about equal to 5 because Bob rated it 5. $w^3$.x^1 should be close to 0 and $w^4$.x^1 should be close to 0 as well. 

The question is, given these values for $w$ that we have up here, what choice for $x_1$ will cause these values to be right? Well, one possible choice would be if the features for that first movie, were 1, 0 in which case, $w^1$.x^1 will be equal to 5, $w^2$.x^1 will be equal to 5 and similarly, $w^3$ or $w^4$ dot product with this feature vector $x_1$ would be equal to 0. What we have is that if we have the parameters for all four users here, and if we have four ratings in this example that we want to try to match, we can take a reasonable guess at what lists a feature vector $x_1$ for movie one that would make good predictions for these four ratings up on top. 

Similarly, if we have these parameter vectors, we can also try to come up with a feature vector $x_2$ for the second movie, feature vector $x_3$ for the third movie, and so on to try to make the algorithm's predictions on these additional movies close to what was actually the ratings given by the users. Let's come up with a cost function for actually learning the values of $x_1$ and $x_2$. By the way, notice that this works only because we have parameters for four users. 

That's what allows us to try to guess appropriate features, $x_1$. This is why in a typical linear regression application if we had just a single user, we don't actually have enough information to figure out what would be the features, $x_1$ and $x_2$, which is why in the linear regression contexts that we saw in course 1, we can't come up with features $x_1$ and $x_2$ from scratch. But in collaborative filtering, is because we have ratings from multiple users of the same item with the same movie. 

That's what makes it possible to try to guess what are possible values for these features. Given $w^1$, $b^1$, $w^2$, $b^2$, and so on through w^$n_u$ and b^$n_u$, for the n subscript u users. If we want to learn the features x^i for a specific movie, i is a cost function we could use which is that. 

we're going to want to minimize squared error as usual. If the predicted rating by user j on movie i is given by this, let's take the squared difference from the actual movie rating y,i,j. As before, let's sum over all the users j. 

But this will be a sum over all values of j, where r, i, j is equal to I. I'll add a 1.5 there as usual. As I defined this as a cost function for x^i. 

Then if we minimize this as a function of x^i we be choosing the features for movie i. So therefore all the users J that have rated movie i, we will try to minimize the squared difference between what our choice of features x^i results in terms of the predicted movie rating minus the actual movie rating that the user had given it. Then finally, if we want to add a regularization term, we add the usual plus Lambda over 2, K equals 1 through n, where n as usual is the number of features of x^i squared. 

Lastly, to learn all the features $x_1$ through x^$n_m$ because we have $n_m$ movies, we can take this cost function on top and sum it over all the movies. Sum from i equals 1 through the number of movies and then just take this term from above and this becomes a cost function for learning the features for all of the movies in the dataset. So if we have parameters $w$ and b, all the users, then minimizing this cost function as a function of $x_1$ through x^$n_m$ using gradient descent or some other algorithm, this will actually allow we to take a pretty good guess at learning good features for the movies. 

This is pretty remarkable for most machine learning applications the features had to be externally given but in this algorithm, we can actually learn the features for a given movie. But what we've done so far in this section, we assumed we had those parameters $w$ and $b$ for the different users. Where do we get those parameters from? 

Well, let's put together the algorithm from the last section for learning $w$ and $b$ and what we just talked about in this section for learning $x$ and that will give us our collaborative filtering algorithm. Here's the cost function for learning the features. This is what we had derived on the last slide. 

Now, it turns out that if we put these two together, this term here is exactly the same as this term here. Notice that sum over j of all values of i is that r,i,j equals 1 is the same as summing over all values of i with all j where r,i,j is equal to 1. This summation is just summing over all user movie pairs where there is a rating. 

What we're going to do is put these two cost functions together and have this where we're just writing out the summation more explicitly as summing over all pairs i and j, where we do have a rating of the usual squared cost function and then let $me$ take the regularization term from learning the parameters $w$ and b, and put that here and take the regularization term from learning the features $x$ and put them here and this ends up being our overall cost function for learning w, b, and x. It turns out that if we minimize this cost function as a function of $w$ and $b$ as well as x, then this algorithm actually works. Here's what I mean. 

If we had three users and two movies and if we have ratings for these four movies, but not those two, over here does, is it sums over all the users. For user 1 has determined the cost function for this, for user 2 has determined the cost function for this, for user 3 has determined the cost function for this. We're summing over users first and then having one term for each movie where there is a rating. 

But an alternative way to carry out the summation is to first look at movie 1, that's what this summation here does, and then to include all the users that rated movie 1, and then look at movie 2 and have a term for all the users that had rated movie 2. we see that in both cases we're just summing over these four areas where the user had rated the corresponding movie. That's why this summation on top and this summation here are the two ways of summing over all of the pairs where the user had rated that movie. 

How do we minimize this cost function as a function of w, b, and x? One thing we could do is to use gradient descent. In course 1 when we learned about linear regression, this is the gradient descent algorithm we had seen, where we had the cost function J, which is a function of the parameters $w$ and b, and we'd apply gradient descent as follows. 

With collaborative filtering, the cost function is in a function of just $w$ and $b$ is now a function of w, b, and x. we're using $w$ and $b$ here to denote the parameters for all of the users and $x$ here just informally to denote the features of all of the movies. But if we're able to take partial derivatives with respect to the different parameters, we can then continue to update the parameters as follows. 

But now we need to optimize this with respect to $x$ as well. We also will want to update each of these parameters $x$ using gradient descent as follows. It turns out that if we do this, then we actually find pretty good values of $w$ and $b$ as well as x. 

In this formulation of the problem, the parameters of $w$ and b, and $x$ is also a parameter. Then finally, to learn the values of x, we also will update $x$ as $x$ minus the partial derivative respect to $x$ of the cost w, b, x. we're using the notation here a little bit informally and not keeping very careful track of the superscripts and subscripts, but the key takeaway I hope we have from this is that the parameters to this model are $w$ and b, and $x$ now is also a parameter, which is why we minimize the cost function as a function of all three of these sets of parameters, $w$ and b, as well as x. 

The algorithm we derived is called collaborative filtering, and the name collaborative filtering refers to the sense that because multiple users have rated the same movie collaboratively, given we a sense of what this movie maybe like, that allows we to guess what are appropriate features for that movie, and this in turn allows we to predict how other users that haven't yet rated that same movie may decide to rate it in the future. This collaborative filtering is this gathering of data from multiple users. This collaboration between users to help we predict ratings for even other users in the future. 

So far, our problem formulation has used movie ratings from 1- 5 stars or from 0- 5 stars. A very common use case of recommender systems is when we have binary labels such as that the user favors, or like, or interact with an item. In the next section, let's take a look at a generalization of the model that we've seen so far to binary labels. 

Let's go see that in the next section. 

## Binary labels: favs, likes and clicks

Many important applications of recommender systems or collective filtering algorithms involved binary labels where instead of a user giving we a one to five star or zero to five star rating, they just somehow give we a sense of they like this item or they did not like this item. Let's take a look at how to generalize the algorithm we've seen to this setting. 

The process we'll use to generalize the algorithm will be very much reminiscent to how we have gone from linear regression to logistic regression, to predicting numbers to predicting a binary label back in course one, let's take a look. Here's an example of a collaborative filtering data set with binary labels. A one the notes that the user liked or engaged with a particular movie. 

So label one could mean that Alice watched the movie "Love at Last" all the way to the end and watch "Romance Forever" all the way to the end. But after playing a few minutes of "Nonstop Car Chases" decided to stop the section and move on. Or it could mean that she explicitly hit like or favorite on an app to indicate that she liked these movies. 

But after checking out nonstop car chasers and swords versus karate did not hit like. And the question mark usually means the user has not yet seen the item and so they weren't in a position to decide whether or not to hit like or favorite on that particular item. So the question is how can we take the collaborative filtering algorithm that we saw in the last section and get it to work on this dataset. 

And by predicting how likely Alice, Bob carol and Dave are to like the items that they have not yet rated, we can then decide how much we should recommend these items to them. There are many ways of defining what is the label one and what is the label zero, and what is the label question mark in collaborative filtering with binary labels. Let's take a look at a few examples. 

In an online shopping website, the label could denote whether or not user j chose to purchase an item after they were exposed to it, after they were shown the item. So one would denote that they purchase it zero would denote that they did not purchase it. And the question mark would denote that they were not even shown were not even exposed to the item. 

Or in a social media setting, the labels one or zero could denote did the user favorite or like an item after they were shown it. And question mark would be if they have not yet been shown the item or many sites instead of asking for explicit user rating will use the user behavior to try to guess if the user like the item. So for example, we can measure if a user spends at least 30 seconds of an item. 

And if they did, then assign that a label one because the user found the item engaging or if a user was shown an item but did not spend at least 30 seconds with it, then assign that a label zero. Or if the user was not shown the item yet, then assign it a question mark. Another way to generate a rating implicitly as a function of the user behavior will be to see that the user click on an item. 

This is often done in online advertising where if the user has been shown an ad, if they clicked on it assign it the label one, if they did not click assign it the label zero and the question mark were referred to if the user has not even been shown that ad in the first place. So often these binary labels will have a rough meaning as follows. A labor of one means that the user engaged after being shown an item And engaged could mean that they clicked or spend 30 seconds or explicitly favorite or like to purchase the item. 

A zero will reflect the user not engaging after being shown the item, the question mark will reflect the item not yet having been shown to the user. So given these binary labels, let's look at how we can generalize our algorithm which is a lot like linear regression from the previous couple sections to predicting these binary outputs. Previously we were predicting label yij as wj.xi+b. 

So this was a lot like a linear regression model. For binary labels, we're going to predict that the probability of yijb=1 is given by not wj.xi+b. But it said by g of this formula, where now g(z) 1/1 +e to the -z. 

So this is the logistic function just like we saw in logistic regression. And what we would do is take what was a lot like a linear regression model and turn it into something that would be a lot like a logistic regression model where will now predict the probability of yij being 1 that is of the user having engaged with or like the item using this model. In order to build this algorithm, we'll also have to modify the cost function from the squared error cost function to the cost function that is more appropriate for binary labels for a logistic regression like model. 

So previously, this was the cost function that we had where this term play their role similar to f(x), the prediction of the algorithm. When we now have binary labels, yij when the labels are one or zero or question mark, then the prediction f(x) becomes instead of wj.xi+b j it becomes g of this where g is the logistic function. And similar to when we had derived logistic regression, we had written out the following loss function for a single example which was at the loss if the algorithm predicts f(x) and the true label was y, the loss was this. 

It was -y log f-y log 1-f. This is also sometimes called the binary cross entropy cost function. But this is a standard cost function that we used for logistic regression as was for the binary classification problems when we're training neural networks. 

And so to adapt this to the collaborative filtering setting, let $me$ write out the cost function which is now a function of all the parameters $w$ and $b$ as well as all the parameters $x$ which are the features of the individual movies or items of. We now need to sum over all the pairs ij where riij=1 notice this is just similar to the summation up on top. And now instead of this squared error cost function, we're going to use that loss function. 

There's a function of f(x), yij. Where f(x) here? That's $my$ abbreviation. 

My shorthand for g(w) j.$x_1$+ej. As we plug this into here, then this gives we the cost function they could use for collaborative filtering on binary labels. So that's it. 

That's how we can take the linear regression, like collaborative filtering algorithm and generalize it to work with binary labels. And this actually very significantly opens up the set of applications we can address with this algorithm. Now, even though we've seen the key structure and cost function of the algorithm, there are also some implementation, all tips that will make our algorithm work much better. 

Let's go on to the next section to take a look at some details of how we implement it and some little modifications that make the algorithm run much faster. Let's go on to the next section.