# Skewed datasets

## Error metrics for skewed datasets (precision and recall)

**If we're working on a machine learning application where the ratio of positive to negative examples is very skewed, (very far from 50-50), then the usual error metrics like accuracy don't work that well**. 

Let's start with an example where we're training a binary classifier to detect a rare disease in patients based on lab tests or based on other data from the patients:

- $y$ is equal to 1 if the disease is present
- $y$ is equal to 0 otherwise

Suppose we find that we've achieved 1% error on the test set, so we have a 99% correct diagnosis. A**lthough this seems like a great outcome, this is a rare disease, so $y$ is equal to 1 only very rarely**, then this may not be as impressive as it sounds. 

Specifically, if it is a rare disease and if only 0.5% of the patients in our population have the disease, then even an "algorithm" that just classified all patients as not having the disease would perform better.

![](2024-02-09-00-25-35.png)

So this really means is that **we can't tell if getting 1% error is actually a good result or a bad result**. 

In particular, if we have one algorithm that achieves 99.5% accuracy, different one that achieves 99.2% accuracy, different one that achieves 99.6% accuracy, but's difficult to know which of these is actually the best algorithm.

Why? Because if we have an algorithm that achieves 0.5% error and a different one that achieves 1% error and a different one that achieves 1.2% error, it's difficult to know which of these is the best algorithm, since the one with the lowest error might not be particularly useful with predictions - like the example which we mentioned, where $y$ always equals 0 and never ever diagnose any patient as having this disease. 

Quite possibly an algorithm that has 1% error, but that at least diagnosis some patients as having the disease could be more useful than just printing $y$ equals 0 all the time. 

![](2024-02-09-00-28-24.png)

When working on problems with **skewed data sets, we usually use a different error metric** rather than just classification error to figure out how well our learning algorithm is doing. A common pair of error metrics are **precision** and **recall**.

Let's define, in this example, when $y$ equals 1, it refers to the rare class, such as the rare disease that we may want to detect. 

To evaluate a learning algorithm's performance with one rare class it's useful to construct what's called a **confusion matrix**, which is a two-by-two matrix or a two-by-two table.

- On the horizontal axis on top, we're going to write the actual class, which could be one or zero. 
- On the vertical axis, we're going to write the predicted class, which is what did our learning algorithm predicts on a given example.

And we will thus have four grid cells:

- **True positives**: where the predicted class was 1 and the actual class was 1.
- **False negative**: where the predicted class was 0 and the actual class was 1.
- **False positive**: where the predicted class was 1 and the actual class was 0.
- **True negative**: where the predicted class was 0 and the actual class was 0.

![](2024-02-09-00-35-13.png)

Assigning our datapoints from our cross-validation or test sets into each of this quadrants, we can then calculate **precision** and **recall**:


### Precision

Of all the patientes where we predicted $y = 1$, what fraction actually have the rare disease?

$$ \text{Precision} = \frac{\text{True positives}}{\text{\# predicted positive}}$$
$$ \text{Precision} = \frac{\text{True positives}}{\text{True positives}+\text{False positives}}$$

In this example:

$$ \text{Precision} = \frac{15}{15+5} =  0.75$$

So, **for all the patients that it predicted to have the desease, it was right 75% of the times.**

### Recall

Of all the patients that actually have the rare disease, what fraction did we correctly detect as having it?

$$ \text{Recall} = \frac{\text{True positives}}{\text{\# actual positives}}$$
$$ \text{Recall} = \frac{\text{True positives}}{\text{True positives}+\text{False negatives}}$$

In this example:

$$ \text{Recall} = \frac{15}{15+10} =  0.6$$

---
Notice that these metrics will help we detect if the learning algorithm is just printing $y$ equals 0 all the time: in that case,  then the numerator of both of these quantities would be zero, since there wouldn't be any true positives. 

In practice, if an algorithm doesn't predict even a single positive, we just say that precision is also equal to zero. But we'll find that computing both precision and recall makes it easier to spot if an algorithm is both reasonably accurate, in that, when our model says a patient has a disease, there's a good chance the patient has a disease, such as 0.75 chance in this example. And also guarantees, with recall,that of all the patients that have the disease, it's helping to diagnose a reasonable fraction of them, such as here it's finding 60% of them. 

## Trading off precision and recall

In the ideal case, we would for learning algorithms that have high precision and high recall. 

- High precision would mean that if a diagnosis of patients have that rare disease, probably the patient does have it and it's an accurate diagnosis. 
- High recall means that if there's a patient with that rare disease, probably the algorithm will correctly identify that they do have that disease. 

But in practice, there's **a trade-off between precision and recall.** 

Remember that precision and recall are: 

$$ \text{Precision} = \frac{\text{True positives}}{\text{True positives}+\text{False positives}}$$

$$ \text{Recall} = \frac{\text{True positives}}{\text{True positives}+\text{False negatives}}$$

If we're using logistic regression to make predictions, then the logistic regression model will output numbers between 0 and 1. We have, till now, typically threshold the output of logistic regression at $0.5$ and predicted 1 if $f(x)$ is greater than or equal to 0.5 and predict 0 if it's less than 0.5. 


But suppose **we want to predict that $y$ is equal to 1, that is, that the rare disease is present only if we're very confident**. 

This might be the case in which we know that we predict a patient to have the disease, we may have to send them for possibly invasive and expensive treatment. If the consequences of the disease aren't that bad, even if left not treated, then we may want to predict $y$ equals 1 only if we're very confident.

![](2024-02-09-15-53-23.png)

In that case, we may choose to set a higher threshold, where we will predict that $y$ is 1 only if $f(x)$ is greater than or equal to, for example, $0.7$. This is saying we'll predict $y$ equals 1 only we're at least 70% sure, rather than just 50% sure.

**By raising this threshold**, we predict $y$ equals 1 only if we're pretty confident. What that means is that **precision will increase** because whenever we predict one, we're more likely to be right .

But: it also **results in lower recall because we're now predicting the disease less often: this means that, from the total number of patients with the disease, we're going to correctly diagnose fewer of them.** 

By raising this threshold to 0.7, **we end up with higher precision, but lower recall.**

In fact, if we want to predict $y$ equals 1 only if we are very confident, we can even raise this higher to 0.9 and that results in an even higher precision and so whenever we predict the patient has the disease, we're probably right and this will give we a very high precision. The recall will go even further down. 

---
**On the contrary, suppose we want to avoid missing too many cases of the rare disease**. So, when in doubt, predict $y$ equals 1. This might be the case where if treatment is not too invasive or painful or expensive but leaving a disease untreated has much worse consequences for the patient. 

If for our application, that is the better way to make decisions, then we would instead **lower threshold**. For example, set it to 0.3. **In that case, we predict one so long as we think there's maybe a 30% chance or better of the disease being present and we predict zero only if we're pretty sure that the disease is absent.**

As we can imagine, the impact on precision and recall will be opposite to what we saw before: lowering this threshold will result in **lower precision** (we're more willing to predict a disease even if we aren't sure) but to result in **higher recall** (because of all the patients that do have that disease, we're probably going to correctly identify more of them). 

![](2024-02-09-16-01-52.png)

More generally, we have the flexibility to predict one only if $f(x)$ is above some threshold. And by choosing this threshold, we can make different trade-offs between precision and recall. 

For most algorithms, there is a trade-off between precision and recall: Precision and recall both go between zero and one. 

![](2024-02-09-16-03-38.png)

Ff we were to set a very high threshold, (a threshold of 0.99), then we enter with very high precision, but lower recall. As we reduce the value of this threshold, we then end up with a curve that trades off precision and recall until eventually, if we have a very low threshold, (threshold of 0.01) and  we end up with very low precision but relatively high recall. 

![](2024-02-09-16-05-21.png)

By plotting this curve, we can  try to pick a threshold which corresponds to picking a point on this curve that balances the cost of false positives and false negatives; or that balances the benefits of high precision and high recall. 

Notice that **picking the threshold** is not something we can really do with cross-validation because **it's up to the creator of the model to manually define the threshold that fits their use case.**

---
However, **if we want to automatically trade-off precision and recall** rather than have to do so yourself, there is another metric called the **F1 score** that is **used to automatically combine precision and recall, to help uswe pick the best value or the best trade-off between the two.** 

One challenge with precision/recall is that we evaluating our algorithms using two different metrics: 

![](2024-02-09-16-15-26.png)

Ff we've trained three different algorithms and the precision-recall numbers look like the ones in the table above, it is not that obvious how to pick which algorithm to use. 

In this example, **Algorithm 2 has the highest precision**, but **Algorithm 3 has the highest recall**, and **Algorithm 1 trades off the two in-between, and so no one algorithm is obviously the best choice**. 

In order to help we decide which algorithm to pick, it may be useful to find a way to combine precision and recall into a single score.

If we were to take the average, we get $0.45$, $0.4$, and $0.5$. But picking the algorithm with the highest average between precision and recall doesn't work that well: in this case, **Algorithm 3** has the highst average, but also has very low precision. In fact, this corresponds maybe to an algorithm that actually does print $y$ equals 1 for all cases, diagnosing all patients as having the disease. And that would explain why recall is perfect but the precision is really low. So, **Algorithm 3** is actually not a particularly useful algorithm, even though the average between precision and recall is quite high.

![](2024-02-09-16-19-58.png)

Instead, **the most common way of combining precision recall is using the F1 score**. It is a way of combining P and R, precision and recall, but that **gives more emphasis to whichever of these values is lower**, because, if an algorithm has very low precision or very low recall it is simply not that useful. 

The formula for computing F1 score is:

$$ F_1 \text{score} = \frac{1}{\frac{1}{2}(\frac{1}{P}+\frac{1}{R})} $$

If we simplify this equation it can also be computed as follows:

$$ F_1 \text{score} = 2\frac{PR}{P+R} $$

If we were to compute the $F_1$ score for these three algorithms, we'll find that, for Algorithm 1 it is 0.444, and for the second algorithm is 0.175, (notice that 0.175 is much closer to the lower value than the higher value) and for the third algorithm is 0.0392.

F1 score gives away to trade-off precision and recall, and in this case, **it tells us that maybe the first algorithm is better than the second or the third algorithms.** In math, this equation is also called the **harmonic mean of P and R**, and the harmonic mean is a way of taking an average that emphasizes the smaller values more. 