# Continuous state spaces

## Example of continuous state space applications

Many robotic control applications, including the lunar lander application that we work on in the practice lab, have continuous state spaces. Let's take a look at what that means and how to generalize the concept we've talked about to these continuous state spaces. 

A simplify Mars rover example we use, I use a discrete set of states, and what that means is that simplify Mars rover could only be in one of six possible positions. But most robots can be in more than one of six or any discrete number of positions, instead, they can be in any of a very large number of continuous value positions. For example, if the Mars rover could be anywhere on a line, so its position was indicated by a number ranging from 0-6 kilometers where any number in between is valid. 

That would be an example of a continuous state space, because the position would be represented by a number such as that is 2.7 kilometers along or 4.8 kilometers or any other number between zero and six. Let's look at another example. we're going to use for this example, the application of controlling a car or a truck. 

Here's a toy car, or actually a toy truck. This one belongs to $my$ daughter. If we're building a self-driving car or self-driving truck and we want to control this to drive smoothly, then the state of this truck might include a few numbers such as this, $x$ position, its $y$ position, maybe it's orientation. 

What way is it facing? Assuming the truck stays on the ground, we probably don't need to worry about how tall is, how high up it is. This is state will include x, y, and is angle Theta, as well as maybe its speeds in x-direction, the speed in the y-direction, and how quickly it's turning. 

Is it turning at one degree per second or is it turning at 30 degrees per second or is it turning really quickly at 90 degrees per second? For a truck or a car, the state might include not just one number, like how many kilometers of this along on this line, but they might includes six numbers, is $x$ position, is $y$ position, is orientation, which we're going to denote using Greek alphabet Theta, as well its velocity in the x-direction, which I will denote using $x$ dot, so that means how quickly is this x-coordinate changing, $y$ dot how quickly is the $y$ coordinate changing, and then finally, Theta dot, which is how quickly is the angle of the car changing. Whereas for the 60 Mars rover example, the state was just one of six possible numbers. 

It could be one, two, three, four, five or six. For the car, the state would comprise this vector of six numbers, and any of these numbers can take on any value within is valid range. For example, Theta should range between zero and 360 degrees. 

Let's look at another example. What if we're building a reinforcement learning algorithm to control an autonomous helicopter, how would we characterize the position of a helicopter? To illustrate, I have with $me$ here a small toy helicopter. 

The positioning of the helicopter would include is $x$ position, such as how far north or south is a helicopter, is $y$ position. Maybe how far on the east-west axis is the helicopter, and then also z, the height of the helicopter above ground. But other than the position, the helicopter also has an orientation, and conventionally, one way to capture its orientation is with three additional numbers, one of which captures of the row of the helicopter. 

Is it rolling to the left or the right? The pitch, is it pitching forward or pitching up, pitching back, and then finally the yaw which is west the compass orientation is it facing. If facing north or east or south or west? 

To summarize, the state of the helicopter includes is position in the say, north-south direction, is positioned in the east-west direction, $y$ is height above ground, and also the row, the pitch, and also that yaw of helicopter. To write this down, the state therefore includes the position x, y, z, and then the row pitch, and yaw denoted with the Greek alphabets Phi, Theta and Omega. But to control the helicopter, we also need to know its speed in the x-direction, in the y-direction, and in the z direction, as well as its rate of turning, also called the angular velocity. 

How fast is this row changing and how fast is this pitch changing and how fast is its yaw changing? This is actually the state used to control autonomous helicopters. Is this list of 12 numbers that is input to a policy, and the job of a policy is look at these 12 numbers and decide what's an appropriate action to take in the helicopter. 

So any continuous state reinforcement learning problem or a continuous state Markov decision process, continuously MTP. The state of the problem isn't just one of a small number of possible discrete values, like a number from 1-6. Instead, it's a vector of numbers, any of which could take any of a large number of values. 

In the practice lab for this week, we get to implement for yourself a reinforcement learning algorithm applied to a simulated lunar lander application. Landing something on the moon is simulation. Let's take a look in the next section at what that application entails, since there will be another continuous state application. 

## Lunar lander

The lunar lander lets we land a simulated vehicle on the moon. It's like a fun little section game that's been used by a lot of reinforcement learning researchers. Let's take a look at what it is. 

In this application we're in command of a lunar lander that is rapidly approaching the surface of the moon. And our job is the fire thrusters at the appropriate times to land it safely on the landing pad. To give we a sense of what it looks like. 

This is the lunar lander landing successfully and it's firing thrusters downward and to the left and right to position itself to land between these two yellow flags. Or if the reinforcement landing algorithm policy does not do well then this is what it might look like where the lander unfortunately has crashed on the surface of the moon. In this application we have four possible actions on every time step. 

we could either do nothing, in which case the forces of inertia and gravity pull we towards the surface of the moon. Or we can fire a left thruster when we see a little red dot come out on the left, that's firing the left. They'll tend to push the lunar lander to the right or we can fire the main engine that's thrusting down the bottom here. 

Or we can fire the right thruster and that's firing the right thruster which will push we to the left and our job is to keep on picking actions over time. So it's the lunar lander safely between these two flags here on the landing pad. In order to give the actions a shorter name we're sometimes going to call the actions nothing meaning do nothing or left meaning fire left thruster or main meaning fire the main engine downward or right. 

So we're going to call the actions nothing left. main and right for short later in this section. How about the states space of this? 

Mtp the states are its position X and Y. So how far to the left or right and how high up is it as well as velocity x.y how fast is it moving in the horizontal and vertical directions and then also is angle. So how far is the lunar lander tilted to the left or tilted to the right? 

Is angular velocity theta dot. And then finally, because a small difference in positioning makes a big difference in whether or not it's landed. We're going to have two other variables in the state vector which we call l and r. 

Which corresponds to whether the left leg is grounded, meaning whether or not the left leg is sitting on the ground as well as r which corresponds to whether or not the right leg is sitting on the ground. So whereas xy x.theta theta.our numbers l and r will be binary valued and can take on only values zero or one depending on whether the left and right legs are touching the ground. Finally his reward function for the lunar lander. 

If it manages to get to the landing pad, didn't receive the reward between 100 and 140 depending on how well it's flown when gotten to the center of the landing pad. We also give it an additional reward for moving toward or away from the pad so it moves closer to the pad it receives a positive reward and it moves away and drifts away. It receives a negative reward. 

If it crashes it gets a large -100 reward, it achieves a soft landing, that is a landing. There's another crash, it gets a +100 reward for each leg, the left leg or the right link that it gets grounded. It receives a +10 reward and finally to encourage it not to waste too much fuel and fire thrusters aren't necessarily. 

Every time it fires the main engine we give it a -0.3 rewards and every time it fires the left or the right side thrusters we give it a -0.03 reward. Notice that this is a moderately complex reward function. The designers of the lunar lander application actually put some thought into exactly what behavior we want and codified it in the reward function. 

To incentivize more of the behaviors we want and fewer of the behaviors like crashing that we don't want. we find when we're building our own reinforcement learning application usually takes some thought to specify exactly what we want or don't want and to codify that in the reward function. But specify the reward function should still turn out to be much easier to specify the exact right action to take from every single state. 

Which is much harder for this and many other reinforcement learning applications. So the lunar lander problem is as follows. Our goal is to learn a policy pi. 

That when given a state S as written here picks an action a equals pi of S. So as to maximize the return the sum of discounted rewards. And usually for the lunar lander would use a fairly large value for gamma ra. 

In fact for the would use the value of gamma that's equal to 0.985 so pretty close to one. And if we can learn a policy pi that does this then we successfully land this lunar lander exciting application and we're now finally ready to develop a learning algorithm which will turn out to use deep learning or neural networks to come up with a policy to land the lunar lander. Let's go into the next section where we start to learn about deep reinforcement. 

## Learning the state-value function

Let's see how we can use reinforcement learning to control the Lunar Lander or for other reinforcement learning problems. The key idea is that we're going to train a neural network to compute or to approximate the state action value function Q of s, a and that in turn will let us pick good actions. Let's see how it works. 

The heart of the learning algorithm is we're going to train a neural network that inputs the current state and the current action and computes or approximates Q of s, a. In particular, for the Lunar Lander, we're going to take the state s and any action a and put them together. Concretely, the state was that list of eight numbers that we saw previously, so we have xy, $x$ dot, $y$ dot, Theta, Theta dot, and then LR for where the legs are grounded, so that's a list of eight numbers that describe the state. 

Then finally, we have four possible actions: nothing, left, main, a main engine, and right. We can encode any of those four actions using a one-hot feature vector. If action were the first action, we may encode it using 1, 0, 0, 0 or if it was the second action to find the left cluster, we may encode it as 0, 1, 0, 0. 

This list of 12 numbers, eight numbers for the state and then four numbers, a one-hot encoding of the action is the inputs we'll have to the neural network, and we're going to call this X. We'll then take these 12 numbers and feed them to a neural network with say, 64 units in the first hidden layer, 64 units in the second hidden layer, and then a single output in the output layer. The job of the neural network is the output Q of s, a. 

The state action-value function for the Lunar Lander given the input s and a. Because we'll be using neural network training algorithms in a little bit, we're also going to refer to this value Q of s, a as the target value Y that were training the neural network to approximate. Notice that I did say reinforcement learning is different from supervised learning, but what we're going to do is not input a state and have it output an action. 

What we're going to do is input a state action pair and have it try to output Q of s, a, and using a neural network inside the reinforcement learning algorithm this way will turn out to work pretty well. We'll see the details in a little bit so don't worry about it if it doesn't make sense yet. But if we can train a neural network with appropriate choices of parameters in the hidden layers and in the upper layer to give we a good estimates of Q of s, a, then whenever we're Lunar Lander is in some state s, we can then use the neural network to compute Q of s, a. 

For all four actions, we can compute Q of s, nothing, Q of s, left, Q of s, main, Q of s, right, and then finally, whichever of these has the highest value, we pick the corresponding action a. For example, if out of these four values, Q of s, main is largest, then we would decide to go and fire the main engine of the Lunar Lander. The question becomes, how do we train a neural network to output Q of s, a? 

It turns out the approach will be to use Bellman's equations to create a training set with lots of examples, $x$ and y, and then we'll use supervised learning exactly as we learned in the second course when we talked about neural networks. To learn using supervised learning, a mapping from $x$ to y, that is a mapping from the state action pair to this target value Q of s, a. But how do we get the training set with values for $x$ and $y$ that we can then train a neural network on? 

Let's take a look. Here's the Bellman equation, Q of s, a equals R of s plus Gamma, max of a prime, Q of s prime, a prime. The right-hand side is what we want Q of s, a to be equal to, so we're going to call this value on the right-hand side $y$ and the input to the neural network is a state and an action so we're going to call that x. 

The job of the neural network is to input x, that is input the state action pair, and try to accurately predict what will be the value on the right. In supervised learning, we were training a neural network to learn a function f, which depends on a bunch of parameters, W and B, the parameters of the various layers of the neural network and it was the job of the neural network to input x. Hopefully I'll put something close to the target value y. 

The question is, how can we come up with a training set with values $x$ and $y$ for a neural network to learn from. Here's what we're going to do. We're going to use the lunar lander and just try taking different actions in it. 

If we don't have a good policy yet, we'll take actions randomly, fire the left thruster, fire the, right thruster, fire the main engine, do nothing. By just trying out different things in the lunar lander simulator we'll observe a lot of examples of when we're in some state and we took some action, may be a good action, maybe a terrible action either way. Then we got some reward R of s for being in that state, and as a result of our action, we got to some new state, S'. 

As we take different actions in the lunar lander, we see this S, a, R of s, S', and we call them tuples in Python code many times. For example, maybe one time we're in some state S and just to give this an index and we call this S^1, and we happen to take some action a^1, this could be nothing left, main thrusters or right. As a result of which we've got some reward, and we wound up at some state S^'1. 

Maybe at different times we're in some other state S2, we took some other action, could be good action, could be bad action, could be any of the four actions, and we got the reward, and then we wound up with S '2 and so on multiple times. Maybe we've done this 10,000 times or even more than 10,000 times, so we would have to save the way with not just S^1, a^1 and so on, but up to S^10,000, a^10,000. It turns out that each of these tuples will be enough to create a single training example, x^1, y^1. 

In particular, here's how we do it. There are four elements in this first tuple. The first two will be used to compute x^1, and the second two would be used to compute y^1. 

In particular, x^1 is just going to be S^1, a^1 put together. S^1 would be eight numbers, the state of the lunar lander, a^1 would be four numbers, the one-hot encoding of whatever action this was and y^1 would be computed using the right-hand side of the Bellman equation. In particular, the Bellman equation says, when we input S^1, a^1, we want Q of S^1, a^1 to be this right hand side, to be equal to R of S^1 plus Gamma max over a' of Q of S^1'a' prime. 

Notice that these two elements of the tuple on the right give we enough information to compute this. we know what is R of S^1? That's the reward we've saved the way here. 

Plus the discount factor gamma times max over all actions, a' of Q of S'^1, that's a state we got to in this example, and then take the maximum over all possible actions, a'. we're going to call this y^1. When we compute this, this will be some number like 12.5 or 17, or 0.5 or some other number. 

We'll save that number here as y^1, so that this pair x^1, y^1 becomes the first training example in this little dataset we're computing. Now, we may be wondering, where does Q of S', a' or Q of S'^1, a' come from. Well, initially we don't know what is the Q function. 

But it turns out that when we don't know what is the Q function, we can start off with taking a totally random guess. What is the Q function? We'll see on the next slide that the algorithm will work nonetheless. 

But in every step Q here is just going to be some guess. They'll get better over time it turns out of what is the actual Q function. Let's look at the second example. 

If we had a second experience where we are in state S^2 to got to a^2, got that reward and then got to that state. Then we would create a second training example in this dataset, x^2, where the input is now S^2, a^2, so the first two elements go to computing the input x, and then y^2 will be equal to R of s^2 plus gamma max of a' Q of S' to a', and whatever this number is, y^2. We put this over here in our small but growing training set, and so on and so forth, until maybe we end up with 10,000 training examples with these x, $y$ pairs. 

What we'll see later is that we'll actually take this training set where the x's are inputs with 12 features and the y's are just numbers. We'll train a neural network with, say, the mean squared error loss to try to predict $y$ as a function of the input x. What I describe here is just one piece of the learning algorithm we'll use. 

Let's put it all together on the next slide and see how it all comes together into a single algorithm. Let's take a look at what a full algorithm for learning the Q-function is like. First, we're going to take our neural network and initialize all the parameters of the neural network randomly. 

Initially we have no idea, whether it's a Q function, let's just pick totally random values of the weights. We'll pretend that this neural network is our initial random guess for the Q-function. This is a little bit like when we are training linear regression and we initialize all the parameters randomly and then use gradient descent to improve the parameters. 

Initializing it randomly for now is fine. What's important is whether the algorithm can slowly improve the parameters to get a better estimate. Next, we will repeatedly do the following; We will take actions in the lunar lander, so float around randomly, take some good actions, take some bad actions. 

It's okay either way. But we get lots of these tuples of when it was in some state, we took some action a get a reward R of S and we got to some state s prime. What we will do is store to 10,000 most recent examples of these tuples. 

As we run this algorithm, we will see many steps in the lunar lander, maybe hundreds of thousands of steps. But to make sure we don't end up using excessive computer memory, common practice is to just remember the 10,000 most recent such tuples that we saw taking actions in the MTP. This technique of storing the most recent examples only is sometimes called the replay buffer in reinforcement learning algorithm. 

For now, we're just flying the lunar lander randomly, sometimes crashing, sometimes not and getting these tuples as experienced for our learning algorithm. Occasionally then we will train the neural network. In order to train the neural network, here's what we'll do. 

We'll look at these 10,000 most recent tuples we had saved and create a training set of 10,000 examples. Training set needs lots of pairs of $x$ and y. For our training examples, $x$ will be the s, a from this part of the tuple. 

There'll be a list of 12 numbers, the 8 numbers for the state and the 4 numbers for the one-hot encoding of the action. The target value that we want a neural network to try to predict would be $y$ equals R of S plus Gamma max of a prime, Q of s prime a prime. How do we get this value of Q? 

Well, initially is this neural network that we had randomly initialized. It may not be a very good guess, but it's a guess. After creating these 10,000 training examples we'll have training examples $x_1$, y1 through $x_10$,000, y10,000. 

We'll train a neural network and we're going to call the new neural network Q new, such that Q new of s, a learns to approximate y. This is exactly training that neural network to output f with parameters $w$ and b, to input $x$ to try to approximate the target value y. Now, this neural network should be a slightly better estimates of what the Q function or the state action value function should be. 

What we'll do is we're going to take Q and set it to this new neural network that we had just learned. Many of the ideas in this algorithm are due to Mnih et al. It turns out that if we run this algorithm where we start with a really random guess of the Q function, then use Bellman's equations to repeatedly try to improve the estimates of the Q function. 

Then by doing this over and over, taking lots of actions, training a model, that will improve our guess for the Q-function. For the next model we train, we now have a slightly better estimate of what is the Q function. Then the next model we train will be even better. 

When we update Q equals Q new. Then for the next time we train a model Q of s prime a prime will be an even better estimate. As we run this algorithm on every iteration, Q of s prime, a prime hopefully becomes an even better estimate of the Q function so that when we run the algorithm long enough, this will actually become a pretty good estimate of the true value of Q of s, a, so that we can then use this to pick, hopefully good actions or the MTP. 

The algorithm we just saw is sometimes called the DQN algorithm which stands for Deep Q-Network because we're using deep learning and neural network to train a model to learn the Q functions. Hence DQN or DQ using a neural network. If we use the algorithm as I described it, it will work, okay, on the lunar lander. 

Maybe it'll take a long time to converge, maybe it won't land perfectly, but it'll work. But it turns out that with a couple of refinements to the algorithm, it can work much better. In the next few sections, let's take a look at some refinements to the algorithm that we just saw. 

## Algorithm refinement: Imporved neural network architecture

In the last section, we saw a neural network architecture that will input the state and action and attempt to output the Q function, Q of s, a. It turns out that there's a change to neural network architecture that make this algorithm much more efficient. Most implementations of DQN actually use this more efficient architecture that we'll see in this section. 

Let's take a look. This was the neural network architecture we saw previously, where it would input 12 numbers and output Q of s, a. Whenever we are in some state s, we would have to carry out inference in the neural network separately four times to compute these four values so as to pick the action a that gives us the largest Q value. 

This is inefficient because we have to carry our inference four times from every single state. Instead, it turns out to be more efficient to train a single neural network to output all four of these values simultaneously. This is what it looks like. 

Here's a modified neural network architecture where the input is eight numbers corresponding to the state of the lunar lander. It then goes through the neural network with 64 units in the first hidden layer, 64 units in the second hidden layer. Now the output unit has four output units, and the job of the neural network is to have the four output units output Q of s, nothing, Q of s, left, Q of s, main, and q of s, right. 

The job of the neural network is to compute simultaneously the Q value for all four possible actions for when we are in the state s. This turns out to be more efficient because given the state s we can run inference just once and get all four of these values, and then very quickly pick the action a that maximizes Q of s, a. we notice also in Bellman's equations, there's a step in which we have to compute max over a prime Q of s prime a prime, this multiplied by gamma and then there was plus R of s up here. 

This neural network also makes it much more efficient to compute this because we're getting Q of s prime a prime for all actions a prime at the same time. we can then just pick the max to compute this value for the right-hand side of Bellman's equations. This change to the neural network architecture makes the algorithm much more efficient, and so we will be using this architecture in the practice lab. 

Next, there's one other idea that'll help the algorithm a lot which is something called an Epsilon-greedy policy, which affects how we choose actions even while we're still learning. Let's take a look at the next section , and what that means. ## Algorithm refinement: $\epsilon$-greedy policy

The learning algorithm that we developed, even while we're still learning how to approximate Q(s,a), we need to take some actions in the lunar lander. 

How do we pick those actions while we're still learning? The most common way to do so is to use something called an Epsilon-greedy policy. Let's take a look at how that works. 

Here's the algorithm that we saw earlier. One of the steps in the algorithm is to take actions in the lunar lander. When the learning algorithm is still running, we don't really know what's the best action to take in every state. 

If we did, we'd already be done learning. But even while we're still learning and don't have a very good estimate of Q(s,a) yet, how do we take actions in this step of the learning algorithm? Let's look at some options. 

When we're in some state s, we might not want to take actions totally at random because that will often be a bad action. One natural option would be to pick whenever in state s, pick an action a that maximizes Q(s,a). We may say, even if Q(s,a) is not a great estimate of the Q function, let's just do our best and use our current guess of Q(s,a) and pick the action a that maximizes it. 

It turns out this may work okay, but isn't the best option. Instead, here's what is commonly done. Here's option 2, which is most of the time, let's say with probability of 0.95, pick the action that maximizes Q(s,a). 

Most of the time we try to pick a good action using our current guess of Q(s,a). But the small fraction of the time, let's say, five percent of the time, we'll pick an action a randomly. Why do we want to occasionally pick an action randomly? 

Well, here's why. Suppose there's some strange reason that Q(s,a) was initialized randomly so that the learning algorithm thinks that firing the main thruster is never a good idea. Maybe the neural network parameters were initialized so that Q(s, main) is always very low. 

If that's the case, then the neural network, because it's trying to pick the action a that maximizes Q(s,a), it will never ever try firing the main thruster. Because it never ever tries firing the main thruster, it will never learn that firing the main thruster is actually sometimes a good idea. Because of the random initialization, if the neural network somehow initially gets stuck in this mind that some things are bad idea, just by chance, then option 1, it means that it will never try out those actions and discover that maybe is actually a good idea to take that action, like fire the main thrusters sometimes. 

Under option 2 on every step, we have some small probability of trying out different actions so that the neural network can learn to overcome its own possible preconceptions about what might be a bad idea that turns out not to be the case. This idea of picking actions randomly is sometimes called an exploration step. Because we're going to try out something that may not be the best idea, but we're going to just try out some action in some circumstances, explore and learn more about an action in the circumstance where we may not have had as much experience before. 

Taking an action that maximizes Q(s,a), sometimes this is called a greedy action because we're trying to actually maximize our return by picking this. Or in the reinforcement learning literature, sometimes we'll also hear this as an exploitation step. I know that exploitation is not a good thing, nobody should ever explore anyone else. 

But historically, this was the term used in reinforcement learning to say, let's exploit everything we've learned to do the best we can. In the reinforcement learning literature, sometimes we hear people talk about the exploration versus exploitation trade-off, which refers to how often do we take actions randomly or take actions that may not be the best in order to learn more, versus trying to maximize our return by say, taking the action that maximizes Q (s,a). This approach, that is option 2, has a name, is called an Epsilon-greedy policy, where here Epsilon is 0.05 is the probability of picking an action randomly. 

This is the most common way to make our reinforcement learning algorithm explore a little bit, even whilst occasionally or maybe most of the time taking greedy actions. By the way, lot of people have commented that the name Epsilon-greedy policy is confusing because we're actually being greedy 95 percent of the time, not five percent of the time. So maybe 1 minus Epsilon-greedy policy, because it's 95 percent greedy, five percent exploring, that's actually a more accurate description of the algorithm. 

But for historical reasons, the name Epsilon-greedy policy is what has stuck. This is the name that people use to refer to the policy that explores actually Epsilon fraction of the time rather than this greedy Epsilon fraction of the time. Lastly, one of the trick that's sometimes used in reinforcement learning is to start off Epsilon high. 

Initially, we are taking random actions a lot at a time and then gradually decrease it, so that over time we are less likely to take actions randomly and more likely to use our improving estimates of the Q-function to pick good actions. For example, in the lunar lander exercise, we might start off with Epsilon very, very high, maybe even Epsilon equals 1.0. we're just picking actions completely at random initially and then gradually decrease it all the way down to say 0.01, so that eventually we're taking greedy actions 99 percent of the time and acting randomly only a very small one percent of the time. 

If this seems complicated, don't worry about it. We'll provide the code in the Jupiter lab that shows we how to do this. If we were to implement the algorithm as we've described it with the more efficient neural network architecture and with an Epsilon-greedy exploration policy, we find that they work pretty well on the lunar lander. 

One of the things that I've noticed for reinforcement learning algorithm is that compared to supervised learning, they're more finicky in terms of the choice of hyper parameters. For example, in supervised learning, if we set the learning rate a little bit too small, then maybe the algorithm will take longer to learn. Maybe it takes three times as long to train, which is annoying, but maybe not that bad. 

Whereas in reinforcement learning, find that if we set the value of Epsilon not quite as well, or set other parameters not quite as well, it doesn't take three times as long to learn. It may take 10 times or a 100 times as long to learn. Reinforcement learning algorithms, I think because they're are less mature than supervised learning algorithms, are much more finicky to little choices of parameters like that, and it actually sometimes is frankly more frustrating to tune these parameters with reinforcement learning algorithm compared to a supervised learning algorithm. 

But again, if we're worried about the practice lab, the program exercise, we'll give we a sense of good parameters to use in the program exercise so that we should be able to do that and successfully learn the lunar lander, hopefully without too many problems. In the next optional section, I want us to drive a couple more algorithm refinements, mini batching, and also using soft updates. Even without these additional refinements, the algorithm will work okay, but these are additional refinements that make the algorithm run much faster. 

It's okay if we skip this section, we've provided everything we need in the practice lab to hopefully successfully complete it. But if we're interested in learning about more of these details of two named reinforcement learning algorithms, then come with $me$ and let's see in the next section, mini batching and soft updates. ## Algorithm refinement: Mini-batch and soft updates

In this section, we'll look at two further refinements to the reinforcement learning algorithm we've seen. 

The first idea is called using mini-batches, and this turns out to be an idea they can both speedup our reinforcement learning algorithm and it's also applicable to supervised learning. They can help we speed up our supervised learning algorithm as well, like training a neural network, or training a linear regression, or logistic regression model. The second idea we'll look at is soft updates, which it turns out will help our reinforcement learning algorithm do a better job to converge to a good solution. 

Let's take a look at mini-batches and soft updates. To understand mini-batches, let's just look at supervised learning to start. Here's the dataset of housing sizes and prices that we had seen way back in the first course of this specialization on using linear regression to predict housing prices. 

There we had come up with this cost function for the parameters $w$ and b, it was 1 over 2m, sum of the prediction minus the actual value y^​2. The gradient descent algorithm was to repeatedly update $w$ as $w$ minus the learning rate alpha times the partial derivative respect to $w$ of the cost J of wb, and similarly to update $b$ as follows. Let $me$ just take this definition of J of wb and substitute it in here. 

Now, when we looked at this example, way back when were starting to talk about linear regression and supervised learning, the training set size $m$ was pretty small. I think we had 47 training examples. But what if we have a very large training set? 

Say $m$ equals 100 million. There are many countries including the United States with over a 100 million housing units, and so a national census will give we a dataset that is this order of magnitude or size. The problem with this algorithm when our dataset is this big, is that every single step of gradient descent requires computing this average over 100 million examples, and this turns out to be very slow. 

Every step of gradient descent means we would compute this sum or this average over 100 million examples. Then we take one tiny gradient descent step and we go back and have to scan over our entire 100 million example dataset again to compute the derivative on the next step, they take another tiny gradient descent step and so on and so on. When the training set size is very large, this gradient descent algorithm turns out to be quite slow. 

The idea of mini-batch gradient descent is to not use all 100 million training examples on every single iteration through this loop. Instead, we may pick a smaller number, let $me$ call it $m$ prime equals say, 1,000. On every step, instead of using all 100 million examples, we would pick some subset of 1,000 or $m$ prime examples. 

This inner term becomes 1 over 2m prime is sum over sum $m$ prime examples. Now each iteration through gradient descent requires looking only at the 1,000 rather than 100 million examples, and every step takes much less time and just leads to a more efficient algorithm. What mini-batch gradient descent does is on the first iteration through the algorithm, may be it looks at that subset of the data. 

On the next iteration, maybe it looks at that subset of the data, and so on. For the third iteration and so on, so that every iteration is looking at just a subset of the data so each iteration runs much more quickly. To see why this might be a reasonable algorithm, here's the housing dataset. 

If on the first iteration we were to look at just say five examples, this is not the whole dataset but it's slightly representative of the string line we might want to fit in the end, and so taking one gradient descent step to make the algorithm better fit these five examples is okay. But then on the next iteration, we take a different five examples like that shown here. we take one gradient descent step using these five examples, and on the next iteration we use a different five examples and so on and so forth. 

we can scan through this list of examples from top to bottom. That would be one way. Another way would be if on every single iteration we just pick a totally different five examples to use. 

we might remember with batch gradient descent, if these are the contours of the cost function J. Then batch gradient descent would say, start here and take a step, take a step, take a step, take a step, take a step. Every step of gradient descent causes the parameters to reliably get closer to the global minimum of the cost function here in the middle. 

In contrast, mini-batch gradient descent or a mini-batch learning algorithm will do something like this. If we start here, then the first iteration uses just five examples. It'll hit in the right direction but maybe not the best gradient descent direction. 

Then the next iteration they may do that, the next iteration that, and that and sometimes just by chance, the five examples we chose may be an unlucky choice and even head in the wrong direction away from the global minimum, and so on and so forth. But on average, mini-batch gradient descent will tend toward the global minimum, not reliably and somewhat noisily, but every iteration is much more computationally inexpensive and so mini-batch learning or mini-batch gradient descent turns out to be a much faster algorithm when we have a very large training set. In fact, for supervised learning, where we have a very large training set, mini-batch learning or mini-batch gradient descent, or a mini-batch version with other optimization algorithms like Atom, is used more common than batch gradient descent. 

Going back to our reinforcement learning algorithm, this is the algorithm that we had seen previously. The mini-batch version of this would be, even if we have stored the 10,000 most recent tuples in the replay buffer, what we might choose to do is not use all 10,000 every time we train a model. Instead, what we might do is just take the subset. 

we might choose just 1,000 examples of these s, a, R of s, s prime tuples and use it to create just 1,000 training examples to train the neural network. It turns out that this will make each iteration of training a model a little bit more noisy but much faster and this will overall tend to speed up this reinforcement learning algorithm. That's how mini-batching can speed up both a supervised learning algorithm like linear regression as well as this reinforcement learning algorithm where we may use a mini-batch size of say, 1,000 examples, even if we store it away, 10,000 of these tuples in our replay buffer. 

Finally, there's one other refinement to the algorithm that can make it converge more reliably, which is, I've written out this step here of Set Q equals $Q_new$. But it turns out that this can make a very abrupt change to Q. If we train a new neural network to new, maybe just by chance is not a very good neural network. 

Maybe is even a little bit worse than the old one, then we just overwrote our Q function with a potentially worse noisy neural network. The soft update method helps to prevent $Q_new$ through just one unlucky step getting worse. In particular, the neural network Q will have some parameters, W and B, all the parameters for all the layers in the neural network. 

When we train the new neural network, we get some parameters $W_new$ and $B_new$. In the original algorithm S as described on that slide, we would set W to be equal to $W_new$ and B equals $B_new$. That's what set Q equals $Q_new$ means. 

With the soft update, what we do is instead Set W equals 0.01 times $W_new$ plus 0.99 times W. In other words, we're going to make W to be 99 percent the old version of W plus one percent of the new version $W_new$. This is called a soft update because whenever we train a new neural network $W_new$, we're only going to accept a little bit of the new value. 

As similarly, B equals 0.01 times $B_new$ plus 0.99 times B. These numbers, 0.01 and 0.99, these are hyperparameters that we could set, but it controls how aggressively we move W to $W_new$ and these two numbers are expected to add up to one. One extreme would be if we were to set W equals one times $W_new$ plus 0 times W, in which case, we're back to the original algorithm up here where we're just copying $W_new$ onto W. 

But a soft update allows we to make a more gradual change to Q or to the neural network parameters W and B that affect our current guess for the Q function Q of s, a. It turns out that using the soft update method causes the reinforcement learning algorithm to converge more reliably. It makes it less likely that the reinforcement learning algorithm will oscillate or divert or have other undesirable properties. 

With these two final refinements to the algorithm, mini-batching, which actually applies very well to supervise learning as well, not just reinforcement learning, as well as the idea of soft updates, we should be able to get our lunar algorithm to work really well on the Lunar Lander. The Lunar Lander is actually a decently complex, decently challenging application and so that we can get it to work and land safely on the moon. I think that's actually really cool and I hope we enjoy playing with the practice lab. 

Now, we've talked a lot about reinforcement learning. Before we wrap up, I'd like to share with we $my$ thoughts on the state of reinforcement learning so that as we go out and build applications using different machine learning techniques via supervised, unsupervised, reinforcement learning techniques that we have a framework for understanding where reinforcement learning fits in to the world of machine learning today. Let's go take a look at that in the next section. 

## The state of reinforcement learning

Reinforcement learning is an exciting set of technologies. In fact, when I was working on $my$ PhD thesis reinforcement learning was the subject of $my$ thesis. So I was and still am excited about these ideas. 

Despite all the research momentum and excitement behind reinforcement learning though, I think there is a bit or maybe sometimes a lot of hype around it. So what I hope to do is share with we a practical sense of where reinforcement learning is today in terms of its utility for applications. One of the reasons for some of the hype about reinforcement learning is, it turns out many of the research publications have been on simulated environments. 

And having worked in both simulations and on real robots myself, I can tell we that it's much easier to get a reinforcement learning album to work in a simulation or in a section game than in a real robot. So a lot of developers have commented that even after they got it to work in simulation, it turned out to be surprisingly challenging to get something to work in the real world or the real robot. And so if we apply these algorithms to a real application, this is one limitation that I hope we pay attention to to make sure what we do does work on the real application. 

Second, despite all the media coverage about reinforcement learning, today there are far fewer applications of reinforcement learning than supervised and unsupervised learning. So if we are building a practical application, the odds that we will find supervised learning or unsupervised learning useful or the right tool for the job, is much higher than the odds that we would end up using reinforcement learning. I have used reinforcement learning a few times myself especially on robotic control applications, but in $my$ day to day applied work, I end up using supervised and unsupervised learning much more. 

There is a lot of exciting research in reinforcement learning right now, and I think the potential of reinforcement learning for future applications is very large. And reinforcement learning still remains one of the major pillars of machine learning. And so having it as a framework as we develop our own machine learning algorithms, I hope will make we more effective at building working machine learning systems as well. 

So I hope we've enjoyed this week's materials on reinforcement learning, and specifically I hope we have fun getting the lunar lander to land for yourself. I hope will be a satisfying experience when we implement an algorithm and then see that lunar lander land safely on the moon because of code that we wrote. That brings us towards the end of this specialization. 

Let's go on to the last section where we wrap up.