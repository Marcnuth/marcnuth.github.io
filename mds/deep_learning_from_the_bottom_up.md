# 从0到1了解深度学习[Deep Learning from the Bottom Up]

> 本文由marcnuth进行翻译，部分内容有删改。英文原文地址: https://metacademy.org/roadmaps/rgrosse/deep_learning


![](https://i.imgur.com/SOjew3N.png)

在机器学习的应用中，花费时间最多和最容易吃力不讨好的过程在于去找到能解
释原始数据特质的“好”的特征。深度学习作为一个最近很火的研究领域，目的就
是想跨过复杂痛苦的特征处理过程，直接从原始数据建立模型。大多数深度学习
算法/模型都是建立在神经网络的基础上。而神经网络，虽然只是由许多简单地计算
非线性方程的神经元组成，但是它却可以用来解释高度复杂的模型。

大家可以参考
[Geoff Hinton的Coursera课程](https://www.coursera.org/course/neuralnets)
去学习神经网络，在Geoff的课上，你将会学到神经网络的核心思想，并且可以
自己动手完成一些简单的算法模型。(Geoff是该领域内的先驱人物，他创建了神
经网络的基础，并影响了大量的后续工作。)

学会基础知识是一回事，但是真正把它应用起来又是另外一回事。当你踏入这个
领域，你会发现，你并不能简单的把你的数据拿过来扔进去，然后期待它跑出一
个好结果。你可以需要不断地去调试和思考：模型过拟合了吗？针对模型的优化
程序起作用了吗？是不是应该添加更多神经元，更多的层？不幸的是，现在没有
任何人能告诉你这些问题明确的答案，你需要做的是，不断的试验、反思和调优。
**因此，理解神经网络算法的本质就显得尤为重要，尤其是你需要思考，神经网络是如何和
机器学习其他子领域内的概念/算法联系在一起的？下面有一张联系图，其中展
示了一些算法的关联关系，可以帮助你更好地理解和学习。**

![](https://i.imgur.com/qu8guyu.png)

另外，这些文章也值得一看，他们的综述对于初学者很有帮助，并且还谈到了神
经网络的最新进展:
 * Y. Bengio. [Learning deep architectures for AI.](https://wiki.eecs.yorku.ca/course_archive/2012-13/F/6328/_media/learning-deep-ai.pdf) Foundations and Trends in Machine Learning, 2009.
 * Y. Bengio, A. Courville, and P. Vincent. [Representation learning: a review and new perspectives.](http://arxiv.org/pdf/1206.5538.pdf) 2014


# 监督模型

If you’re interested in using neural nets, it’s likely that you want to automatically predict something. Supervised learning is a machine learning framework where you have a particular task you’d like the computer to solve, and a training set where the correct predictions are labeled. For instance, you might want to automatically classify email messages as spam or not-spam, and in supervised learning, you have a  dataset of 100,000 emails labeled as "spam" or "not spam" that you use to train your classifier so it can classify new emails it has never seen before.

Before diving into neural nets, you'll first want to be familiar with “shallow” machine learning algorithms, such as [[linear regression]], [[logistic regression]], and [support vector machines (SVMs)](support_vector_machine). 
These are far easier to implement, and there also exist pretty good software packages (e.g. [scikit.learn](http://scikit-learn.org/stable/)). They serve as a sanity check for your neural net implementations: you should at least be able to beat these simple generic approaches. Plus, neural nets are built out of simple units which are closely related to these models. Therefore, by taking the time to learn about these, you automatically gain a deeper understanding of neural nets.

In order to have any hope of doing supervised learning, you need to understand the idea of [generalization](generalization), the ability to make good predictions on novel examples. You’ll need to understand how to balance the tradeoff between underfitting and overfitting: you want your model to be expressive enough to model relevant aspects of the data, but not so complex that it “overfits” by modeling all the idiosyncrasies. In the case of regression, this can be formalized in terms of [bias and variance](bias_variance_decomposition), which provides a useful intuition more generally. You should be able to measure generalization performance using [cross-validation](cross_validation).


The vanilla deep learning model is the [feed-forward neural net](feed_forward_neural_nets), which is trained with [backpropagation](backpropagation).

<img width="250px" class="center-image" src="https://i.imgur.com/jXmajdl.png" alt="feed-forward network">


Vision is one of the major application areas of deep learning, and [convolutional nets](convolutional_nets) have been applied there with tremendous success. 

[Recurrent neural nets](recurrent_neural_networks) are a kind of neural net model for data with temporal structure. Backpropagation through time is an elegant training algorithm, but it's a beast to get to work in practice.

# Unsupervised models

In supervised learning, you have data labeled with the correct predictions for a particular task. But in many cases, labeled data is hard to obtain, or the correct behavior is hard to define. All you have is a lot of unlabeled data. This is the setting known as unsupervised learning. For instance, you may want to classify emails as "spam" or "not spam" but you don't have a dataset of labeled emails -- you only have the emails without spam/not-spam labels.

What can you do with unlabeled data?  One thing you can do is simply look for patterns. Maybe your data is explainable in terms of a small number of underlying factors, or dimensions. This can be captured with [principal component analysis](principal_component_analysis) or [factor analysis](factor_analysis). Or maybe you think the data are better explained in terms of clusters, where data points within a cluster are more similar than data points in different clusters. This can be captured with [k-means](k_means) or [mixture of Gaussians](mixture_of_gaussians).

In the context of neural nets, there is another reason to care about unsupervised learning: it can often help you solve a supervised task better. In particular, unlabeled data is often much easier to obtain than labeled data. E.g., if you’re working on object recognition, labeling the objects in images is a laborious task, whereas unlabeled data includes the billions of images available on the Internet.

[Unsupervised pre-training](unsupervised_pre_training) has been shown to improve performance of supervised neural nets on a wide variety of tasks. The idea is that you start by training an unsupervised neural net on the unlabeled data (I’ll cover examples shortly), and then convert it to a supervised network with a similar architecture. As a result of having to model the data distribution, the network will be primed to pick up relevant structure. Also, for reasons that are still not very well understood, **deep unsupervised models are often easier to train than deep supervised ones**. Initializing from an unsupervised network helps the optimizer avoid local optima.

The evidence for generative pre-training is still mixed, and many of the most successful applications of deep neural nets have avoided it entirely, especially in the big data setting. But it has a good enough track record that it is worth being aware of.

So what are these unsupervised neural nets?  The most basic one is probably the [autoencoder](http://www.sciencemag.org/content/313/5786/504.short), which is a feed-forward neural net which tries to predict its own input. While this isn’t exactly the world’s hardest prediction task, one makes it hard by somehow constraining the network. Often, this is done by introducing a bottleneck, where one or more of the hidden layers has much lower dimensionality than the inputs.
Alternatively, one can constrain the hidden layer activations to be [sparse](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2006_804.pdf) (i.e. each unit activates only rarely), or feed the network corrupted versions of its inputs and make it reconstruct the clean ones (this is known as a [denoising autoencoder](http://machinelearning.org/archive/icml2008/papers/592.pdf)). 

Another approach to unsupervised learning is known as generative modeling. Here, one assumes the data are drawn from some underlying distribution, and attempts to model the distribution. [Restricted Boltzmann machines (RBMs)](restricted_boltzmann_machines) are a simple generative neural network with a single hidden layer. They can be stacked to form multilayer generative models, including [deep belief nets (DBNs)](deep_belief_networks) and [deep Boltzmann machines (DBMs)](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS09_SalakhutdinovH.pdf). There are a wide variety of variations on this basic idea, many of which are covered below. 

<img width="500px" class="center-image" src="https://i.imgur.com/7JJxhT7.png" alt="layerwise training">

DBMs can learn to model some pretty complex data distributions:

<img width="500px" class="center-image" src="https://i.imgur.com/JNFgTKR.png" alt="DBM samples">

Generative modeling is a deep and rich area, and you can find lots more examples in the [Bayesian machine learning roadmap](http://www.metacademy.org/roadmaps/rgrosse/bayesian_machine_learning).


# Optimization algorithms

You’ve defined your neural net architecture. How the heck do you train it?  The basic workhorse for neural net training is [stochastic gradient descent (SGD)](stochastic_gradient_descent), where one visits a single training example at a time (or a “minibatch” of training examples), and takes a small step to reduce the loss on those examples. This requires computing the [gradient](gradient) of the loss function, which can be done using [backpropagation](backpropagation). Be sure to [check your gradient computations](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization) with finite differences to make sure you’ve derived them correctly. SGD is conceptually simple and easy to implement, and with a bit of tuning, can work very well in practice.

There is a broad class of optimization problems known as [convex optimization](convex_optimization), where SGD and other local search algorithms are guaranteed to find the global optimum. This occurs because the function being optimized is "bowl shaped" (convex)
and local improvements in the optimization function work towards the global optimum.
Much of machine learning research is focused on trying to formulate things as convex optimization problems. Unfortunately, deep neural net training is usually not convex, so you are only guaranteed to find a local optimum. This is a bit disappointing, but ultimately it’s [something we can live with](http://videolectures.net/eml07_lecun_wia/). For most feed-forward networks and generative networks, the local optima tend to be pretty reasonable. (Recurrent neural nets are a different story — more on that below.)

A bigger problem than local optima is that the curvature of the loss function can be pretty extreme. While neural net training isn’t convex, the problem of curvature also shows up for convex problems, and many of the techniques for dealing with it are borrowed from convex optimization. As general background, it’s useful to read the following sections of Boyd and Vandenberghe’s book, [Convex Optimization](http://www.stanford.edu/~boyd/cvxbook/):

* [Sections 9.2-9.3](http://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=477) talk about gradient descent, the canonical first-order optimization method (i.e. a method which only uses first derivatives)
* [Section 9.5](http://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=498) talks about Newton's method, the canonical second-order optimization method (i.e. a method which accounts for second derivatives, or curvature)

While Newton’s method is very good at dealing with curvature, it is impractical for large-scale neural net training for two reasons. First, it is a batch method, so it requires visiting every training example in order to make a single step. Second, it requires constructing and inverting the Hessian matrix, whose dimension is the number of parameters. ([Matrix inversion](computing_matrix_inverses) is only practical up to tens of thousands of parameters, whereas neural nets typically have millions.) Still, it serves as an idealized second-order training method which one can try to approximate. Practical algorithms for doing so include:

* [conjugate gradient](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
* limited memory BFGS

Compared with most neural net models, training RBMs introduces another complication: computing the objective function requires computing the partition function, and computing the gradient requires performing [inference](inference_in_mrfs). Both of these problems are [intractable](complexity_of_inference). (This is true for [learning Markov random fields (MRFs)](mrf_parameter_learning) more generally.) [Contrastive divergence](http://learning.cs.toronto.edu/~hinton/csc2535/readings/nccd.pdf) and [persistent contrastive divergence](http://www.cs.utoronto.ca/~tijmen/pcd/pcd.pdf) are widely used approximations to the gradient which often work quite well in practice. Evaluating the models remains a difficult problem, though. One can [estimate the model likelihood](http://www.cs.utoronto.ca/~rsalakhu/papers/dbn_ais.pdf) using [annealed importance sampling](annealed_importance_sampling), but this is delicate, and failures in estimation tend to overstate the model's performance.

<img width="500px" class="center-image" src="https://i.imgur.com/lxpulRn.png" alt="contrastive divergence">

Even once you understand the math behind these algorithms, the devil's in the details. Here are some good practical guides for getting these algorithms to work in practice:

* G. Hinton. [A practical guide to training restricted Boltzmann machines.](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf) 2010.
* J. Martens and I. Sutskever. [Training deep and recurrent networks with Hessian-free optimization.](http://www.cs.utoronto.ca/~ilya/pubs/2012/HF_for_dnns_and_rnns.pdf) Neural Networks: Tricks of the Trade, 2012.
* Y. Bengio. [Practical recommendations for gradient-based training of deep architectures.](http://arxiv.org/pdf/1206.5533) Neural Networks: Tricks of the Trade, 2012.
* L. Bottou. [Stochastic gradient descent tricks.](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) Neural Networks: Tricks of the Trade, 2012.


# Other tricks

[TODO: dropout]

[TODO: rectified linear units]

[TODO: GPU implementation]


# Applications

## Vision

Computer vision has been one of the major application areas of neural nets and deep learning. As early as 1998, [convolutional nets](convolutional_nets) were [successfully applied](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) to recognizing handwritten digits, and the [MNIST handrwritten digit dataset](http://yann.lecun.com/exdb/mnist/) has long been a major benchmark for neural net research. More recently, convolutional nets made a big splash by significantly pushing forward the state of the art in [classifying between thousands of object categories](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). Vision was a large part of [DeepMind's](http://deepmind.com/) system which learned to [play Atari games](http://arxiv.org/pdf/1312.5602) using only the raw pixels.

There's also been lots of work on generative models of images. Various work has focused on [learning sparse representations](https://papers.nips.cc/paper/3313-sparse-deep-belief-net-model-for-visual-area-v2.pdf) and on [modeling the local covariance structure](http://papers.nips.cc/paper/4138-generating-more-realistic-images-using-gated-mrfs.pdf) of images. If you build a deep generative model with a [convolutional architecture](http://people.csail.mit.edu/rgrosse/icml09-cdbn.pdf), you can high-level feature representations of objects:

<img width="500px" class="center-image" src="https://i.imgur.com/twwWX3u.png" alt="convolutional DBN">

## Text

[TODO]

## Speech

[TODO]


# Software

* [Caffe](http://caffe.berkeleyvision.org/) is an increasingly popular deep learning software package designed for image-related tasks, e.g. object recognition. It's one of the fastest deep learning packages available -- it's written in C++ and CUDA.
* The [University of Toronto machine learning group](http://learning.cs.toronto.edu/index.shtml) has put together some nice GPU libraries for Python. [GNumPy](http://www.cs.toronto.edu/~tijmen/gnumpy.html) gives a NumPy-like wrapper for GPU arrays. It wraps around [Cudamat](https://github.com/cudamat/cudamat), a GPU linear algebra library, and [npmat](http://www.cs.toronto.edu/~ilya/npmat.py), which pretends to be a GPU on a CPU machine (for debugging).
* [PyLearn](https://github.com/lisa-lab/pylearn2/) is a neural net library developed by the [University of Montreal machine learning group](http://lisa.iro.umontreal.ca/index_en.html). It is intended for researchers, so it is built to be customizable and extendable.
* PyLearn is built on top of [Theano](http://deeplearning.net/software/theano/), a Python library for neural nets and related algorithms (also developed at Montreal), which provides symbolic differentiation and GPU support.
* If for some reason you hate Python, [Torch](http://torch.ch/) is a powerful machine learning library for Lua.

# Relationships with other machine learning techniques

Neural nets share non-obvious relationships with a variety of algorithms from the rest of machine learning. Understanding these relationships will help you decide when particular architectural decisions are appropriate.

Many neural net models can be seen as nonlinear generalizations of "shallow" models. Feed-forward neural nets are essentially nonlinear analogues of algorithms like [logistic regression](logistic_regression). Autoencoders can be seen as nonlinear analogues of dimensionality reduction algorithms like [PCA](principal_component_analysis). 

RBMs with all Gaussian units are [equivalent to Factor analysis](http://deeplearning.cs.cmu.edu/pdfs/Marks_Movellan.2001.pdf). RBMs can also be [generalized](https://papers.nips.cc/paper/2672-exponential-family-harmoniums-with-an-application-to-information-retrieval.pdf) to other [exponential family](exponential_families) distributions.

Kernel methods are another set of techniques for converting linear algorithms into nonlinear ones. There is actually a surprising relationship between neural nets and kernels: Bayesian neural nets converge to Gaussian processes (a kernelized regression model) in the limit of infinitely many hidden units. (See [Chapter 2](http://www.db.toronto.edu/~radford/ftp/thesis.pdf) of Radford Neal's Ph.D. thesis. Background: [Gaussian processes](gaussian_processes))



# Relationship with the brain

If these models are called "neural" nets, it's natural to ask whether they have anything to do with [how the brain works](https://www.youtube.com/watch?v=mlXzufEk-2E). In a certain sense, they don't: you can understand and apply the algorithms without knowing anything about neuroscience. Mathematically, feed-forward neural nets are just adaptive [basis function expansions](basis_function_expansions). But the connections do run pretty deep between practical machine learning and studies of the mind and brain. 

Unfortunately, Metacademy doesn't have any neuroscience content (yet!), so the background links in this section will be fairly incomplete. Doubly unfortunately, neuroscience and cognitive science seem not to have the same commitment to open access that machine learning does, so this section might only be useful if you have access to a university library.

When trying to draw parallels between learning algorithms and the brain, we need to be precise about what level we're talking about. In "The philosophy and the approach" (Chapter 1 of [Vision: a Computational Investigation](http://www.amazon.com/gp/product/0262514621/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=0262514621&linkCode=as2&tag=metacademy-20&linkId=YBDQETAAAIA6XCGN)), David Marr argued for explicitly separating different levels of analysis: computation, algorithms, and implementation. (This is worth reading, even if you read nothing else in this section.) While not all researchers agree with this way of partitioning things, it's useful to keep in mind when trying to understand exactly what someone is claiming.

## Neuroscience

Jeff Hawkins's book [On Intelligence](http://www.amazon.com/gp/product/0805078533/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=0805078533&linkCode=as2&tag=metacademy-20&linkId=THGXZGGMBPFLFVAV) aims to present a unifying picture of the computational role of the neocortex. While the theory itself is fairly speculative, the book is an engaging and accessible introduction to the structure of the cortex.

Many neural net models have learned similar response properties to neurons in the primary visual cortex (V1).

* Olshausen and Field's [sparse coding](http://redwood.psych.cornell.edu/papers/olshausen_field_nature_1996.pdf) model ([background](sparse_coding)) was the first to demonstrate that a purely statistical learning algorithm discovered filters similar to those of V1. (Whether or not this is a neural net is a matter of opinion.) Since then, a wide variety of representation learning algorithms based on seemingly different ideas have recovered similar representations.
* Other statistical models [TODO] have learned topological representations similar to the layout of cell types in V1.
* [Karklin and Lewicki](http://www.nature.com/nature/journal/v457/n7225/abs/nature07481.html) fit a more sophisticated statistical model which reproduced response properties of complex cells.
* While the connection between V1 and learned filters may seem tidy, Olshausen highlights a lot of things we [still don't understand about V1](https://redwood.berkeley.edu/bruno/CTBP/olshausen-field05.pdf).

For more on the neuroscience of the visual system, check out [Eye, Brain, and Vision](http://hubel.med.harvard.edu/book/bcontex.htm), a freely available book written by David Hubel, one of the pioneers who first studied V1. (Chapters 3, 4, and 5 are the most relevant.)

There have also been neural nets explicitly proposed as models of the brain. Riesenhuber and Poggio's [HMAX model](http://maxlab.neuro.georgetown.edu/hmax/) is a good example. Jim DiCarlo [found](http://papers.nips.cc/paper/4991-hierarchical-modular-optimization-of-convolutional-networks-achieves-representations-similar-to-macaque-it-and-human-ventral-stream) that deep convolutional networks yield neurons which behave similarly to those high up in the primate visual hierarchy.



## Cognitive science

It's not just at the level of neurons that researchers have tried to draw connections between the brain and neural nets. Cognitive science refers to the interdisciplinary study of thought processes, and can be thought of a study of the mind rather than the brain. [Connectionism](http://plato.stanford.edu/entries/connectionism/) is a branch of cognitive science, especially influential during the 1980s, which attempted to model high-level cognitive processes in terms of networks of neuron-like units. (Several of the most influential machine learning researchers came out of this tradition.) 

McClelland and Rumelhart's book Parallel Distributed Processing (volumes [1](http://mitpress.mit.edu/books/parallel-distributed-processing) and [2](http://mitpress.mit.edu/books/parallel-distributed-processing-0)) is the connectionist Bible. Other significant works in the field include: 

* J. McClelland and T. Rogers. [The parallel distributed processing approach to semantic cognition.](http://www.nature.com/nrn/journal/v4/n4/abs/nrn1076.html) Nature Reviews Neuroscience, 2003.
* [TODO]

One of the most perplexing questions about the brain is how neural systems can model the compositional structure of language. Linguists tend to model language in terms of recursive structures like grammars, which are very different from the representations used in most neural net research. Paul Smolensky and Geraldine Legendre's book [The Harmonic Mind](http://mitpress.mit.edu/books/harmonic-mind) presents a connectionist theory of language, where neurons implement a system of constraints between different linguistic features.
