# TensorFlow-Developer-Certification

![tensflow](https://user-images.githubusercontent.com/66157611/122498320-87b67a00-d00c-11eb-98b8-251375771c2f.png)

The main aim of the course is "HOW TO PASS TENSORFLOW DEVELOPER CERTIFICATION"

# 0- TensorFlow fundamentals 

* Introduction to tensors (creating tensors)
* Getting information from tensors (tensor attributes)
* Manipulating tensors (tensor operations)
* Tensors and NumPy
* Using @tf.function (a way to speed up your regular Python functions)
* Using GPUs with TensorFlow

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/00_tensorflow_fundamentals.ipynb

### MyKeyTakeAways

**1. What is Deep Learning?**
*Machine Learning is turning things(data) into numbers and finding patterns in those numbers. Deep learning is a subset of Machine Learning.*

**2. Why use Machine Learning(or deep learning)?**
 **Better reasone**: *For a complex problem, can you think of all the rules?*
 *"If you can build a **simple rule-based)**(maybe not very simple.)system that doesn't require machine learning, do that"*
 
 [Rules of ML by Google](https://developers.google.com/machine-learning/guides/rules-of-ml)
 
 **3. What deep learning is good for**

- **Problems with long lists of rules** --*when the traditional approach fails, machine learning/deep learning may help.*
- **Continually changing environments** — *deeplearning can adapt('learn') to new scenaries*.
- **Discovering insights within large collections of data**—*can you imagine trying to hand-craft rules for what 101 different kinds of food look like?*

**4. What deep learning is (typically) not good for**
- **When you need explainability**—*the patterns learned by a deep learning model are typically uninterpretable by a human.*
- **When the traditional approach is a better option**— *if you can accomplish what you need with a simple rule-based system.*
- **When errors are unacceptable**— *since the outputs of deep learning model aren't always predictable.*
- **When you don't have much data**—- *deep learnings models usually require a fairly large amount of data to produce great results.*

**5. Machine Learning Vs Deep Learning**
![Machine Learning vs Deep Learning](https://user-images.githubusercontent.com/66157611/127770179-daeecd4e-fc16-4622-98f4-f5dbde90a123.png)

**6. Alogrithms**

|**Machine Learning**|**Neural Network**|
|--------------------|------------------|
| Random Forest | Neural Networks|
| Naive Bayes | Fully Connected Neural Network|
|Nearest Neighbour| Convolutional Neural Network|
|Support Vector Machine| Recurrent Neural Network|
|....Many More| Transformer and ..many more|

**7. What are Neural Networks?**
![Neural Network](https://user-images.githubusercontent.com/66157611/127770385-a1d295f0-7e88-4792-b876-89aaed63bb5b.png)
![2021-06-23 (3)](https://user-images.githubusercontent.com/66157611/127770411-60cd2c7a-40e6-45d3-9a41-4478478d4a4e.png)
![2021-08-01 (4)](https://user-images.githubusercontent.com/66157611/127770554-72fc767c-ea23-4db0-9739-656bcd787da6.png)

**8. What is Deep Learning already being used for?**
* *Recommedation Engine*
* *Translation*
* *Speech Recognition*
* *Computer Vision*
* *Natural Language Processing(NLP)*

**9. What is and why use TensorFlow?**
- End-to-end platform for machine learning.
- Write fast deep learning code in Python/other accessible languages (able to run on a GPU/TPU)
- Able to access many pre-built deep learning models (TensorFlow Hub).
- Whole stack: preprocess data, model data, deploy model in your application
- Originally designed and used in-house by Google (now open-source).*


**10. What is Tensor?**
*It is a numerical representation of data in an aray*



> **Note: A scalar has no dimension, A vector has one dimension (ex: [[10,10]]), A matrix has two dimensions. A tensor has three or more dimensions**

**11. Creating tensors with TensorFlow and tf.Variable()**

*We can assign a num to tf.Variable()(changeable tensors)[e.g. by changeable_tensors.assign[0].assign(7)] and we can't assign  a num to tf.Constant()(unchangeable tensors)*

> 🔑**Note:** Rarely in practice will you need to decide whether to use tf.constant or tf.Variable to create tensors, as TensorFlow does this for you. However, if in doubt, use tf.constant and change it later if needed.

**12. Shuffle Dataset**

*Why we have to shuffle dataset?*

*The main reason we have to shuffle the train data, for example if have images dataset with length 15000. The first 10000 images have one same category and last 5000 has different category. Then if we train with only 10000 dataset to avoid overfitting. Then our model only learn one category. To train other category we have to shuffle the dataset*

 > **Rule 4:** "If both the global and the operation seed are set: Both seeds are used in conjunction to determine the random sequence".

**13. Tensor Attributes**

|**Attribute**|**Meaning**|**Code**|
|-------------|-----------|--------|
|Shape| The length (number of elements) of each of the dimensions of a tensor| `tensor.shape`|
|Rank| The number of tensor dimensions, A scalar has rank 0, a vector has rank 1, a matrix is rank 2, a tensor has rank n| `tensor.ndim`|
|Axis or dimension| A particular dimension of a tensor| `tensor[0], tensor[:,1]`..|
|Size| The total number of items in the tensor|`tf.size(tensor)`|

**14. Matrix Multiplication with Tensors**

*In machine learning, matrix multiplication is one of the most common tensor operations.*
[http://matrixmultiplication.xyz/](http://matrixmultiplication.xyz/)

[matrix-multiplying.html](https://mathsisfun.com/algebra/matrix-multiplying.html)

> There are two rules our tensors (or matrices) need to fulfill if we're going to matrix multiply them.
  - The inner dimensions must match.
  - The resulting matrix has the shape of the outer dimensions.

 ![2021-06-21](https://user-images.githubusercontent.com/66157611/127771382-5342f1d4-a4aa-4a05-bfb5-9e5d9f6d43ee.png)


> **Generally, when performing matrix multiplication on two tensors and one of the axes doesn't line up, you will transpose (rather than reshape ) one of the tensors to get satisfy the matrix multiplication rules.**

**15. Changing the datatype of a tensor**

 [mixed_precision](https://www.tensorflow.org/guide/mixed_precision)

 **16. Aggregating tensors**

  *Aggregating tensors = condensing them from multiple values down to a smaller amount of values.*

  *For **tf.math.reduce_std** and **tf.math.reduce_variance** input tensor must be in real or complex type. So, just convert your data to float before passing to these functions*

 *Try different functions from these below link.*

[tf.math](https://www.tensorflow.org/api_docs/python/tf/math/reduce_variance)

 **17. Squeezing a tensor.**
 
 What squeezing does is removes dimensions of size 1 from the shape of a tensor.

**18.One-Hot Encoding Tensors.**

 [https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)

**19. Tensors and NumPy**

> 🔑**Note:** One of the main differences between a TensorFlow tensor and a NumPy array is that a Tensor can be run on GPU or TPU (for faster numerical processing).

*****************************************************************************************************************************************************************************************
# 1 - Neural Network Regression With TensorFlow

* Build TensorFlow sequential models with multiple layers

* Prepare data for use with a machine learning model

* Learn the different components which make up a deep learning model (loss function, architecture, optimization function)

* Learn how to diagnose a regression problem (predicting a number) and build a neural network for it

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/01_neural_network_regression_with_tensorflow_video.ipynb

### MyKeyTakeaways

#### Evaluating a model
- [Trainable Params](https://deeplizard.com/learn/video/8d-9SnGt5E0)
- [Intro To DeepLearning](http://introtodeeplearning.com/)

**Regression Evaluation Metrics**
- Mean absolute error(MAE)
- Mean square error(MSE)
- Huber

[Preprocessing Data](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029)

[Standardize Data](https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02)

****************************************************************************************************************************************************************************************

# 2 - Neural Network Classification With TensorFlow

* Learn how to diagnose a classification problem (predicting whether something is one thing or another)

* Build, compile & train machine learning classification models using TensorFlow

* Build and train models for binary and multi-class classification

* Plot modelling performance metrics against each other

* Match input (training data shape) and output shapes (prediction data target)

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/02_neural_network_classification_with_tensorflow_video.ipynb

### MyKeyTakeAways

- http://karpathy.github.io/2019/04/25/recipe/
- https://www.ritchieng.com/
- https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
- https://numpy.org/doc/stable/reference/generated/numpy.c_.html
- https://cs231n.github.io/neural-networks-case-study/

**Classification Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

****************************************************************************************************************************************************************************************

# 3 -  Computer Vision and Convolutional Neural Networks with TensorFlow

* Build convolutional neural networks with Conv2D and pooling layers

* Learn how to diagnose different kinds of computer vision problems

* Learn to how to build computer vision neural networks

* Learn how to use real-world images with your computer vision models

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/03_introduction_to_computers_vision_with_tensorflow_video.ipynb

### MyKeyTakeAways

- https://poloclub.github.io/cnn-explainer/

*****************************************************************************************************************************************************************************************

# 4 - Transfer Learning with TensorFlow Part 1: Feature Extraction

* Learn how to use pre-trained models to extract features from your own data

* Learn how to use TensorFlow Hub for pre-trained models

* Learn how to use TensorBoard to compare the performance of several different models

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/04_transfer_learning_in_tensorflow_part_1_feature_extraction.ipynb

### MyKeyTakeAways

**What is transfer learning?**

Surely someone has spent the time crafting the right model for the job

Leverage our work, we will adapt the model which has already built on it like IMAGENET and use it on our model.

**Why use transfer learning?**

- Can leverage an existing neural network architecture proven to work on problems similar to our own.
- Can leverage a working network architecture which has already learned patterns on similar data to our own (often results in great results with less data)

**What are callbacks?**

- Callbacks are a tool which can add helpful functionality to your models during training, evaluation or inference.
- Some popular callbacks include:

![2021-07-19 (2)](https://user-images.githubusercontent.com/66157611/128628547-404c0a65-080d-49b6-9930-756a5cc10dec.png)

**What is TensorFlow Hub?**

- A place to find a plethora of pre-trained machine learning models (ready to be applied adn fine-tuned for your own problems)

![2021-07-20](https://user-images.githubusercontent.com/66157611/128628577-55d84449-8270-4c8b-82be-43b7d6ad1858.png)
![2021-07-20 (1)](https://user-images.githubusercontent.com/66157611/128628580-8e82a996-c9a2-4c03-95fe-0ca5763868b0.png)

*****************************************************************************************************************************************************************************************


# 5 — Transfer Learning with TensorFlow Part 2: Fine-tuning

* Learn how to setup and run several machine learning experiments

* Learn how to use data augmentation to increase the diversity of your training data

* Learn how to fine-tune a pre-trained model to your own custom problem

* Learn how to use Callbacks to add functionality to your model during training

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/05_transfer_learning_in_tensorflow_part_2_fine_tuning.ipynb

### MyKeyTakeAways

![2021-07-21](https://user-images.githubusercontent.com/66157611/128628646-b4234b9a-1660-4762-b419-744df660da21.png)

**What is a feature vector?**
* *A feature vector is **a learned representation of the input data**(a compresed form of the input data based on how the model see's it).

![2021-07-22 (2)](https://user-images.githubusercontent.com/66157611/128628700-202b183d-37b6-4590-b7f9-6fbd155c0084.png)

****************************************************************************************************************************************************************************************

# 6 — Transfer Learning with TensorFlow Part 3: Scaling Up (Food Vision mini)

* Learn how to scale up an existing model

* Learn to how evaluate your machine learning models by finding the most wrong predictions

* Beat the original Food101 paper using only 10% of the data

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/06_transfer_learning_in_tensorflow_part_3_scaling_up_video.ipynb

### MyKeyTakeAways

![2021-07-26](https://user-images.githubusercontent.com/66157611/128628782-6e4d7059-ee09-4094-9da7-37d8308ec790.png)
![2021-07-28](https://user-images.githubusercontent.com/66157611/128628795-a130634c-b3e2-446a-b11a-bc8a8b431afa.png)

*****************************************************************************************************************************************************************************************

# 7 — Milestone Project 1: Food Vision

* Combine everything you've learned in the previous 6 notebooks to build Food Vision: a computer vision model able to classify 101 different kinds of foods. Our model well and truly beats the original Food101 paper.

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/07_milestone_project_1_food_vision_video.ipynb

### MyKeyTakeAways

**What is TensorFlow Datasets(TFDS)?**

**TensorFlow Datasets** is a place for prepared and ready-to-use machine learning datasets.
* Why use TensorFlow Datasets?
   * Load data already in tensor format.
   * Practice on well established datasets(for many different problem types)
   * Experiment with different modelling techniques on a consistent datasets
   
**Why not use TensorFlow Datasets?**
* Datasets are static (do not change like  real-world datasets).

![2021-07-31](https://user-images.githubusercontent.com/66157611/128628944-d2d0635c-1bc5-4f3a-a121-6647d0c728b8.png)

*****************************************************************************************************************************************************************************************

# 8- NLP Fundamentals in TensorFlow
* Learn to:

     * Preprocess natural language text to be used with a neural network

     * Create word embeddings (numerical representations of text) with TensorFlow

     * Build neural networks capable of binary and multi-class classification using:

              *  RNNs (recurrent neural networks)

              *  LSTMs (long short-term memory cells)

              *  GRUs (gated recurrent units)

              * CNNs

* Learn how to evaluate your NLP models

**Notebook**: https://github.com/Parvez13/TensorFlow-Developer-Certification/blob/master/08_introduction_to_nlp_in_tensorflow.ipynb

### MyKeyTakeAways

* [How to Read Files in Python](https://realpython.com/read-write-files-python/ )
* [Working With JSON Files in Python](https://realpython.com/python-json/)
* [RNN from MIT](https://www.youtube.com/watch?v=SEnXr6v2ifU)
* http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* https://colah.github.io/posts/2015-08-Understanding-LSTMs/
* https://stackoverflow.com/questions/38080035/how-to-calculate-the-number-of-parameters-of-an-lstm-network
* https://medium.com/@priyadarshi.cse/calculating-number-of-parameters-in-a-lstm-unit-layer-7e491978e1e4
* https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
* https://en.wikipedia.org/wiki/Gated_recurrent_unit
* [Understanding Convolutional Neural Networks in Text classification](https://aclanthology.org/W18-5408.pdf)
* [Universal Sentence Encoder](https://aclanthology.org/D18-2029.pdf)
