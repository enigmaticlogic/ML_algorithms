
<details>
  <summary>Getting Started</summary>
  
  ## Intro
  Hello! I have designed this readme to be a beginner friendly introduction to running Python programs via the command line, as well as showing off the power of various machine learning algorithms I have coded as part of my own edification. Please feel free to contact me at David.Patrick.Storey@gmail.com if you have any questions.
  ## Anaconda
  <a href="http://anaconda.com/">Anaconda</a> is a great tool to help you get started with Python! It will help you install Python, manage packages, and integrate with Jupyter Notebooks. 
  ## Required Libraries
  To run any of my programs, you will need to have the correct libraries installed. In the section for each algorithm I will list all required libraries. To install them, you can use conda (through the Anaconda command prompt), or pip, the native Python installer. For example, to install Pandas you can execute the following in the Anaconda prompt:
  

```
conda install -c pytorch pytorch
```

  ## How to Use 
  Once you have python and the proper libraries installed, simply download the ML_models folder, navigate to it via the command line, and use the following command (filling in the name of the program you wish to run):
  
  ```
  python <program_file_name>
  ```
  
  After the program has finished running, the accuracy of the algorithm will be printed. For classification tasks, this is simply the number of correctly classified samples to     the total number of classified samples. For regression tasks, this is the root mean squared error.
  
  ## Using Different Data Sets and Hyperparameters
  Feel free to use your own data sets! Just add them to the ML_models folder and add a couple lines of code to replace the lines below, which can be found easily by searching for "X_train, y_train =":
  
  ```
  X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
  X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
  ```
 
 X_train and X_test should be of the form (samples x features), while y_train and y_test should be of the form (labels x 1). Guides on how to change hyperparameters and what they mean will be included for each algorithm.
 
 ## Common Terms
 In this section I will include some common terms used throughout the documentation.
 
 Sample: This refers to a single piece of data in the training or test set.  
 Label: This is what we are trying to predict. For example in the digits/MNIST dataset, the sample is an image of a hand drawn digit, and the label is the value of that digit.  
 Predictor: The predictor is a function of the feature values of a sample which outputs a prediction for what the label of that sample should be.  
 Loss Function: The loss function is used to determine the global error of our model. Typically, it is a function of the weights in our model and involves the predictions made on the training data. The goal is to minimize this function for the training data, which in turn gives a stronger prediction function.  
 Gradient Descent: This is a technique to minimize differentiable functions. The idea is to start with random or fixed inputs (the weights) and take the negative gradient of the loss function, which will give the direction of steepest decrease in the weight space. Then, we update the weights by moving a tiny bit in that direction, hopefully closer to a minimum. This process is repeated many times. Convex loss functions are desired to prevent becoming "stuck" in local minima, but there are more advanced methods such as stochastic gradient descent to combat this.  
 Epoch: An epoch is one iteration of optimization.  
 Training: Machine learning models must be "trained" on sample data. All of the models here perform what is called supervised learning, meaning that we have access to data with known labels. This known data is fed into our model and used to tune it, often by optimizing a loss function.  
  
</details>

<details>
  <summary>Linear Regression</summary>
  
  ## Where it all Began...
  Linear regression predicts the label for a piece of data as a linear combination of its feature values. This is not a classification task but rather attempts to predict real number values, and thus I use the root mean squared error to measure accuracy. The pearson correlation coefficient is another popular accuracy metric. The bread and butter of these types of algorithms revolve around using calculus to minimize the value of a loss function which, when evaluated for all predicted values, provides some notion of total error. My implementation includes a variation known as ridge regression as well.
  
  ## Required Libraries
  The following Python libraries are required for this program: Numpy and Pandas.
  
  ## Hyperparameters 
  lr: Learning rate, a number specifying how much to update the loss function at each gradient descent step.  
  mode: Choose between ordinary linear regression ('OLR') and ridge regression ('Ridge').  
  reg: Regularization constant, a number specifying how heavily to weight the regularization term (only needed for the ridge regression mode).  
  
  Hyperparameters can be adjusted in the line below. Feel free to experiment!
  
  ```
  my_Lin_Reg = Lin_Reg(X_train_norm, y_train, c, lr=0.01, mode='Ridge', reg=0.5) # Adjust parameters here
  ```
  
</details>

<details>
  <summary>Logistic Regression</summary>
  
  ## It's Classification Time!
  Logistic regression is used for classification tasks. Rather than predict the value of a sample, it uses the logistic function to predict the probability that a sample is in one of two categories. But what if we want more than two possible labels? Don't worry, there are a several ways that this problem can be solved! This is what is known as a multiclass classification problem, and the strategy we elect to use is called the one vs. all method. Essentially, we calculate the probability that the sample should have each label seperately, and then we choose the label with the highest probability as our prediction. Another problem is the non numerical nature of categorical data. How are we supposed to use a function to predict a non numerical value? Thankfully, we have a solution to this problem as well. We use a technique called one hot encoding to transform our labels into vectors. For example, if we had three labels, we could view them as the vectors [1,0,0], [0,1,0], and [0,0,1].
  
  ## Required Libraries
  The following Python libraries are required for this program: Numpy and Pandas.
  
  ## Hyperparameters 
  lr: Learning rate, a number specifying how much to update the loss function at each gradient descent step.  
  reg: Regularization constant, a number specifying how heavily to weight the regularization term.  
  
  Hyperparameters can be adjusted in the line below. Feel free to experiment!
  
  ```
  my_Log_Reg = Log_Reg(X_train_norm, y_train_ohe, c, lr=0.01, reg=0.5)
  ```
  
</details>

<details>
  <summary>K Nearest Neighbors</summary>
  
  ## Simple but Powerful!
  K nearest neighbors (or knn for short) is perhaps the easiest of these algorithms to understand, but can nonetheless be a very powerful classification tool. To begin, a notion of distance between data samples is defined (often just Euclidean or "physical" distance). Then, to classify a sample you pick a number (this is the k value) and find the k samples in the test set closest to it under the notion of distance chosen to the sample you want to classify. The predicted value is simply whatever the most common label in this subset is, and if there is a tie you choose randomly! Knn also has the added bonus of being easy to train, since the only adjustable parameter is the k value. Notice that this program uses the IRIS dataset which can be retrieved directly in the script using sklearn!
  
  ```
  iris_dataset = load_iris()
  X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)
  ```
  
  ## Required Libraries
  The following Python libraries are required for this program: Numpy, Pandas, Collections, and Sklearn.
  
  ## Hyperparameters 
  k: The number of neighbors considered, an integer.  
  mode: The distance function used, either euclidean ('eucl') or manhattan ('manhattan').  
  
  Hyperparameters can be adjusted in the line below. Feel free to experiment!
  
  ```
  my_kNN = kNN(X_train_norm, y_train, X_test_norm, k=3, mode='manhattan')
  ```
  
</details>

<details>
  <summary>Decision Tree</summary>
  
  ## Decisions, Decisions, Decisions...
  Note: Decision trees can also be used for regression (which my code includes an option for), but I only explain how they are used for classification here. 
  
  The way a decision tree classifies a sample is essentially a flowchart. The tree is comprised of many nodes, and the sample is passed from one node to the next until it reaches a so called leaf node. At each node, a series of conditions determine the next node the sample is passed to. These conditions could be anything, and are easiest to think about in the form of questions. For example, a node might ask "What color is the sample?" and branch out to 3 nodes representing red, blue, and green. One strength of decision trees is the ability to follow a sample down the tree and see the exact process by which it was classified! For numerical feature values, these conditions are often just inequalities. In my program, each node is either a leaf node or splits into two other nodes. A leaf node simply classifies a sample that reaches it, so it does not split into any more nodes. 
  
  But how does one choose when to stop splitting and designate a node as a leaf node, which conditions to use to split nodes, and how many times to branch nodes? To determine when a node should stop splitting and become a leaf node, we use something called the purity of the node. Essentially, we run our training data through the tree and if a large portion of the data at a node has the same label, that node is considered to be more pure than if there is an even mix. If a node ends up with all samples with the same label, that node is pure. There are several ways to measure purity, but we use the most common which is called the GINI score. If a node is completely pure, we designate it as a leaf node. The effectiveness of splitting conditions of a node are measured by something called GAIN, which is a function of the GINI score of the parent node and that of the nodes it splits to. Since there are a finite number of features, we can test every possible splitting condition for a single feature to find the best split for that feature, and repeat this process to find the best splitting condition across all features. It is by this process that the tree is generated, but the user still must decide how many times to branch (called the max depth), and typically will assign a minimum number of samples in a node to consider splitting it (for example, it doesn't make sense to split a node with only one sample in it). Due to these practicies, not all leaf nodes will be completely pure. If there is a tie for training samples classified in one of these nodes, the label designated by the node is chosen arbitrarily. The maximum depth and minimum split are hyper parameters that must be tuned by the user. Decision trees are often "pruned" by removing some nodes from the tree, but I will not cover pruning techniques here.
  
  ## Required Libraries
  The following Python libraries are required for this program: Numpy, Pandas, Scipy, and Sklearn.
  
  ## Hyperparameters 
  max_depth: The maximum number of layers the tree can have, an integer.  
  min_samples: The minimum number of training samples in a node for it to be considered for splitting, an integer.  
  mode: Set to 'classifier' for classification tasks and 'regressor' for regression tasks.  
  
  Hyperparameters can be adjusted in the line below. Feel free to experiment!
  
  ```
  myTree = DecisionTree(max_depth=11, min_samples_split=1, mode='classifier')
  ```
  
</details>

<details>
  <summary>Random Forest</summary>
  
  ## See the Forest for the Trees
  The random forest is an ensemble method built on the decision tree to reduce overfitting. It is a very powerful method used for both classification and regression that typically provides high accuracy while also being easier to tune than neural networks, so it is a great place for beginners to start digging in to model training. For this method, a large number of decision trees are built and a prediction is made by choosing the label predicted by the majority of these trees. 
  
  To build each of these trees, a number of samples are selected randomly from the sample set to run through the tree, and at each node a random subset of features are chosen to determine the splitting condition. Trees are typically much smaller in a random forest and are not pruned. 

  ## Required Libraries
  The following Python libraries are required for this program: Numpy, Pandas, Scipy, Collections, and Sklearn.
  
  ## Hyperparameters 
  num_trees: The number of trees generated, an integer.  
  num_samples: The number of samples used to construct each individual tree, an integer.  
  max_depth: The maximum number of layers each tree can have, an integer.  
  mode: Set to 'classifier' for classification tasks and 'regressor' for regression tasks.  
  
  Hyperparameters can be adjusted in the line below. Feel free to experiment!
  
  ```
  myForest = Random_Forest(X_train_norm, y_train, num_trees=1000, num_samples=100, max_depth=6, mode='classifier')
  ```
  
</details>

<details>
  <summary>Artificial Neural Network</summary>
  
  ## A Revolutionary Powerhouse
  The neural network is an infamous machine learning model that really propelled the rise of the big data era due to its extreme accuracy when large amounts of training data is available. It works by feeding a sample through a series of interconnected layers of "neurons". The input layer consists of one neuron for each feature, with values determined by the feature values. Then, each of these neurons is connected to the next layer, called a hidden layer. The value at each neuron in this hidden layer is determined in two steps. First, we take a weighted (the wieghts are trained) linear combination of the neuron values from the previous layer. Then we apply an activation function (used to add non linearity) to this linear combination and the result is the value of our neuron. In this way, each neuron is connected to all the neurons of the previous layer, and the weights determine how much each neuron in the previous layer contributes to the neurons of the current layer. This process is repeated, possibly many times for big data applications, with various numbers of neurons and activation functions at each layer. At the end of the network there is an output layer which typically consists of one neuron for each label category for classification tasks, or one neuron for regression tasks. A different activation function is used for the output layer, for example the softmax function is used for multiclass classification problems. Finally, an algorithm called backpropagation is used to update and optimize the weights. I won't go into detail here, but it essentially amounts to using the chain rule to perform gradient descent by accumulating global error one layer at a time. 
  
  The tricky part about neural networks is that they can be difficult to design and train, and often don't offer much of a performance boost without huge amounts of data. The number of layers, which activation function to use at each layer, and how many neurons are in each layer are all hyper parameters. To make matters more complicated, most sophisticated models involving neural networks combine them with other models, or use variations of the neural network which require even more tuning. For this reason, despite their potential predictive power, they are often not the first choise for initial exploration of data or for tasks where accuracy optimization isn't a priority.

  ## Required Libraries
  The following Python libraries are required for this program: Numpy, Pandas, and Pytorch (used for the ability to run off of GPU).
  
  ## Hyperparameters 
  num_hidden_layers: The number of hidden layers in the network, an integer.  
  num_hidden_neurons: The number of neurons in each layer, a list of length num_hidden_layers (e.g. [100,150] would set the first hiden layer to have 100 neurons, and the second to have 150 neurons).  
  epoch_num: The number of times to run backpropagation, an integer.  
  mode: Set to 'classifier' for classification tasks and 'regressor' for regression tasks.  
  reg: The regularization constant, a number.  
  lr: The learning rate, a number.  
  
  Hyperparameters can be adjusted in the line below. Feel free to experiment!
  
  ```
  myNN = whole_network(X_train_norm, y_train_ohe, num_hidden_layers=2, num_hidden_neurons=[100, 100], epoch_num=2500, reg=0.1, lr=0.01)
  ```
  
</details>

<details>
  <summary>Convolutional Neural Network</summary>
  
  ## Pictures Please!
  The convolutional neural network is a variation of the neural network which has shown a lot of success in image recognition problems. It has a similar structure of layers which an image is passed through, but in this case the output is often used as the input for an artificial network which makes the final prediction. Instead of hidden layers, there are alternating convolution layers and pooling layers. The convolution layers involve passing small filters (e.g. 3x3) over the input of the layer (the image, or in later layers a distorted version of it). As the filter is passed over the image, each patch is convolved with the filter, and the sum of these convolutions is taken and an activation fucntion is applied. Since there are multiple filters, we end up with multiple image outputs. The values of these filters are learned in the training process, and by looking at the output images of the first layer, one can see that the learned filters often perform specific functions such as blurring, embossing, and edge detection. The pooling layers downsample the outputs of the convolutional layers to further reduce the resolution and dimension of the image. This serves to reduce the number of parameters in the network, as well as reduce overfitting. After the image is fed through the convolution and max pooling layers, it is flattened into a vector and fed into an artifical neural network.
  
  ## Required Libraries
  The following Python libraries are required for this program: Numpy, Pandas, Scipy, Skimage, and Random.
  
  ## Hyperparameters 
  NOTE: My implementation is not well optimized, so it can take a while to run.

  hidden_nn_1 = The number of neurons in first layer of the ANN, an integer.  
  hidden_nn_2 = The number of neurons in the second layer of the ANN, an integer.  
  conv_h = The height of the convolution filters, an integer.  
  conv_w = The width of the convolution filters, an integer.  
  pool_h = The height of the pooling filter, an integer.  
  pool_w = The width of the pooling filter, an integer.  
  num_filters = The number of convolution filters to be applied in the convolution layer, an integer.  
  lr = The learning rate, a number.  
  reg = The regularization constant, an integer.  
  
  For the second two lines, you can choose how many epochs to run as well as how many samples to use for the mini batch gradient descent.
  
  Hyperparameters can be adjusted in the lines below. Feel free to experiment!
  
  ```
  my_NN = conv_NN(X_train_norm, y_train_ohe, hidden_nn_1=100, hidden_nn_2=100, conv_h=3, conv_w=3,\
                pool_h=2, pool_w=2, num_filters=16, lr=0.1, reg=.000001)
  . . .             
  for epoch in range(3000):
    my_NN.convolution_minibatch_fit(20)
  ```
  
</details>
