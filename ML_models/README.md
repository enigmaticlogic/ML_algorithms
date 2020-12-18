
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
  
  ## Using Different Data Sets and Parameters
  Feel free to use your own data sets! Just add them to the ML_models folder and add a couple lines of code to replace the lines below, which can be found easily by searching for "X_train, y_train =":
  
  ```
  X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
  X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
  ```
 
 X_train and X_test should be of the form (samples x features), while y_train and y_test should be of the form (labels x 1). Guides on how to change parameters and what they mean will be included for each algorithm.
  

</details>

<details>
  <summary>Linear Regression</summary>
  
  ## Where it all Began...
  Linear regression predicts the label for a piece of data as a linear combination of its feature values. This is not a classification task but rather attempts to predict real number values, and thus I use the root mean squared error to measure accuracy. The pearson correlation coefficient is another popular accuracy metric. The bread and butter of these types of algorithms revolve around using calculus to minimize the value of a loss function which, when evaluated for all predicted values, provides some notion of total error. My implementation includes a variation known as ridge regression as well.
  
  ## Required Libraries
  The following Python libraries are required for this program: Numpy and Pandas.
  
  ## Parameters 
  lr: Learning rate, a number specifying how much to update the loss function at each gradient descent step.
  mode: Choose between ordinary linear regression ('OLR') and ridge regression ('Ridge').
  reg: Regularization constant, a number specifying how heavily to weight the regularization term (only needed for the ridge regression mode).
  
  
  Parameters can be adjusted in the line below. Feel free to experiment!
  
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
  
  ## Parameters 
  lr: Learning rate, a number specifying how much to update the loss function at each gradient descent step.
  reg: Regularization constant, a number specifying how heavily to weight the regularization term.
  
  
  Parameters can be adjusted in the line below. Feel free to experiment!
  
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
  
  ## Parameters 
  k: The number of neighbors considered, an integer. 
  mode: The distance function used, either euclidean ('eucl') or manhattan ('manhattan').
  
  
  Parameters can be adjusted in the line below. Feel free to experiment!
  
  ```
  my_kNN = kNN(X_train_norm, y_train, X_test_norm, k=3, mode='manhattan')
  ```
  
</details>
