
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
  Once you have python and the proper libraries installed, simply download the ML_models folder, navigate to it via the command line, and use the following command, filling in the name of the program you wish to run:
  
  ```
  python <program_file_name>
  ```
  
  After the program has finished running, the accuracy of the algorithm will be printed. For classification tasks, this is simply the number of correctly classified samples to     the total number of classified samples. For regression tasks, this is the root mean squared error.
  
  ## Using different data sets
  Feel free to use your own data sets! Just add them to the ML_models folder and add a couple lines of code to replace the lines below, which can be found easily by searching for "X_train, y_train =":
  
  ```
  X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
  X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')
  ```
 
 X_train and X_test should be of the form (samples x features), while y_train and y_test should be of the form (labels x 1). 
  

</details>
