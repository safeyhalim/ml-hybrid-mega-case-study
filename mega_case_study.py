# Mega Case Study - A Hybrid Deep Learning Model

# Part 1 - Identifying the credit card frauds with the Self-Organizing Map (SOM)

# Unsupervised Deep Learning with Self Organizing Map (SOM)
# used for fraud detection among bank customers (Credit Card applications from customers)

# Importing the libraries
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset (the dataset is taken from: https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval))
dataset = pd.read_csv('Credit_Card_Applications.csv')
print(dataset)
# We are going to create subsets from the dataset. The first will be the whole 
# dataset without the last column. The last column is the "class" of the customer
# which shows whether her application for the credit card was actually approved or not
# We want to separate that column from our input to the SOM 
# IMPORTANT: the SOM we are going to create will NOT predict Y because what we are doing
# is unsupervised learning. We are only separating Y because it's irrelevant to the model
X = dataset.iloc[:, :-1].values # get the dataset except for the last column
Y = dataset.iloc[:, -1].values # the last column (the class)

# Feature Scaling
# We will use Normalization (making all values between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training SOM 
# Self Organizaing Map doesn't have a ready made implementation in scikitlearn
# We are going to use an existing implementation of the Self Organizaing Maps: MiniSom 2.2.9 (the latest version at the time of writing)
# https://pypi.org/project/MiniSom/
# It was possible to installing this library with pip install, but following the tutorial in the course,
# I chose to just add the minisom.py file to the project's working directly and work with it.
from minisom import MiniSom

# x, y in the argument list are the dimensions of the self organizing map. We chose x = 10 and y = 10 (10x10 grid) because it yields good results (also because our dataset is not that large)
# We are free to choose the dimensions of the SOM so that the map fits our input space
# input_len: the number of features we have in the input dataset(we have 14 features and the customer ID
# which we are going to keep in order to determine the cheaters. So in total 15)
# sigma: The radius of the neighborhood of the grid (We will keep the default value: 1.0)
# learning_rate: decides by how much the weights are updated during the learning process in each iteration. The higher the learning_rate the faster there will be convergence
# we will keep the default value: 0.5
# decay_function: used to improve the convergence. We won't use it (We should be fine with the other hyperparameters)
# random_seed: We won't use this hyperparameter
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# Check the following steps in the SOM lecture notes
som.random_weights_init(X)
# num_iteration: the number of iterations we won't to apply steps 4 to 9 in the lecture notes
# we chose 100 empirically (it's enough to yielding good results)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results:
# The grid will show the winning nodes, and for each we are going to show the MID (Mean Inter-neuron Distance)
# which is the mean of the distance between the winning node and it's neighboring nodes in its neighborhood
# For a certain winning node, the more the MID the more the likelihood that this winning node is an outlier
# Since in our example the majority of the winning node kinda represents the rules that are respected
# when applying for a credit card, an outlier winning node represents a likely fradulent applicant
# We are going to identify that using the colors on the map. The winning node will have colors so that the larger
# the MID, the closet to white its color will be.
from pylab import bone, pcolor, colorbar, plot, show # pylab is a matplotlib module
bone() # creates a window that will contain the map
pcolor(som.distance_map().T) # som.distance_map() returns a matrix of MID for all the winning nodes. We need to apply the transpose method T to fit it to the pcolor method
colorbar()
markers = ['o', 's'] # Create two markers o: circle, and s: square
colors = ['r', 'g'] # Create two colors r: red, and g: green
for i, x in enumerate(X): # i will be the indexes in X, and x will be the row of that index in the dataset (associated with the customer)
    w = som.winner(x) # returns the winning node associated with the customer. The winning node is the square in the plotted map
    plot(w[0] + 0.5, w[1] + 0.5, # w[0] and w[1] represent the lower left corner of the square representing the winning node in the plot. We want the maker to be displayed in the middle of the square, therefore we add 0.5
        markers[Y[i]], # if Y[i] = 0 that means that the customer's application was not approved, so we want to display a red circle in this case, if Y[1] mean approved and we want to display a green square
        markeredgecolor=colors[Y[i]], markerfacecolor = 'None', # we don't want to color the markers, because for some squares (winning nodes), there are some customers who got approved and other who didn't: in the same winning node square, we can find both red circles and green squares
        markersize = 10,
        markeredgewidth = 2)
    
show()

# Finding the frauds
mappings = som.win_map(data = X) # this method returns a mapping of all winning nodes to their associated data points in the dataset. The data returned is in the form of a dictionary where the key is a tuple that represents the coordinates of the winning node, and the value is the list of customers that are associated to that winning node
# In the course, there were two squares whose color was white.
# The coordinates of these squares are (8, 1) and (6, 8) respectively (the coordinates of a square are the ones of the bottom left point of the square)
# White squares reflect high MID and therefore potentially contain the cheating customers
# Note: that for every run of this code, the squares with white color change. But we will stick 
# to the squares (the coordinates) that were mentioned in the course.
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis = 0) # concatenating the list of customers associated with the two winning nodes vertically
# after the concatenation the list in frauds acutally contain the customers who are *potentially* frauds
frauds = sc.inverse_transform(frauds) # Since we scaled (normalized) the data set, we need to return the data associated with the list potentially fraudulent customers to its original form, so we do an inverse transform
# after descaling, the first column is that of the customer IDs
# We can therefore give this list of IDs to the bank employees so that they can conduct further investigation to know which customers actualy cheated although they were granted the credit card