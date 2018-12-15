import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

#Load movielens ratings from .csv file into a dataframe
data = pd.read_csv("ml-latest-small//ratings.csv", header = None)

#Drop the 'timestamps' as we dont need them
data = data.drop([3], axis=1)

#Pivot the data to be listed by user
data_pivot = pd.DataFrame(data).pivot_table(index=0, columns=1, values=2, aggfunc='first').fillna(0)

#Create training and testing sets
trX, teX = train_test_split(data_pivot, test_size=0.2)

#Number of nodes per layer
input_nodes = 9725
hidden_nodes = 256
output_nodes = 9725

#Creating the network layers
hidden_layer = {'weights':tf.Variable(tf.random_normal([input_nodes+1,hidden_nodes]))}
output_layer = {'weights':tf.Variable(tf.random_normal([hidden_nodes+1,output_nodes]))}

#Input layer with 1 rating per movie
input_layer = tf.placeholder('float', [None, 9725])

#Adding a bias node to the input layer
input_layer_const = tf.fill([tf.shape(input_layer)[0], 1], 1.0)
input_layer_concat = tf.concat([input_layer, input_layer_const], 1)

#Multiply the output of the input layer with the weight matrix
hidden_layer = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_layer['weights']))

#Adding a bias node to the hidden layer
hidden_layer_const = tf.fill([tf.shape(hidden_layer)[0], 1], 1.0)
hidden_layer_concat = tf.concat([hidden_layer, hidden_layer_const], 1)

#Multiply the output of the hidden layer with the weight matrix
output_layer = tf.matmul(hidden_layer_concat,output_layer['weights'])

#The original shape, used for error calculations
shape = tf.placeholder('float', [None, 9725])

#Cost function, learning rate, optimizer
rate = 0.1
cost = tf.reduce_mean(tf.square(output_layer - shape))
optimizer = tf.train.AdagradOptimizer(rate).minimize(cost)

#Initialize all Tensorflow variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Batch size, epochs, users
batch_size = 100
hm_epochs = 200
users = trX.shape[0]

for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    
    for i in range(int(users/batch_size)):
        epoch_x = trX[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, cost],\
               feed_dict={input_layer: epoch_x, \
               shape: epoch_x})
        epoch_loss += c
        
    output_train = sess.run(output_layer,\
               feed_dict={input_layer:trX})
    output_test = sess.run(output_layer,\
                   feed_dict={input_layer:teX})
        
    print('MSE train', MSE(output_train, trX),'MSE test', MSE(output_test, teX))      
    print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
    
# pick a user
sample_user = teX.iloc[99,:]
#get the predicted ratings
sample_user_pred = sess.run(output_layer, feed_dict={input_layer:[sample_user]})