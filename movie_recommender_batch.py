import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import tkinter as tk
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#Load movielens ratings from .csv file into a dataframe
data = pd.read_csv("ml-latest-small//ratings.csv", header = None)

#Drop the 'timestamps' as we dont need them
data = data.drop([3], axis=1)
data = data.drop(0)

#Pivot the data to be listed by user
data_pivot = pd.DataFrame(data).pivot_table(index=0, columns=1, values=2, aggfunc='first').fillna(0)

#Create training and testing sets
trX, teX = train_test_split(data_pivot, test_size=0.2)

#Number of nodes per layer
input_nodes = 9724
hidden_nodes = 256
output_nodes = 9724

#Batch size, epochs, users
#batch_size = 100
epochs = 200
users = trX.shape[0]

TrainMSE = []
TestMSE = []
rate = 0.1
#rates = np.array([0.001,0.01,0.1,0.5])
#rates = np.array([500])
#hiddenNodes = np.array([2,4,8])
batches = np.array([100])
#batches = np.array([5,25,50,100,250,500])

for batch_size in batches:
    print("Batch Size = ", batch_size)
    MSEtrain = np.array([])
    MSEtest = np.array([])
    
    #Creating the network layers
    hidden_layer = {'weights':tf.Variable(tf.random_normal([input_nodes+1,hidden_nodes]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([hidden_nodes+1,output_nodes]))}
    
    #Input layer with 1 rating per movie
    input_layer = tf.placeholder('float', [None, 9724])
    
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
    shape = tf.placeholder('float', [None, 9724])

    #Cost function, learning rate, optimizer
    #rate = 0.1
    cost = tf.reduce_mean(tf.square(output_layer - shape))
    optimizer = tf.train.AdagradOptimizer(rate).minimize(cost)
    
    #Initialize all Tensorflow variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for epoch in range(epochs):
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
        mseTrain = MSE(output_train,trX)
        mseTest = MSE(output_test,teX)
        MSEtrain = np.append(MSEtrain,mseTrain)
        MSEtest = np.append(MSEtest,mseTest)
        print('Epoch', epoch, '/', epochs, 'loss:',epoch_loss)
        print('MSE train:', mseTrain,'    MSE test:', mseTest)
    TrainMSE.append(MSEtrain)
    TestMSE.append(MSEtest)

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
One = plt.errorbar(np.arange(epochs),TrainMSE[0])#, color = 'C1', label = 'Batches = 5')
#Two = plt.errorbar(np.arange(epochs),TrainMSE[1], color = 'C2', label = 'Batches = 25')
#Three = plt.errorbar(np.arange(epochs),TrainMSE[2], color = 'C3', label = 'Batches = 50')
#Four = plt.errorbar(np.arange(epochs),TrainMSE[3], color = 'C4', label = 'Batches = 100')
#Five = plt.errorbar(np.arange(epochs),TrainMSE[4], color = 'C5', label = 'Batches = 250')
#Six = plt.errorbar(np.arange(epochs),TrainMSE[5], color = 'C6', label = 'Batches = 500')
####Seven = plt.errorbar(np.arange(epochs),TrainMSE[6], color = 'C7', label = 'Hidden Neurons = 2')
####Eight = plt.errorbar(np.arange(epochs),TrainMSE[7], color = 'C8', label = 'Hidden Neurons = 4')
####Nine = plt.errorbar(np.arange(epochs),TrainMSE[8], color = 'C9', label = 'Hidden Neurons = 8')
#####Ten = plt.errorbar(np.arange(epochs),TrainMSE[9], color = 'C7', label = 'Rate = 50.0')
#####Eleven = plt.errorbar(np.arange(epochs),TrainMSE[10], color = 'C8', label = 'Rate = 500')
#####
###One = plt.errorbar(np.arange(25),TrainMSE[0][0:25], color = 'C1', label = 'Hidden Neurons = 16')
###Two = plt.errorbar(np.arange(25),TrainMSE[1][0:25], color = 'C2', label = 'Hidden Neurons = 32')
###Three = plt.errorbar(np.arange(25),TrainMSE[2][0:25], color = 'C3', label = 'Hidden Neurons = 64')
###Four = plt.errorbar(np.arange(25),TrainMSE[3][0:25], color = 'C4', label = 'Hidden Neurons = 128')
###Five = plt.errorbar(np.arange(25),TrainMSE[4][0:25], color = 'C5', label = 'Hidden Neurons = 256')
###Six = plt.errorbar(np.arange(25),TrainMSE[5][0:25], color = 'C6', label = 'Hidden Neurons = 512')
######Ten = plt.errorbar(np.arange(50),TrainMSE[9][0:50], color = 'C7', label = 'Rate = 50')
######Eleven = plt.errorbar(np.arange(50),TrainMSE[10][0:50], color = 'C8', label = 'Rate = 500')
#####
plt.title("Training MSE vs. Epochs")
plt.xlabel("Number of Epochs")
plt.ylabel("MSE")
#plt.legend(handles = [One,Two,Three,Four,Five,Six], loc = 1)
#plt.legend(handles = [Four,Five,Six,Seven,Eight,Nine,Ten,Eleven], loc = 1)
plt.grid()
plt.show()
###
plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
One = plt.errorbar(np.arange(epochs),TestMSE[0])#, color = 'C1', label = 'Batches = 5')
#Two = plt.errorbar(np.arange(epochs),TestMSE[1], color = 'C2', label = 'Batches = 25')
#Three = plt.errorbar(np.arange(epochs),TestMSE[2], color = 'C3', label = 'Batches = 50')
#Four = plt.errorbar(np.arange(epochs),TestMSE[3], color = 'C4', label = 'Batches = 100')
#Five = plt.errorbar(np.arange(epochs),TestMSE[4], color = 'C5', label = 'Batches = 250')
#Six = plt.errorbar(np.arange(epochs),TestMSE[5], color = 'C6', label = 'Batches = 500')
#####Seven = plt.errorbar(np.arange(epochs),TestMSE[6], color = 'C1', label = 'Rate = 2.0')
#####Eight = plt.errorbar(np.arange(epochs),TestMSE[7], color = 'C6', label = 'Rate = 5.0')
#####Nine = plt.errorbar(np.arange(epochs),TestMSE[8], color = 'C9', label = 'Rate = 10')
#####Ten = plt.errorbar(np.arange(epochs),TestMSE[9], color = 'C7', label = 'Rate = 50')
#####Eleven = plt.errorbar(np.arange(epochs),TestMSE[10], color = 'C8', label = 'Rate = 500')
####
###One = plt.errorbar(np.arange(25),TestMSE[0][0:25], color = 'C1', label = 'Hidden Neurons = 16')
###Two = plt.errorbar(np.arange(25),TestMSE[1][0:25], color = 'C2', label = 'Hidden Neurons = 32')
###Three = plt.errorbar(np.arange(25),TestMSE[2][0:25], color = 'C3', label = 'Hidden Neurons = 64')
###Four = plt.errorbar(np.arange(25),TestMSE[3][0:25], color = 'C4', label = 'Hidden Neurons = 128')
###Five = plt.errorbar(np.arange(25),TestMSE[4][0:25], color = 'C5', label = 'Hidden Neurons = 256')
###Six = plt.errorbar(np.arange(25),TestMSE[5][0:25], color = 'C6', label = 'Hidden Neurons = 512')
####Ten = plt.errorbar(np.arange(50),TestMSE[9][0:50], color = 'C7', label = 'Rate = 50')
####Eleven = plt.errorbar(np.arange(50),TestMSE[10][0:50], color = 'C8', label = 'Rate = 500')
plt.title("Testing MSE vs. Epochs")
#plt.title("Testing MSE vs. Epochs (Learning Rate = 2.0, Hidden Neurons = 16)")
plt.xlabel("Number of Epochs")
plt.ylabel("MSE")
#plt.legend(handles = [One,Two,Three,Four,Five,Six], loc = 1)
#plt.legend(handles = [Four,Five,Six,Seven,Eight,Nine,Ten,Eleven], loc = 1)
plt.grid()
plt.show()

##Rating window
#window = tk.Tk()
#window.title("Movie Recommender")
#
##Load movie names
#movies = pd.read_csv("ml-latest-small//movies.csv", header = None)
#movies = movies.drop([2], axis=1)
#movies = movies.drop(0)
#movies = np.asarray(movies[1])
#
##Create new user
#user = np.zeros((9724,), dtype=np.float)
#
#index = tk.Variable()
#index.set(random.randint(0, 9724))
#
#rating = tk.Variable()
#rating.set(0)
#
#movieName = tk.StringVar()
#movieName.set(movies[index.get()])
#
#frame1 = tk.Frame(window)
#frame1.pack()
#frame2 = tk.Frame(window)
#frame2.pack()
#frame3 = tk.Frame(window)
#frame3.pack()
#frame4 = tk.Frame(window)
#frame4.pack()
#frame5 = tk.Frame(window)
#frame5.pack()
#
#questionLabel = tk.Label(frame1, text="Please rate this movie:", font=10).pack()
#
#movieLabel = tk.Label(frame2, textvariable=movieName, font=10).pack()
#
#ratingLabel1 = tk.Label(frame3, text = "1", font=10)
#ratingLabel1.grid(row = 0, column = 0)
#
#ratingLabel1_5 = tk.Label(frame3, text = "1.5", font=10)
#ratingLabel1_5.grid(row = 0, column = 1)
#
#ratingLabel2 = tk.Label(frame3, text = "2", font=10)
#ratingLabel2.grid(row = 0, column = 2)
#
#ratingLabel2_5 = tk.Label(frame3, text = "2.5", font=10)
#ratingLabel2_5.grid(row = 0, column = 3)
#
#ratingLabel3 = tk.Label(frame3, text = "3", font=10)
#ratingLabel3.grid(row = 0, column = 4)
#
#ratingLabel3_5 = tk.Label(frame3, text = "3.5", font=10)
#ratingLabel3_5.grid(row = 0, column = 5)
#
#ratingLabel4 = tk.Label(frame3, text = "4", font=10)
#ratingLabel4.grid(row = 0, column = 6)
#
#ratingLabel4_5 = tk.Label(frame3, text = "4.5", font=10)
#ratingLabel4_5.grid(row = 0, column = 7)
#
#ratingLabel5 = tk.Label(frame3, text = "5", font=10)
#ratingLabel5.grid(row = 0, column = 8)
#
#check1 = tk.Radiobutton(frame3, value = 1, width=1, variable=rating)
#check1.grid(row = 1, column = 0)
#
#check1_5 = tk.Radiobutton(frame3, value = 1.5, width=1, variable=rating)
#check1_5.grid(row = 1, column = 1)
#
#check2 = tk.Radiobutton(frame3, value = 2, width=1, variable=rating)
#check2.grid(row = 1, column = 2)
#
#check2_5 = tk.Radiobutton(frame3, value = 2.5, width=1, variable=rating)
#check2_5.grid(row = 1, column = 3)
#
#check3 = tk.Radiobutton(frame3, value = 3, width=1, variable=rating)
#check3.grid(row = 1, column = 4)
#
#check3_5 = tk.Radiobutton(frame3, value = 3.5, width=1, variable=rating)
#check3_5.grid(row = 1, column = 5)
#
#check4 = tk.Radiobutton(frame3, value = 4, width=1, variable=rating)
#check4.grid(row = 1, column = 6)
#
#check4_5 = tk.Radiobutton(frame3, value = 4.5, width=1, variable=rating)
#check4_5.grid(row = 1, column = 7)
#
#check5 = tk.Radiobutton(frame3, value = 5, width=1, variable=rating)
#check5.grid(row = 1, column = 8)
#
#def onNext():
#    user[index.get()] = rating.get()
#    index.set(random.randint(0, 9724))
#    movieName.set(movies[index.get()])
#    rating.set(0)
#    
#def onFinish():
#    user[index.get()] = rating.get()
#    window.destroy()
#
#nextButton = tk.Button(frame4, text="Next Movie", font=10, command = onNext).pack()
#
#finishedButton = tk.Button(frame5, text="Finished Rating", font=10, command = onFinish).pack()
#
#window.mainloop()
#
#user_pred = sess.run(output_layer, feed_dict={input_layer:[user]})
#user_pred = user_pred.reshape(9724,)
#
#top_five = user_pred[np.argsort(user_pred)[-5:]]
#indexes = []
#for i in range(5):
#    indexes.append(np.where(user_pred==top_five[i]))
#indexes = np.asarray(indexes).reshape(5,)
#
#movieNames = []
#for i in range(5):
#    movieNames.append(movies[indexes[i]])
#
##Final window with recommendations
#message = tk.Tk()
#message.title("Movie Recommender")
#
#mframe1 = tk.Frame(message).pack()
#mframe2 = tk.Frame(message).pack()
#mframe3 = tk.Frame(message).pack()
#mframe4 = tk.Frame(message).pack()
#mframe5 = tk.Frame(message).pack()
#mframe6 = tk.Frame(message).pack()
#mframe7 = tk.Frame(message).pack()
#
#textLabel = tk.Label(mframe1, text="Movies recommended for you:", font=10).pack()
#
#movie1_text = "1." + movieNames[0]
#movie1 = tk.Label(mframe2, text=movie1_text, font=10).pack()
#
#movie2_text = "2." + movieNames[1]
#movie2 = tk.Label(mframe3, text=movie2_text, font=10).pack()
#
#movie3_text = "3." + movieNames[2]
#movie3 = tk.Label(mframe4, text=movie3_text, font=10).pack()
#
#movie4_text = "4." + movieNames[3]
#movie4 = tk.Label(mframe5, text=movie4_text, font=10).pack()
#
#movie5_text = "5." + movieNames[4]
#movie5 = tk.Label(mframe6, text=movie5_text, font=10).pack()
#
#okButton = tk.Button(mframe7, text="OK", font=10, command = message.destroy).pack()
#
#message.mainloop()