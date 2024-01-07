import tensorflow as tf
from keras import models 
from keras.models import Sequential
from keras.layers import Dense, Lambda
import math as mt
import numpy as np

class engine:

        epsilon=1
        model_trained=None
        
        def __init__(self, X_train, X_val, Y_train, Y_val ,epsilon):
            
            self.epsilon=epsilon
            self.model_trained=self.piin(X_train=X_train, X_val=X_val, Y_train=Y_train, Y_val=Y_val)
            
            pass

        def mexican_hat(x):
            
            return  (1 - x**2)*mt.e**(-(x^2)/2)

        def prediction(model, x):
            
            return model.predict(x)
        
        def Lf(self,prediction,Nf):
            
            return ((2 * prediction[3]+prediction[0]*prediction[2])**2)/Nf
        
        def Lb(self,prediction,Nf):
            
            return (((prediction[0]**2) + prediction[1])**2 + (prediction[1]-1)**2)/Nf
            
        def blasius_equation( prediction,self ,Nf ):
            
            L_pred=self.Lf(self,prediction,Nf ) + self.Lb(self,prediction,Nf)
            
            if L_pred < self.epsilon:
                print("sharp model")
            
            return L_pred   
        
        def decay_rate(self ,List_decay_epoch,init_rate,decay_step): # calculate the decay rate
            
            List = list((element[0]/init_rate)**(element[1]/decay_step) for element in List_decay_epoch)
            
            result=0
            for i in List:
                result+=i
            
            return result/len(List) 
        
        def piin(self, X_train, X_val, Y_train, Y_val):

            tf.random.set_seed(42)

            # Create a model similar to model_1 but add an extra layer and increase the number of hidden units in each layer
            model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat)
            ])
        
            initial_learning_rate = 0.01
            decay_steps = 3000
            decay_rate = self.decay_rate(List_decay_epoch=[[0.001,1000],[0.0005,3000]],init_rate=initial_learning_rate,decay_step=decay_steps)

            # Define the learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False)
            
            # Create an optimizer with the learning rate schedule
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
            # Compile the model
            model.compile(loss= lambda y_true , y_pred: self.blasius_equation(y_pred,self,Nf=len(X_train)+len(X_val)+len(Y_train)+len(Y_val)),
                        optimizer=optimizer,
                        metrics=["accuracy"])


            X_train = tf.constant(X_train)
            Y_train = tf.constant(Y_train)
 
            # Fit the model
            history = model.fit(X_train, Y_train,
                                validation_data=(X_val, Y_val),
                                epochs=3000,
                                steps_per_epoch=50,
                                validation_steps=None)
            
            return model
            