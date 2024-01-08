import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import models 
from keras.models import Sequential
from keras.layers import Dense, Lambda
import math as mt
import numpy as np

class engine:

        model_trained=None
        history=None
        
        def __init__(self, X_train, X_val, Y_train, Y_val ,Nf):
            
            self.model_trained=self.piin(X_train=X_train, X_val=X_val, Y_train=Y_train, Y_val=Y_val,Nf=Nf)
            
            pass

        def mexican_hat(self,x):
        
            return (1 - x**2) * tf.math.exp(-(x**2)/2)

        def prediction(self):
                        
            history_df = pd.DataFrame(self.history.history)

            # Create a figure
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            # Plot loss
            ax[0].plot(history_df['loss'], label='Train Loss')
            ax[0].plot(history_df['val_loss'], label='Validation Loss')
            ax[0].set_title('Loss')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Loss')
            ax[0].legend()

            # Plot accuracy
            ax[1].plot(history_df['accuracy'], label='Train Accuracy')
            ax[1].plot(history_df['val_accuracy'], label='Validation Accuracy')
            ax[1].set_title('Accuracy')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend()

            plt.tight_layout()
            plt.show()            
            pass          
  
        
        def Lf(self,prediction,Nf):
            
            return ((2 * prediction[3] + prediction[0] * prediction[2])**2) / Nf
        
        def Lb(self,prediction,Nf):
            
            return (((prediction[0]**2) + prediction[1])**2 + (prediction[1] - 1)**2) / Nf
            
        def blasius_equation( self,prediction ,Nf ):
            
            lf=self.Lf(prediction,Nf )
            lb=self.Lb(prediction,Nf)

            print(type(lf),type(lb))

            if lf > 1e308:
                lf=1e308
    
            if lb > 1e308:
                lb=1e308
                
            L_pred=lf + lb
            
            if L_pred > 1e308:
                L_pred=1e308
            
            return L_pred   
        
        def decay_rate(self ,List_decay_epoch,init_rate,decay_step): # calculate the decay rate
            
            List = list((element[0]/init_rate)**(element[1]/decay_step) for element in List_decay_epoch)
            
            result=0
            for i in List:
                result+=i
            
            return result/len(List) 

        def piin(self, X_train, X_val, Y_train, Y_val,Nf):
            
            tf.random.set_seed(42)
            
            # Create a model similar to model_1 but add an extra layer and increase the number of hidden units in each layer
            model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(1, activation=self.mexican_hat)
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
            custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
            # Compile the model    
            model.compile(loss= lambda y_true , y_pred: self.blasius_equation(y_pred,Nf=Nf),
                        optimizer=custom_optimizer,
                        metrics=["accuracy"])
 
            # Fit the model
            history = model.fit(X_train, Y_train,
                                validation_data=(X_val, Y_val),
                                epochs=100)
            
            model.summary()
            self.history= history
            
            return model
            