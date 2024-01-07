from functools import partial
import tensorflow as tf
from keras import models 
from keras.models import Sequential
from keras.layers import Dense, Lambda
from sympy import symbols, diff , integrate ,sqrt
import math as mt
import sys

class engine:

        epsilon=1
        model_trained=None
        
        def __init__(self, train_data , valid_data,epsilon,alpha,beta,U):
            
            self.epsilon=epsilon
            self.model_trained=self.model_arquitecture(train_data=train_data,valid_data=valid_data,alpha=alpha,beta=beta,U=U)
            
            pass

        def mexican_hat(x):
            
            return  (1 - x**2)*mt.e**(-(x^2)/2)

        def prediction(model, x):
            
            return model.predict(x)
        
        def Lf(self,x0,y0,alpha,beta,U,Nf):
            
            f=None
            
            x,y=symbols("x y")
            
            if alpha!=0 :
                f = ( alpha * ( y**2 ) * U )/( 2 * self.v * x ) #f_a
            else:
                f = y * sqrt( U / (self.v * x) ) - beta
            
            lf=((diff(f,x,3,y,3)+f*diff(f,x,2,y,2))**2)/Nf
            
             
            try: 
                    
                return lf.subs({x:x0,y:y0})

            except:
                return sys.float_info.max  
            
        def Lb(self,x0,y0,alpha,beta,U,Nf):
            
            f=None
            
            x,y=symbols("x y")
            
            if alpha!=0 :
                f = ( alpha * ( y**2 ) * U )/( 2 * self.v * x ) #f_a
            else:
                f = y * sqrt( U / (self.v * x) ) - beta
            
            lb = ((f**2) + (diff(f,x,y)**2) + (diff(f,x,y)-1)**2)/Nf
           
            try: 
                    
                return lb.subs({x:x0,y:y0})

            except:
                return sys.float_info.max  

        def blasius_equation( prediction,self ,alpha,beta,U,Nf ):
            
            L_pred=self.Lf(self,prediction[0],prediction[1],alpha,beta,U,Nf ) + self.Lb(self,prediction[0],prediction[1],alpha,beta,U,Nf)
            
            if L_pred < self.epsilon:
                print("sharp model")
            
            return L_pred   
        
        def decay_rate(List_decay_epoch,init_rate,decay_rate): # calculate the decay rate
            
            list = ((element[0]/init_rate)**(element[1]/decay_rate) for element in List_decay_epoch)
            
            result=0
            for i in list:
                result+=i
            
            return result/len(list) 
        
        def model_arquitecture(self, train_data, valid_data,alpha,beta,U):

            tf.random.set_seed(42)

            # Create a model similar to model_1 but add an extra layer and increase the number of hidden units in each layer
            model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(4, activation=self.mexican_hat), # add an extra layer
            tf.keras.layers.Dense(2, activation=self.mexican_hat)
            ])
        
            initial_learning_rate = 0.01
            decay_steps = 3000
            decay_rate = self.decay_rate([[0.001,1000],[0.0005,3000]])

            # Define the learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False)
            
            # Create an optimizer with the learning rate schedule
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
            # Compile the model
            model.compile(loss= lambda y_true , y_pred: self.blasius_equation(y_pred,self,alpha=alpha,beta=beta,U=U,Nf=len(train_data)),
                        optimizer=optimizer,
                        metrics=["accuracy"])

            # Fit the model
            history = model.fit(train_data,
                                    epochs=3000,
                                    steps_per_epoch=len(train_data),
                                    validation_data=valid_data,
                                    validation_steps=len(valid_data))
            
            return model
            