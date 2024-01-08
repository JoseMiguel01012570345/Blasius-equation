from sympy import Float
import tensorflow as tf
import random
import boundary_conditions as bc
import time
import numpy as np
class data:
    
    train_data,val_data=[],[]
    
    def __init__(self,no_data) -> None:
        
        train_data,val_data=self.generate_data(no_data)
        
        self.train_data=train_data
        self.val_data=val_data
        
        pass
    
    def storm_data(self):

        # Assume X and Y are your features and labels
        X_train, X_val, Y_train, Y_val =[],[],[],[]
        
        print ("train_data: ",self.train_data.shape)
        
        for i in self.train_data:
            
            numb=random.randint(0,len(self.train_data)-1)

            # Compute the difference between the tensors
            X_train =  tf.sets.difference(self.train_data,self.train_data[numb])
            Y_train =  tf.sets.difference(self.val_data,self.val_data[numb])
            
            self.train_data = X_train
            self.val_data   = Y_train
        
        for i in range(len(self.train_data)):
            
            numb=random.randint (0,len(self.train_data)-1)

            X_val =  tf.sets.difference(self.val_data,self.val_data[numb])
            Y_val =  tf.sets.difference(self.val_data,self.val_data[numb])
            
            self.val_data = X_val
            self.val_data = Y_val
        
        return X_train, X_val, Y_train, Y_val
    
    def generate_data(self,no_data):
        
        epsilon=1e-300
        
        steps=0.001
        limit= steps * no_data
        start_value=1e-30
        boundary_x=start_value
        boundary_y=start_value
        
        hyper_parameters_start_values=0
        
        rho=hyper_parameters_start_values
        U=hyper_parameters_start_values
        
        query_data=tf.Variable([])
        f_data=tf.Variable([])
        
        # clean terminal
        import os
        os.system('cls')
        
        print("collecting data.")
        start_cronos= time.time()
        
        count=0
        for i in range(no_data):
            
            if time.time()-start_cronos >= 4  and time.time()-start_cronos < 8 :
                os.system('cls')
                print(f"collecting data.. {count}/{no_data} {round((count/no_data)*100,ndigits=2)}%")
            
            elif  time.time()-start_cronos >= 8 :
                os.system('cls')
                print(f"collecting data... {count}/{no_data} {round((count/no_data)*100,ndigits=2)}%")
                start_cronos=time.time()
                
            elif  time.time()-start_cronos >= 2 and time.time()-start_cronos < 4 :
                os.system('cls')
                print(f"collecting data. {count}/{no_data} {round((count/no_data)*100,ndigits=2)}%")
                
                
            if rho < 5:
                rho += steps
                U   += steps

            if boundary_x < 5:
                boundary_x += steps
            else:
                boundary_y += steps
                
            if bc.mass_continuity(x0=boundary_x,y0=boundary_y,U=U,epsilon=epsilon) and \
               bc.x_momentum(x0=boundary_x,y0=boundary_y,U=U,rho=rho,epsilon=epsilon): 
                                
                    # Create a SymPy Float
                    sympy_float_X = Float(boundary_x)

                    sympy_float_Y = Float(boundary_y)
                    # Convert the SymPy Float to a native Python float
                    native_float_x = float(sympy_float_X)
                    native_float_y = float(sympy_float_X)

                    # Now you can convert the native Python float to a TensorFlow Tensor
                    tensor_x = tf.constant(native_float_x)
                    tensor_y = tf.constant(native_float_y)
                    
                    # -----------
                    f_list=bc.f_vector(x0=boundary_x, y0=boundary_y, U=U)
                    # ----------
                    sympy_float_f = Float(f_list[0])
                    sympy_float_f1 = Float(f_list[1])
                    sympy_float_f2 = Float(f_list[2])
                    sympy_float_f3 = Float(f_list[3])
                    
                    # Convert the SymPy Float to a native Python float
                    native_float_f = float(sympy_float_f)
                    native_float_f1 = float(sympy_float_f1)
                    native_float_f2 = float(sympy_float_f2)
                    native_float_f3 = float(sympy_float_f3)
                    # Now you can convert the native Python float to a TensorFlow Tensor
                    tensor_f = tf.constant(native_float_f)
                    tensor_f1 = tf.constant(native_float_f1)
                    tensor_f2 = tf.constant(native_float_f2)
                    tensor_f3 = tf.constant(native_float_f3)
                    
                    query_data = tf.concat([query_data, [tensor_x,tensor_y]], axis=0)
                    
                    f_data =  tf.concat([f_data, [tensor_f,tensor_f1,tensor_f2,tensor_f3] ], axis=0)
        
            count +=1            
    
        os.system('cls')
        return query_data,f_data