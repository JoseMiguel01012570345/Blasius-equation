from sympy import Float
import tensorflow as tf
import random
import boundary_conditions as bc
import time
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
        
        length=len(self.train_data)
        
        for i in range(int( len(self.train_data) * 0.8 )):
            
            numb=random.randint(0,len(self.train_data)-1)
            
            sympy_float_X = Float(self.train_data[numb][0])
            sympy_float_Y = Float(self.train_data[numb][1])
            
            # Convert the SymPy Float to a native Python float
            native_float_x = float(sympy_float_X)
            native_float_y = float(sympy_float_Y)
            
            X_train.append([native_float_x,native_float_y])
            
            #-----------------------------------------------------
            f_list=self.val_data[numb]
            # ----------
            sympy_float_f = Float(f_list[0])
            sympy_float_f1 = Float(f_list[1])
            sympy_float_f2 = Float(f_list[2])
            sympy_float_f3 = Float(f_list[3])
                    
                    # Convert the SymPy Float to a native Python float
            native_float_f  = float(sympy_float_f)
            native_float_f1 = float(sympy_float_f1)
            native_float_f2 = float(sympy_float_f2)
            native_float_f3 = float(sympy_float_f3)
            
            Y_train.append([native_float_f,native_float_f1,native_float_f2,native_float_f3])
            
            self.train_data.pop(numb)
            self.val_data.pop(numb)
        
        for i in range(len(self.train_data)):
            
            numb=random.randint(0,len(self.train_data)-1)
            
            sympy_float_X = Float(self.train_data[numb][0])
            sympy_float_Y = Float(self.train_data[numb][1])
            
            native_float_x = float(sympy_float_X)
            native_float_y = float(sympy_float_Y)
            
            X_val.append([native_float_x,native_float_y])
            
            #-----------------------------------------------------
            f_list=self.val_data[numb]
            # ----------
            sympy_float_f = Float(f_list[0])
            sympy_float_f1 = Float(f_list[1])
            sympy_float_f2 = Float(f_list[2])
            sympy_float_f3 = Float(f_list[3])
                    
                    # Convert the SymPy Float to a native Python float
            native_float_f  = float(sympy_float_f)
            native_float_f1 = float(sympy_float_f1)
            native_float_f2 = float(sympy_float_f2)
            native_float_f3 = float(sympy_float_f3)
            
            Y_val.append([native_float_f,native_float_f1,native_float_f2,native_float_f3])
                
        return X_train , X_val , Y_train , Y_val ,length
    
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
        
        query_data=[]
        f_data=[]
        
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

                    query_data.append([boundary_x,boundary_y])
                    
                    f_data.append(bc.f_vector(x0=boundary_x,y0=boundary_y,U=U))
        
            count +=1            
    
        os.system('cls')
        return query_data,f_data