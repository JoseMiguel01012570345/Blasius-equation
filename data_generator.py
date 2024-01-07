import boundary_conditions as bc

class data:
    
    all_data=[]
    
    def __init__(self,no_data) -> None:
        
        self.all_data=self.generate_data(no_data)
        
        pass
    
    
    def generate_data(self,no_data):
        
        epsilon=1e-300
        
        steps=0.001
        limit= steps * no_data
        start_value=0
        boundary_x=start_value
        boundary_y=start_value
        
        
        hyper_parameters_start_values=0
        
        rho=hyper_parameters_start_values
        U=hyper_parameters_start_values
        
        data=[]
        
        for i in range(no_data):
            
            if rho < limit:
                rho += steps
            else:
                U += steps

            if boundary_x < limit:
                boundary_x += steps
            else:
                boundary_y += steps
                
            if bc.mass_continuity(x0=boundary_x,y0=boundary_y,U=U,epsilon=epsilon) and \
               bc.x_momentum(x0=boundary_x,y0=boundary_y,U=U,rho=rho,epsilon=epsilon): 
                    
                    data.append([boundary_x,boundary_y])
        
        
        return data