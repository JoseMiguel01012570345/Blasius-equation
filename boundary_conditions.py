import math as mt
from sympy import symbols , diff , sqrt ,integrate

v=1/(100*mt.pi)
alpha=0.33205
beta=1.72078
            
def mass_continuity(x0,y0,U,epsilon): # du/dx + dv/dy = 0
    
    x,y=symbols("x y")
    
    f=None
    if alpha != 0:
        
        f = ( alpha * ( y**2 ) * U )/( 2 * v * x )
    else :
        f = y * mt.sqrt( U / (v * x) ) - beta
    
    # n
    n = y * sqrt( U / ( v * x ))
    
    u= U * diff(f,x,y) 
    du_dx= diff(u,x)
    
    V = sqrt(( v * U )/ x ) * ( n * diff(f,x,y) - f )
    dV_dy=diff(V,y)
    
    return du_dx.subs({x:x0,y:y0}) + dV_dy.subs({x:x0,y:y0}) < epsilon

def x_momentum(x0,y0,U,rho,epsilon):
    
    x,y=symbols("x y")
    
    f=None
    if alpha != 0:
        
        f = ( alpha * ( y**2 ) * U )/( 2 * v * x )
    else :
        f = y * mt.sqrt( U / (v * x) ) - beta
    
    # n
    n = y * sqrt( U / ( v * x ))
    
    # du/dx
    u = U * diff(f,x,y) 
    du_dx= diff(u,x)
    
    # dV/dy
    V = sqrt(( v * U )/x) * ( n * diff(f,x,y) - f )
    dV_dy=diff(V,y)

    # (v*d2u)/d2y    
    u_second_order = v*diff(u,y,2)
    
    # dP_dx
    n= y * sqrt( U / ( v * x ))

    func = n * diff(f, x, 3, y, 3) + \
                diff(f, x, 2, y, 2) - \
                f * diff(f, x, y) + \
                n * (diff(f, x, y)**2) + \
                (n * f * diff(f, x, 2, y, 2))

    g = integrate(func ,y)

    dPdx = diff(g, x)
                
    return (  u.subs({x:x0,y:y0}) * du_dx.subs({x:x0,y:y0}) + V.subs({x:x0,y:y0})*dV_dy.subs({x:x0,y:y0}) - \
        
            (-(dPdx.subs({x:x0,y:y0}))/rho ) + u_second_order.subs({x:x0,y:y0})  ) < epsilon
