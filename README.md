# Redes neuronales con información física basadas en wavelet ( PINN ) . Aplicación a la solución de modelos de Ecuaciones Diferenciales en el Procesamiento de Imágenes.

## Autores: 
<center>
<li>
Abdel Fregel Hernández C312
</li>
<li>
Yonatan Jose Guerra Perez C311
</li>
<li>
Jose Miguel Perez Perez C311
</li>
</center>	

<br/>

Las redes neuronales con información física, **Physics Informed Neuronal Network ( PINNN )** por sus siglas en inglés, son un tipo de redes neuronales usadas en la aproximación de la solución de ecuaciones diferenciales que, como su nombre indica, agrega información física a la función de pérdida **(loss-function)** de la misma. En este trabajo aplicamos este tipo de red neuronal a la solución de la ecuación de Blasius **(Blasius Equation)**, una ecuación diferencial no lineal definida en un dominio no acotado y uno de los más prominentes ecuaciones en fluidos dinámicos.

La ecuación de Blasius está dada por la siguiente expresión:
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$
$$
u\frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} =  - \frac{1}{\rho}\frac{\partial p} {\partial x} + v \frac{\partial^2u }{\partial y^2}
$$
que se resuelve utilizando la expresiones siguientes:
$$
	\frac{x^2}{U^2\delta^*}\frac{1}{\rho}\frac{\partial P}{\partial y} = \frac{1}{2}\eta f^{\left(3\right)} + \frac{1}{2}f^{''} - \frac{1}{4}ff^{'} + \frac{1}{4}\eta f^{'2} + \frac{1}{4}\eta ff^{''}
$$
$$
	u\left(x,y\right) = \frac{\partial\psi}{\partial y} = Uf^{'}\left(\eta\right)
$$
$$
	v\left(x,y\right) = -\frac{\partial\psi}{\partial x} = \frac{1}{2}\sqrt{\frac{vU}{x}}\left[ \eta f^{'}\left(\eta\right) - f\left(\eta\right)\right]
$$
con $\eta$ expresado como sigue:
$$
\eta = \frac{y}{\delta\left(x\right)} = y\sqrt{\frac{U}{vx}}
$$
Para la implementación de la red se usó el lenguaje Python3.9.5 y las siguientes librerias externas:
- `sympy`
- `tensorflow`
- `keras`
- `pandas`
- `numpy`
- `matplotlib`

El projecto consta de 4 submódulos:
- `boundary-conditions`: Es donde está definida la ecuación de Blasius.
- `data-generator`: Es donde se generan los datos de entrenamiento de la red.
- `engine`: Es donde se definen todos los métodos necesarios para resolver el problema, así como definir la red neuronal.
- `main`: Es donde se ejecuta el programa para obtener la aproximación de la solución del problema.

Basándonos en el artículo **Wavelets based physics informed neural networks to**, diseñamos la red con una estructura de 4 capas con 4 neuronas cada una, usando como función de activación la función del sombrero mexicano **mexican-hat-activation-function**, definida en la siguiente expresión:
$$
f\left(x\right) = \left(1 - x^2\right)e^{\frac{-x^2}{2}}
$$
La función de pérdida usada se define como sigue:
$$
L = L_f + L_b
$$
donde
$$
L_f = \frac{\Sigma^{N_f}_{i = 1}\left\lbrace\left(2f^{'''}\left(x_i\right) + f\left(x_i\right)f^{''}\left(x_i\right)\right)^2\right\rbrace}{N_f}
$$
y
$$
L_b = \frac{\Sigma^{N_f}_{i = 1}\left\lbrace\left(2f\left(x_{0i}\right)^2 + f^{'}\left(x_{0i}\right)\right)^2 + \left(f^{'}\left(x_{0i}\right) - 1\right)^2\right\rbrace}{N_b}
$$
Las constantes usadas para la generación de datos fueron $v = \frac{1}{100*\pi}$, $\alpha = 0.33205$ y $\beta =1.72078$. Las cuales fueron usadas en estas ecuaciones:
$$
f\left(\eta\right) = \frac{1}{2} \alpha\eta^2
$$
cuando $\eta << 1$; y
$$
f\left(\eta\right) = \eta - \beta
$$
cuando $\eta >> 1$.
El decaimiento del ritmo de aprendizaje esta implementado de la siguiente forma:
<image src="pictures\decay-rate_implementation.png">

A continuación se muestran imágenes de los detalles de implementación del proyecto.

## Learning-rate-implementation
<image src="pictures\learning-rate_implementation.png">

## Model hyperparameters customization
<image src="pictures\model_hyper-parameters_customization.png">

## Model layers
<image src="pictures\model-layers.png">

## Visualización del modelo usado
<image src="pictures\visual-model.png">

Los resultados del proyecto se muestran en las siguientes imágenes:

## Loss-val function relationship
<image src="pictures\loss-val_function-relationship.png">

## Loss-val function relationship extended
<image src="pictures\loss-val_function-relationship-extended.png">


##  Nota: Para correr el proyecto corra el archivo main.py en la raiz de este