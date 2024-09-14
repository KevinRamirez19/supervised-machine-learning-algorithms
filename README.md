# Regresion-Lineal
En este repositorio se alojara los códigos de python 

# Bibliotecas necesarias:
numpy: para crear y manejar arreglos de datos.

train_test_split de sklearn.model_selection: para dividir los datos en conjuntos de entrenamiento y prueba.

LinearRegression de sklearn.linear_model: para aplicar el modelo de regresión lineal.

matplotlib.pyplot como plt: para graficar los resultados.

# Variables de datos:

advertising_expenditure: un array de numpy que contiene los gastos en publicidad.

sales: un array de numpy que contiene las ventas correspondientes.

# Proceso:

Se reorganiza el array de gastos (x = advertising_expenditure.reshape(-1,1)) para que sea compatible con el modelo de regresión lineal.
Se divide el dataset en entrenamiento (80%) y prueba (20%) utilizando train_test_split.
Se entrena un modelo de regresión lineal con los datos de entrenamiento (model.fit).
Se predicen las ventas en función de los datos de prueba (model.predict).
Se calcula el coeficiente de determinación 𝑅^2
y se imprimen los coeficientes y la intersección del modelo.
Se genera una gráfica que muestra los datos reales y la línea de predicción del modelo.

# Ejecucion
#  1. Instalar las bibliotecas necesarias:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
 # 2. Definir los datos:
    advertising_expenditure = np.array([...])
    sales = np.array([...])
 # 3. Reshape de los datos 
    x = advertising_expenditure.reshape(-1, 1)
    y = sales
 # 4. Dividir el conjunto de datos en entrenamiento y prueba:
    * Conjunto de entrenamiento (80%)
    * Conjunto de prueba (20%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
 # 5. Entrenar el modelo de regresión lineal:
    model = LinearRegression()
    model.fit(x_train, y_train)
 # 6. Hacer predicciones:
    y_pred = model.predict(x_test)
 # 7. Calcular la precisión del modelo (R2 Score):
   r2 = model.score(x_test, y_test)
   print("R2 Score:", r2)
 # 8. Obtener el coeficiente y el intercepto:
   coeffient = model.coef_[0]
   print("Coefficient:", coeffient)
   intercept = model.intercept_
   print("Intercept:", intercept)
 # 9. Graficar los resultados:
   plt.scatter(x_test, y_test, color='green')
   plt.plot(x_test, y_pred, color='orange', linestyle = '-', linewidth = 2 , label='predicted      sales' )
   plt.xlabel('advertising_expenditure')
   plt.ylabel('sales')
   plt.title('advertising_expenditure vs sales')
   plt.legend()
   plt.show()
 # 10. Ejecutar el codigo
 # 10.1 Resultado Esperado:
 1. El código imprimirá el coeficiente 𝑅2 , el coeficiente de la regresión y la intersección.
 
 2. Luego mostrará una gráfica con los datos reales (ventas) y la línea de predicción generada por el modelo.
 3. ![image](https://github.com/user-attachments/assets/61037b1f-451a-4e12-af2e-d062dd1960aa)

 
 








      
 
