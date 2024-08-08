#En esta parte importamos las librerias necesarias 
import statistics
import wfdb
import matplotlib.pyplot as plt
import numpy as np
#En esta parte importamos la señal
senalizita = wfdb.rdrecord('fetal_PCG_p03_GW_37')
#guardamos la señal en un arreglo
valores = senalizita.p_signal
#tamaño de la señal
tamano = senalizita.sig_len
#imprimimos la señal y la graficamos 
print (valores)
plt.figure(figsize=(12, 6))
plt.plot(valores)
print("----------------- SIN EL USO DE FUNCIONES PREESTABLECIDAS -----------------")
#creamos una variable suma para guardar los datos de la señal
suma = 0
valores_a_planar = valores.flatten()
#a traves de dos un bucles for llenamos la variable suma con la sumatoria de los datos
for fila in valores:
    for col in fila:
        suma += col
# Calculamos la media, tomando la sumatoria de datos y la dividimos en su cantidad
#usando la funcion len()
media = suma / len(valores)
#imprimimos el resultado de la media
print("La media es: ", media)
#declaramos dos variables que vamos a usar
radi = 0
resta = 0
#usamos un bucle for para realizar la diferencia 
for valor in valores:
#llenamos la variable que creamos por medio de la diferencia al cuadrado de los datos y la media al cuadrado
    resta += (valor - media)**2
#llenamos radi usando resta y dividiendola en la cantidad de datos
#usando la funcion len()
    radi = resta / (len(valores)-1)
#multiplicamos la variable len por 0.5, lo que equivale a la raiz cuadrada
desv = radi**0.5
print("La desviacion estandar es: ", desv)

#calculo del coeficiente de variacion dividiendo la desviacion estandar y la media, para luego multiplicarlo por 100
cV = (desv / media) * 100

print("El coeficiente de variacion es: ", cV)


min_val = np.min(valores_a_planar)
max_val = np.max(valores_a_planar)
num_bins = 30
interval_width = (max_val - min_val) / num_bins

# Calcular frecuencias
frecuencias = [0] * num_bins
for valor in valores_a_planar:
    index = int((valor - min_val) / interval_width)
    if index == num_bins:  # Para valores que estén exactamente en el límite superior
        index -= 1
    frecuencias[index] += 1

# Graficar el histograma manualmente
plt.figure(figsize=(10, 6))

# Graficar las barras
for i in range(num_bins):
    plt.bar(
        min_val + i * interval_width,
        frecuencias[i],
        width=interval_width,
        edgecolor='black'
    )

plt.title('Histograma de Valores')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()
# Calcular las probabilidades
#---------------------------------------------------------------
probabilidades = [frecuencia / tamano for frecuencia in frecuencias]

# Graficar la función de probabilidad
plt.figure(figsize=(10, 6))
for i in range(num_bins):
    plt.bar(
        min_val + i * interval_width,
        probabilidades[i],
        width=interval_width,
        edgecolor='black'
    )
plt.title('Función de Distribución de Probabilidad de la Señal')
plt.xlabel('Valor')
plt.ylabel('Probabilidad')
plt.grid(True)
plt.show()
#-----------------------------------------------------------------------------
print("----------------- CON EL USO DE FUNCIONES PREESTABLECIDAS -----------------")

# Calcular la media
media2 = statistics.mean(valores_a_planar)

# Calcular la desviación estándar
desE = statistics.stdev(valores_a_planar)  # Para la desviación estándar muestral

# Calcular el coeficiente de variación
coeficienteV = (desE / media2) * 100

print("La media es: ", media2)
print("La desviacion estandar es: ", desE)
print("El coeficiente de variacion es: ", coeficienteV)
# Grafica de un histograma
plt.figure(figsize=(10, 6))
plt.hist(valores_a_planar, bins=30, edgecolor='black')
plt.title('Histograma de Valores')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()
# Graficar la función de probabilidad 
num_bins = 30
frecuencias, bins = np.histogram(valores, bins=num_bins)
plt.figure(figsize=(10, 6))
plt.bar(bins[:-1], probabilidades, width=(bins[1] - bins[0]), edgecolor='black')
plt.title('Función de Probabilidad de la Señal')
plt.xlabel('Valor')
plt.ylabel('Probabilidad')
plt.grid(True)
plt.show()
print("----------------- SNR -----------------")
#En esta parte vamos a generar el ruido gaussiano usando una funcion del numpy
ruido = np.random.normal (0,1,len(valores)) 

AmpRuido = (ruido * (max_val*0.3))
print("La amplitdo del ruido es ", AmpRuido)

# Contaminar la señal con el ruido escalado
senal_junta = AmpRuido + valores_a_planar

# Calcular la potencia de la señal original
potencia_senal = np.mean(valores_a_planar ** 2)

# Calcular la potencia del ruido gaussiano 
potencia_ruido = np.mean(AmpRuido ** 2)

# Calcular la relación señal a ruido (SNR)
SNR = 10 * np.log10(potencia_senal /potencia_ruido)

print("La relación señal a ruido (SNR) es: ", SNR)

# Graficar la señal original y la señal con ruido
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(AmpRuido)
plt.title('ruido gaussiano')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')


plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(valores)
plt.title('Señal Original')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 2)
plt.plot(senal_junta)
plt.title('Señal con Ruido Gaussiano')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()
#En esta parte vamos a generar el ruido de impulso
# Generar ruido de impulso
impulsos = np.zeros_like(valores_a_planar)
num_impulsos = int(0.002*len(valores_a_planar))  # 5% de los puntos serán impulsos
posiciones_impulsos = np.random.choice(len(valores_a_planar), num_impulsos, replace=False)
impulsos[posiciones_impulsos] = max_val * np.random.choice([-1, 1], num_impulsos) * np.random.uniform(0, 1, num_impulsos)

# Contaminar la señal con el ruido de impulso
signal_contaminada = valores_a_planar + impulsos
# Calcular la potencia de la señal original
potencia_senal = np.mean(valores_a_planar ** 2)

# Calcular la potencia del ruido de impulso
potencia_ruido_impulso = np.mean((impulsos - valores_a_planar) ** 2)

# Calcular la relación señal a ruido (SNR) para el ruido de impulso
SNR_impulso = 10 * np.log10(potencia_senal / potencia_ruido_impulso)

print("La relación señal a ruido (SNR) con ruido de impulso es: ", SNR_impulso)

# Graficar la señal original y la señal contaminada
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(impulsos)
plt.title('ruido impulso')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')


plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(valores)
plt.title('Señal Original')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')


plt.subplot(2, 1, 2)
plt.plot(signal_contaminada)
plt.title('Señal con Ruido Impulso')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')

plt.grid(True)
plt.show()

#Generar ruido artefacto
frecuencia_ruido = 50  # Frecuencia del ruido, ajusta este valor según sea necesario
amplitud_ruido = np.std(valores_a_planar)  # Amplitud del ruido
tiempo = np.arange(tamano)
ruido_artefacto = amplitud_ruido * np.sin(2 * np.pi * frecuencia_ruido * tiempo / tamano)

senal_con_artefacto = valores_a_planar + ruido_artefacto

# Calcular la potencia de la señal original
potencia_senal = np.mean(valores_a_planar ** 2)

# Calcular la potencia del ruido tipo artefacto
potencia_ruido_artefacto = np.mean(ruido_artefacto ** 2)

# Calcular la relación señal a ruido (SNR) para el ruido tipo artefacto
SNR_artefacto = 10 * np.log10(potencia_senal / potencia_ruido_artefacto)

print("La relación señal a ruido (SNR) con ruido tipo artefacto es: ", SNR_artefacto)

# Graficar la señal original y la señal con ruido tipo artefacto
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ruido_artefacto)
plt.title('ruido artefacto')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(valores)
plt.title('Señal Original')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')

plt.subplot(2, 1, 2)
plt.plot(senal_con_artefacto)
plt.title('Señal con Ruido Tipo Artefacto')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()

