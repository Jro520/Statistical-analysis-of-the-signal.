# Análisis estadístico de la señal.
Este proyecto tiene como objetivo entender cómo realizar un análisis estadístico de una señal fisiológica utilizando dos métodos: con funciones predefinidas y con funciones no predefinidas. También se utilizaron tres tipos de ruido para contaminar la señal, y se calculó la relación señal-ruido (SNR) de estos dos.
# librerias que vamos a usar
    import wfdb
	import matplotlib.pyplot as plt
    import numpy as np
    import statistics

  # La señal
 
importamos la señal y guardamos los datos en dos arreglos.
    
    senalizita = wfdb.rdrecord('fetal_PCG_p03_GW_37')
	valores = senalizita.p_signal
    tamano = senalizita.sig_len
Graficamos la señal

    plt.figure(figsize=(12, 6))
    plt.plot(valores)

![Figure 2024-08-08 072227 (0)](https://github.com/user-attachments/assets/1d1d9ece-854f-41c9-b7b0-9b7b5c8a5e27)


# análisis estadístico 

primero, vamos a hacer un análisis estadístico usando las funciones no predefinidas de Python.

## Cálculo de la media de una señal

Este código muestra cómo calcular la media de una señal en un array bidimensional.

```python
# Creamos una variable suma para guardar los datos de la señal
suma = 0
valores_a_planar = valores.flatten()

# A través de dos bucles for llenamos la variable suma con la sumatoria de los datos
for fila in valores:
    for col in fila:
        suma += col

# Calculamos la media, tomando la sumatoria de datos y la dividimos en su cantidad
# usando la función len()
media = suma / len(valores)

# Imprimimos el resultado de la media
print("La media es: ", media
````
## Cálculo de la desviación estandar de una señal

Este código muestra cómo calcular la desviacion estandar de una señal en un array bidimensional.
```python
# Declaramos dos variables que vamos a usar
radi = 0
resta = 0

# Usamos un bucle for para realizar la diferencia 
for valor in valores:
    # Llenamos la variable que creamos por medio de la diferencia al cuadrado de los datos y la media al cuadrado
    resta += (valor - media)**2

# Llenamos radi usando resta y dividiéndola en la cantidad de datos
# usando la función len()
radi = resta / (len(valores) - 1)

# Multiplicamos la variable radi por 0.5, lo que equivale a la raíz cuadrada
desv = radi**0.5
print("La desviación estándar es: ", desv)
````
## Cálculo del Coeficiente de Variación

El coeficiente de variación (CV) es una medida de dispersión relativa que se calcula dividiendo la desviación estándar entre la media y luego multiplicando el resultado por 100.

```python
# Cálculo del coeficiente de variación
cV = (desv / media) * 100

print("El coeficiente de variación es: ", cV)
````
## Creación de un Histograma Manual

Este código muestra cómo crear un histograma manualmente en Python usando `matplotlib` y `numpy`.

```python
# Encontrar los valores mínimo y máximo
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
```
![Figure 2024-08-08 072227 (1)](https://github.com/user-attachments/assets/9aacff04-f81d-4485-8aec-23bce5b52472)

## Función de Probabilidad

Este código calcula y grafica la función de distribución de probabilidad para un conjunto de datos.

```python
# Calcular probabilidades
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
````

![Figure 2024-08-08 072227 (2)](https://github.com/user-attachments/assets/d9310512-0b92-4249-a1a0-211ff52a54d7)

## Cálculo de Estadístico usando funciones predefinidas

```python
# Calcular la media
media2 = statistics.mean(valores_a_planar)

# Calcular la desviación estándar
desE = statistics.stdev(valores_a_planar)  # Para la desviación estándar muestral

# Calcular el coeficiente de variación
coeficienteV = (desE / media2) * 100

print("La media es: ", media2)
print("La desviación estándar es: ", desE)
print("El coeficiente de variación es: ", coeficienteV)

# Graficar un histograma
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
probabilidades = frecuencias / np.sum(frecuencias)  # Calcular probabilidades

plt.figure(figsize=(10, 6))
plt.bar(bins[:-1], probabilidades, width=(bins[1] - bins[0]), edgecolor='black')
plt.title('Función de Probabilidad de la Señal')
plt.xlabel('Valor')
plt.ylabel('Probabilidad')
plt.grid(True)
plt.show()
````
# Contaminacion de la señal
En esta seccion contaminaremos la señal con tres tipos de ruido y calcularemos la SNR de esto
## Generación de Ruido Gaussiano y Cálculo de SNR

```python
# En esta parte vamos a generar el ruido gaussiano usando una función del numpy
ruido = np.random.normal(0, 1, len(valores)) 

AmpRuido = (ruido * (max_val * 0.3))
print("La amplitud del ruido es ", AmpRuido)

# Contaminar la señal con el ruido escalado
senal_junta = AmpRuido + valores_a_planar

# Calcular la potencia de la señal original
potencia_senal = np.mean(valores_a_planar ** 2)

# Calcular la potencia del ruido gaussiano 
potencia_ruido = np.mean(AmpRuido ** 2)

# Calcular la relación señal a ruido (SNR)
SNR = 10 * np.log10(potencia_senal / potencia_ruido)

print("La relación señal a ruido (SNR) es: ", SNR)

# Graficar la señal original y la señal con ruido
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(AmpRuido)
plt.title('Ruido Gaussiano')
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

````
![Figure 2024-08-08 072227 (5)](https://github.com/user-attachments/assets/cb287575-146f-4325-a589-f0600cedf32d)
![Figure 2024-08-08 072227 (6)](https://github.com/user-attachments/assets/71cf0d11-8562-4fe3-b400-178aa4417912)


## Generación de Ruido de Impulso y Cálculo de SNR

```python
# En esta parte vamos a generar el ruido de impulso
# Generar ruido de impulso
impulsos = np.zeros_like(valores_a_planar)
num_impulsos = int(0.002 * len(valores_a_planar))  # 5% de los puntos serán impulsos
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
plt.title('Ruido Impulso')
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
```
![Figure 2024-08-08 072227 (7)](https://github.com/user-attachments/assets/fbaf6964-6d58-44f4-b2a2-3ea8e8cfa7ec)
![Figure 2024-08-08 072227 (8)](https://github.com/user-attachments/assets/edc5d19d-4b26-45da-b0c5-4eff4a3190f3)

## Generación de Ruido Tipo Artefacto y Cálculo de SNR

```python
# Generar ruido artefacto
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
plt.title('Ruido Artefacto')
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
```
![Figure 2024-08-08 072227 (9)](https://github.com/user-attachments/assets/6be0e016-2f47-4735-90e3-38c6a0e4f41d)
![Figure 2024-08-08 072227 (10)](https://github.com/user-attachments/assets/810f464a-cb5c-462c-94d9-190e67f5fba0)

