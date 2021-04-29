# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Ejercicios del Seminario 1 (Introducción a Python)
# 
# En este notebook se pide realizar un par de ejercicios sencillos para ir practicando Python.
# %% [markdown]
# ## Ejercicio 1
# 
# 
# Escriba una función _califica_ que reciba un vector con los valores de los
# estudiantes, y devuelva "MH" si la nota se encuentra en 10, "SOB" si la nota media
# se encuentra entre 9 y 10, "NOT" si se encuentra entre 7 y 9, "BIEN" si se encuentra
# entre 6 y 7, "APRO" si tiene entre 5 y 6, y "SUS" si es menor que cinco.
# 

# %%

def califica(vector):
    result = []
    cad = ""
    
    for valor in vector:
        if valor == 10:
            cad = "MH"
        elif valor >= 9 and valor < 10:
            cad = "SOB"
        elif valor >= 7 and valor < 9:
            cad="NOT"
        elif valor >= 6 and valor < 7:
            cad="BIEN"
        elif valor >= 5 and valor < 6:
            cad="APRO"
        elif valor < 5:
            cad="SUS"
        result.append(cad)
    
    return result

def main():
    result = califica([2.3, 5.6, 8, 10, 9, 4.2])
    print(result)
    assert result == ["SUS", "APRO", "NOT", "MH",  "SOB", "SUS"]
    
main()

# %% [markdown]
# ## Ejercicio 2
# 
# Escribir las funciones _potencia_ y _factorial_.

# %%
def potencia(a, b):
    result = 1
    for i in range(b):
        result = result*a
    return result

def factorial(n):
    result = 1    
    for i in range(n):
        result = result*(i+1)
    return result

def test_potencia():
    assert potencia(2, 2)==4
    assert potencia(2, 3)==8
    assert potencia(2, 1)==2
    assert potencia(1, 10)==1
    
def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(2) == 2
    assert factorial(3) == 6
    assert factorial(5) == 120
    
test_potencia()
test_factorial()


