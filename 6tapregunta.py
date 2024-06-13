import random

num_generaciones = 100
tamaño_población = 50
prob_mutación = 0.1

grafo = {
    'A': {'B': 7, 'C': 9, 'D': 10, 'E': 20},
    'B': {'A': 7, 'D': 8, 'E': 4},
    'C': {'A': 9, 'E': 5},
    'D': {'A': 10, 'B': 8, 'E': 17},
    'E': {'A': 20, 'B': 4, 'C': 5, 'D': 17}
}

def generar_ruta():
    return random.sample(grafo.keys(), len(grafo))

def calcular_distancia(ruta):
    distancia = 0
    for i in range(len(ruta) - 1):
        if ruta[i+1] in grafo[ruta[i]]:
            distancia += grafo[ruta[i]][ruta[i+1]]
        else:
            return float('inf') 
    return distancia

def evaluar_fitness(ruta):
    distancia = calcular_distancia(ruta)
    if distancia == float('inf'):
        return 0
    return 1 / distancia

def seleccionar_padres(población):
    fitness_total = sum(evaluar_fitness(ruta) for ruta in población)
    seleccion = random.uniform(0, fitness_total)
    acumulado = 0
    for ruta in población:
        acumulado += evaluar_fitness(ruta)
        if acumulado >= seleccion:
            return ruta

def cruzar(padre1, padre2):
    punto1, punto2 = sorted(random.sample(range(len(padre1)), 2))
    hijo = padre1[:punto1] + padre2[punto1:punto2] + padre1[punto2:]
    return hijo

def mutar(ruta):
    if random.random() < prob_mutación:
        i, j = random.sample(range(len(ruta)), 2)
        ruta[i], ruta[j] = ruta[j], ruta[i]

def algoritmo_genetico():
    poblacion = [generar_ruta() for _ in range(tamaño_población)]
    for _ in range(num_generaciones):
        nueva_poblacion = []
        for _ in range(tamaño_población // 2):
            padre1 = seleccionar_padres(poblacion)
            padre2 = seleccionar_padres(poblacion)
            hijo1 = cruzar(padre1, padre2)
            hijo2 = cruzar(padre2, padre1)
            mutar(hijo1)
            mutar(hijo2)
            nueva_poblacion.extend([hijo1, hijo2])
        poblacion = nueva_poblacion
    mejor_ruta = min(poblacion, key=calcular_distancia)
    return mejor_ruta, calcular_distancia(mejor_ruta)

mejor_ruta, mejor_distancia = algoritmo_genetico()
print(f"Mejor ruta: {mejor_ruta} con distancia {mejor_distancia}")
