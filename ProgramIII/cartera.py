import numpy as np

# Simulamos retornos esperados y covarianza entre activos
n_activos = 5
retornos_esperados = np.random.uniform(0.05, 0.15, size=n_activos)
matriz_covarianza = np.random.rand(n_activos, n_activos)
matriz_covarianza = (matriz_covarianza + matriz_covarianza.T) / 2  # Hacerla simétrica
np.fill_diagonal(matriz_covarianza, 0.1)  # Riesgo individual

# Parámetros del AG
tam_poblacion = 30
n_generaciones = 100
prob_mutacion = 0.2
elite_ratio = 0.2

# Crear un individuo (portafolio válido)
def crear_individuo():
    pesos = np.random.dirichlet(np.ones(n_activos))
    return pesos

# Evaluar fitness: retorno esperado - aversión al riesgo
def evaluar_fitness(individuo, lambda_riesgo=0.5):
    retorno = np.dot(retornos_esperados, individuo)
    riesgo = np.dot(individuo.T, np.dot(matriz_covarianza, individuo))
    return retorno - lambda_riesgo * riesgo

# Selección por torneo
def seleccionar_padres(poblacion, fitnesses, k=3):
    indices = np.random.choice(len(poblacion), k)
    mejor = max(indices, key=lambda i: fitnesses[i])
    return poblacion[mejor]

# Cruce por promedio (simple)
def cruzar(padre1, padre2):
    hijo = (padre1 + padre2) / 2
    hijo /= np.sum(hijo)
    return hijo

# Mutación tipo swap
def mutacion_swap(individuo):
    hijo = individuo.copy()
    i, j = np.random.choice(len(hijo), size=2, replace=False)
    delta = np.random.uniform(0, min(hijo[i], 1 - hijo[j]))
    hijo[i] -= delta
    hijo[j] += delta
    return hijo

# Inicializar población
poblacion = [crear_individuo() for _ in range(tam_poblacion)]

# Evolución
for generacion in range(n_generaciones):
    fitnesses = [evaluar_fitness(ind, lambda_riesgo=0.5) for ind in poblacion]

    # Elitismo
    n_elite = int(tam_poblacion * elite_ratio)
    elite_indices = np.argsort(fitnesses)[-n_elite:]
    nueva_poblacion = [poblacion[i] for i in elite_indices]

    # Crear nuevos individuos
    while len(nueva_poblacion) < tam_poblacion:
        p1 = seleccionar_padres(poblacion, fitnesses)
        p2 = seleccionar_padres(poblacion, fitnesses)
        hijo = cruzar(p1, p2)

        if np.random.rand() < prob_mutacion:
            hijo = mutacion_swap(hijo)

        nueva_poblacion.append(hijo)

    poblacion = nueva_poblacion

# Evaluar resultados
fitnesses_finales = [evaluar_fitness(ind) for ind in poblacion]
mejor_indice = np.argmax(fitnesses_finales)
mejor_portafolio = poblacion[mejor_indice]

# Mostrar resultados
print("Mejor portafolio encontrado:")
for i, peso in enumerate(mejor_portafolio):
    print(f"Activo {i+1}: {peso:.2%}")
print(f"\nFitness final: {fitnesses_finales[mejor_indice]:.4f}")

