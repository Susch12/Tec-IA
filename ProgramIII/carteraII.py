import numpy as np

# ------------------ CONFIGURACIÓN ------------------
np.random.seed(55)  # Para reproducibilidad

n_activos = 2
retornos_esperados = np.random.uniform(0.05, 0.15, size=n_activos)

# Covarianza simétrica
matriz_covarianza = np.random.rand(n_activos, n_activos)
matriz_covarianza = (matriz_covarianza + matriz_covarianza.T) / 2
np.fill_diagonal(matriz_covarianza, 0.1)

tam_poblacion = 10
n_generaciones = int(input("Ingrese el número de generaciones: "))
prob_mutacion = 0.8
elite_ratio = 0.2

# ------------------ OPERACIONES BÁSICAS ------------------

def crear_individuo():
    pesos = np.random.dirichlet(np.ones(n_activos))
    return pesos

def evaluar_fitness(individuo, lambda_riesgo=0.5):
    retorno = np.dot(retornos_esperados, individuo)
    riesgo = np.dot(individuo.T, np.dot(matriz_covarianza, individuo))
    return retorno - lambda_riesgo * riesgo

def seleccionar_padres_ruleta(poblacion, fitnesses):
    fitnesses = np.array(fitnesses)
    min_fit = np.min(fitnesses)
    if min_fit < 0:
        fitnesses = fitnesses - min_fit + 1e-6
    total_fit = np.sum(fitnesses)
    probabilidades = fitnesses / total_fit if total_fit > 0 else np.ones(len(fitnesses)) / len(fitnesses)
    elegido_idx = np.random.choice(len(poblacion), p=probabilidades)
    return poblacion[elegido_idx]

# Cruza promedio
def cruzar_promedio(p1, p2):
    hijo = (p1 + p2) / 2
    return hijo / np.sum(hijo)

# Cruza aleatoria ponderada
def cruzar_ponderado(p1, p2):
    alpha = np.random.rand()
    hijo = alpha * p1 + (1 - alpha) * p2
    return hijo / np.sum(hijo)

# Mutación tipo swap (solo tiene sentido con 2 activos, intercambia pesos)
def mutacion_swap(individuo):
    hijo = individuo.copy()
    delta = np.random.uniform(0, hijo[0])
    hijo[0] -= delta
    hijo[1] += delta
    return hijo

# ------------------ EVOLUCIÓN ------------------

poblacion = [crear_individuo() for _ in range(tam_poblacion)]

for generacion in range(n_generaciones):
    fitnesses = [evaluar_fitness(ind, lambda_riesgo=0.5) for ind in poblacion]

    n_elite = 2
    elite_indices = np.argsort(fitnesses)[-n_elite:]
    nueva_poblacion = [poblacion[i] for i in elite_indices]

    while len(nueva_poblacion) < tam_poblacion:
        p1 = seleccionar_padres_ruleta(poblacion, fitnesses)
        p2 = seleccionar_padres_ruleta(poblacion, fitnesses)

        # Usar aleatoriamente uno de los dos tipos de cruce
        if np.random.rand() < 0.5:
            hijo = cruzar_promedio(p1, p2)
        else:
            hijo = cruzar_ponderado(p1, p2)

        if np.random.rand() < prob_mutacion:
            hijo = mutacion_swap(hijo)

        nueva_poblacion.append(hijo)

    poblacion = nueva_poblacion

# ------------------ RESULTADOS ------------------

fitnesses_finales = [evaluar_fitness(ind) for ind in poblacion]
mejor_indice = np.argmax(fitnesses_finales)
mejor_portafolio = poblacion[mejor_indice]

print("\n--- Individuos de la última generación ---")
for i, (ind, fit) in enumerate(zip(poblacion, fitnesses_finales), start=1):
    print(f"Individuo {i}: Activo A = {ind[0]:.2%}, Activo B = {ind[1]:.2%}, Fitness = {fit:.4f}")

print("\n--- Mejor portafolio encontrado ---")
print(f"Activo A: {mejor_portafolio[0]:.2%}")
print(f"Activo B: {mejor_portafolio[1]:.2%}")
print(f"Fitness final: {fitnesses_finales[mejor_indice]:.4f}")


