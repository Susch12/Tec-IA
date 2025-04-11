import numpy as np

class AlgoritmoGeneraciones:
    def __init__(self):
        self.n_activos = 2
        self.retornos_esperados = np.array([0.08, 0.06])
        self.matriz_covarianza = np.array([
            [0.05, 0.01],
            [0.01, 0.08]
        ])
        self.tam_poblacion = 10
        self.prob_mutacion = 0.8
        self.n_elite = 2

    def crear_individuo(self):
        pesos = np.random.dirichlet(np.ones(self.n_activos))
        return pesos

    def evaluar_fitness(self, individuo):
        retorno = np.dot(self.retornos_esperados, individuo)
        riesgo = np.dot(individuo.T, np.dot(self.matriz_covarianza, individuo))
        epsilon = 1e-9  # Para evitar división por cero
        riesgo = max(riesgo, epsilon)
        return retorno / riesgo

    def seleccionar_padres_ruleta(self, poblacion, fitnesses):
        fitnesses = np.array(fitnesses)
        min_fit = np.min(fitnesses)
        if min_fit < 0:
            fitnesses = fitnesses - min_fit + 1e-6
        total_fit = np.sum(fitnesses)
        probabilidades = fitnesses / total_fit if total_fit > 0 else np.ones(len(fitnesses)) / len(fitnesses)
        elegido_idx = np.random.choice(len(poblacion), p=probabilidades)
        return poblacion[elegido_idx]

    def cruzar_promedio(self, p1, p2):
        hijo = (p1 + p2) / 2
        return hijo / np.sum(hijo)

    def cruzar_ponderado(self, p1, p2):
        alpha = np.random.rand()
        hijo = alpha * p1 + (1 - alpha) * p2
        return hijo / np.sum(hijo)
    
    def cruzar_BLX(self, p1, p2, alpha=0.5):
        hijo = []
        for gene1, gene2 in zip(p1, p2):
            c_min = min(gene1, gene2)
            c_max = max(gene1, gene2)
            delta = c_max - c_min
            lower_bound = c_min - alpha * delta
            upper_bound = c_max + alpha * delta
            gene = np.random.uniform(lower_bound, upper_bound)
            hijo.append(gene)
        hijo = np.array(hijo)
        hijo = np.maximum(hijo, 0)  # Evitar valores negativos
        if np.sum(hijo) > 0:
            hijo = hijo / np.sum(hijo)
        else:
            hijo = self.crear_individuo()  # En caso de tener suma cero
        return hijo

    def mutacion_gaussiana(self, individuo, sigma=0.05):
        hijo = individuo.copy()
        hijo += np.random.normal(0, sigma, size=hijo.shape)
        hijo = np.maximum(hijo, 0)  # Evitar pesos negativos
        if np.sum(hijo) > 0:
            hijo = hijo / np.sum(hijo)
        else:
            hijo = self.crear_individuo()  # Fallback en caso de suma 0
        return hijo

    def run(self, n_generaciones):
        poblacion = [self.crear_individuo() for _ in range(self.tam_poblacion)]

        generaciones_x = []
        generaciones_y = []
        generaciones_z = []

        for generacion in range(n_generaciones):
            fitnesses = [self.evaluar_fitness(ind) for ind in poblacion]

            generaciones_x.append([ind[0] for ind in poblacion])
            generaciones_y.append([ind[1] for ind in poblacion])
            generaciones_z.append(fitnesses)

            # Conservamos la elite
            elite_indices = np.argsort(fitnesses)[-self.n_elite:]
            nueva_poblacion = [poblacion[i] for i in elite_indices]

            # Generamos nuevos individuos hasta completar la población
            while len(nueva_poblacion) < self.tam_poblacion:
                p1 = self.seleccionar_padres_ruleta(poblacion, fitnesses)
                p2 = self.seleccionar_padres_ruleta(poblacion, fitnesses)

                # Seleccionamos aleatoriamente el operador de cruzamiento
                op = np.random.rand()
                if op < 0.33:
                    hijo = self.cruzar_promedio(p1, p2)
                elif op < 0.66:
                    hijo = self.cruzar_ponderado(p1, p2)
                else:
                    hijo = self.cruzar_BLX(p1, p2)

                # Aplicamos la mutación con cierta probabilidad
                if np.random.rand() < self.prob_mutacion:
                    hijo = self.mutacion_gaussiana(hijo)

                nueva_poblacion.append(hijo)

            poblacion = nueva_poblacion

        return generaciones_x, generaciones_y, generaciones_z
