#!/usr/bin/env python3
import heapq
def define_state_initial(N):
    """
    Genera el estado inicial para N discos (todos en la torre 0).
    N discos numerados de mayor a menor: [N, N-1, ..., 1]
    Retorna una tupla de 3 tuplas, una por cada torre.
    """
    # torre0 con los discos: (N, N-1, ..., 1)
    torre0 = tuple(range(N, 0, -1))
    torre1 = ()
    torre2 = ()
    return (torre0, torre1, torre2)


def define_state_final(N):
    """
    Genera el estado final para N discos (todos en la torre 2).
    Retorna una tupla de 3 tuplas.
    """
    torre0 = ()
    torre1 = ()
    torre2 = tuple(range(N, 0, -1))
    return (torre0, torre1, torre2)


def funcion_objetivo(estado):
    """
    Verifica si todos los discos están en la torre 2.
    Es decir, torres 0 y 1 vacías.
    """
    torre0, torre1, torre2 = estado
    return len(torre0) == 0 and len(torre1) == 0


def funcion_sucesor(estado):
    sucesores = []
    torre0, torre1, torre2 = estado
    torres = [list(torre0), list(torre1), list(torre2)]
    
    for i in range(3):
        for j in range(3):
            if i != j:
                origen = torres[i]
                destino = torres[j]
                if len(origen) > 0:
                    disco_tope = origen[-1]
                    if (len(destino) == 0) or (destino[-1] > disco_tope):
                        nuevo_torres = [list(t) for t in torres]
                        nuevo_torres[i].pop()
                        nuevo_torres[j].append(disco_tope)
                        nuevo_estado = tuple(tuple(t) for t in nuevo_torres)
                        sucesores.append((nuevo_estado, 1))  # Agregamos costo
    return sucesores



def funcion_heuristica(estado):
    """
    Estima el número de movimientos restantes para resolver el problema.
    Considera la distancia de cada disco a la torre2 y si está bloqueado.
    """
    torre0, torre1, torre2 = estado
    total = 0
    N = len(torre0) + len(torre1) + len(torre2)

    # Evaluamos cada disco
    for i in range(N, 0, -1):
        if i in torre0:
            index = torre0.index(i)
            total += 2 ** index  # Se requieren al menos 2^index movimientos
        elif i in torre1:
            index = torre1.index(i)
            total += 2 ** index  # Similar a torre0
        # Si está en torre2, ya está en su posición final (0 costo)

    return total

def reconstruir_camino(padres, estado_meta):
    """
    Dado un diccionario padres que mapea estado_hijo -> estado_padre,
    y un estado_meta, reconstruimos el camino (lista de estados)
    desde el inicio hasta la meta.
    """
    camino = []
    actual = estado_meta
    while actual in padres:
        camino.append(actual)
        actual = padres[actual]
    camino.append(actual)  # el estado inicial
    camino.reverse()
    return camino

def a_star(start, goal, heuristic, neighbors):
    """
    Algoritmo A* optimizado para las Torres de Hanói.
    
    start: estado inicial
    goal: estado objetivo
    heuristic: función heurística h(n)
    neighbors: función que devuelve vecinos y sus costos
    """
    pq = []  # Cola de prioridad
    heapq.heappush(pq, (heuristic(start), 0, start, []))  # (f(n), g(n), nodo, camino)
    
    visitados = {}  # Diccionario de costos más bajos vistos
    
    while pq:
        f, g, node, path = heapq.heappop(pq)  # Extrae el mejor nodo
        
        if node == goal:  # Si llegamos a la meta
            return path + [node], g
        
        if node in visitados and visitados[node] <= g:
            continue  # No expandimos si ya hay un camino mejor
        
        visitados[node] = g  # Guardamos el mejor costo hasta el momento
        
        for neighbor, cost in neighbors(node):
            new_g = g + cost  # Costo acumulado (siempre +1 en este problema)
            new_f = new_g + heuristic(neighbor)  # f(n) = g(n) + h(n)
            heapq.heappush(pq, (new_f, new_g, neighbor, path + [node]))
    
    return None, float('inf')  # Si no se encuentra solución



def print_state(estado):
    """
    Muestra un estado en consola de forma textual.
    Ejemplo:
      Torre 0: [7, 6, 5]
      Torre 1: [4, 3]
      Torre 2: [2, 1]
    """
    torre0, torre1, torre2 = estado
    print(f"Torre 0: {list(torre0)}")
    print(f"Torre 1: {list(torre1)}")
    print(f"Torre 2: {list(torre2)}")
    print("-" * 40)


if __name__ == "__main__":
    # 7 discos, 3 torres
    N = 8
    
    # Definimos estados inicial y final
    inicio = define_state_initial(N)
    objetivo = define_state_final(N)
    
    print("Estado inicial:")
    print_state(inicio)
    print("Estado final esperado:")
    print_state(objetivo)
    
    resultado = a_star(inicio, objetivo, funcion_heuristica, funcion_sucesor)

    if resultado is None:
        print("No se encontró una solución.")
    else:
        camino_sol, costo_total = resultado  # Separamos la tupla

        print(f"¡Solución encontrada!\nNúmero de movimientos (costo): {costo_total}")
        print("Mostrando el camino solución:")

        for i, st in enumerate(camino_sol):
            print(f"Paso {i}:")
            print_state(st)
        print("Fin de la ruta solución.")

