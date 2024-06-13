import random
import math

num_classes = 5
num_timeslots = 3
num_rooms = 3

def evaluate_solution(solution):
    conflicts = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if solution[i][0] == solution[j][0]:  
                if solution[i][1] == solution[j][1]:
                    conflicts += 1

    return conflicts

def generate_neighbor(solution):
    neighbor = [list(item) for item in solution]  # Copia profunda de la solución actual
    class1, class2 = random.sample(range(num_classes), 2)
    neighbor[class1][1], neighbor[class2][1] = neighbor[class2][1], neighbor[class1][1]
    return neighbor

def simulated_annealing(initial_solution, initial_temp, cooling_rate, num_iterations):
    current_solution = initial_solution
    best_solution = current_solution
    current_cost = evaluate_solution(current_solution)
    best_cost = current_cost
    temp = initial_temp
    iteration = 0
    
    while temp > 1e-3 and iteration < num_iterations:
        neighbor_solution = generate_neighbor(current_solution)
        neighbor_cost = evaluate_solution(neighbor_solution)
        
        if neighbor_cost < current_cost:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
        else:
            delta_cost = neighbor_cost - current_cost
            if random.random() < math.exp(-delta_cost / temp):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
        
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
        
        temp *= cooling_rate
        iteration += 1
    
    return best_solution, best_cost

initial_solution = [[0, 0], [1, 1], [2, 2], [0, 1], [1, 0]]  #(clase, horario)
initial_temp = 100.0
cooling_rate = 0.95
num_iterations = 1000

best_solution, best_cost = simulated_annealing(initial_solution, initial_temp, cooling_rate, num_iterations)

print("Mejor solución encontrada:", best_solution)
print("Costo de la mejor solución:", best_cost)
