# y = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6

from tkinter import OFF
import numpy as np
import ga

def main():
    #variáveis x1
    equation_inputs = [4, -2, 3.5, 5, -11, -4.7] #new ArrayList<>()
    
    #numero de pesos (w)
    num_weights = 6 #nr genes
    
    #tamanho da população
    solution_per_population = 8 #nr cromossomos
    
    #conjunto 8X6
    population_size = (solution_per_population, num_weights)

    #população inicial
    population = np.random.uniform(low=-4.0, high=4.0, size=population_size) #numero aleatorio (float) entre valores
    print("Initial population:")
    print(population)

    #nr gerações
    num_generations = 5
    #nr genitores
    num_parents_crossover = 4

    #para cada geração
    for generation in range(num_generations):
        print(f"Generation {generation}:")
        #calcular fitness
        fitness = ga.fitness(equation_inputs, population)
        print("Fitness:")
        print(fitness)

        #selecionar melhores
        selected_parents = ga.selection(population, fitness, num_parents_crossover)
        print("Selected parents:")
        print(selected_parents)

        #crossover entre melhores
        offspring_crossover = ga.crossover(selected_parents, (solution_per_population-num_parents_crossover, num_weights))
        print("\nOffspring:")
        print(offspring_crossover)

        #adicionar mutação
        offspring_mutation = ga.mutation(offspring_crossover)
        print("\nMutations")
        print(offspring_mutation)

        #criar nova população
        #elite
        population[0:selected_parents.shape[0], :] = selected_parents
        #crossover + mutation
        population[selected_parents.shape[0]:, :] = offspring_mutation

        print("\nNew population:")
        print(population)

        print("Best result: ", np.max(ga.fitness(equation_inputs, population)))

    fitness = ga.fitness(equation_inputs, population)
    best_fit_idx = np.where(fitness == np.max(fitness))

    print("Best fitness: ", population[best_fit_idx, :])
    print("Fitness of the best: ", fitness[best_fit_idx])

#se executar o script, chama main()
if __name__ == '__main__':
    main()