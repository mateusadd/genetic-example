import sys
import numpy as np

def fitness(equation_inputs, population):
    #calcular fitness da população atual
    #calcular soma dos produtos de w*x (população * inputs)
    return np.sum(population*equation_inputs, axis=1)

def selection(population, fitness, num_parents):
    #selecionar os melhores indivíduos para cruzamento
    #cria vetor com numero de genitores
    parents = np.empty((num_parents, population.shape[1]))
    
    #preenche vetor de genitores
    for idx in range(num_parents):
        #indice do fitness com maior valor
        max_fitness_idx = np.where(fitness == np.max(fitness)) 
        max_fitness_idx = max_fitness_idx[0][0]

        #na posição <idx>, coloca os elementos do fitness maximo na nova matriz <parents>
        parents[idx, :] = population[max_fitness_idx, :]#":" significa "pegar todos os elementos"
        fitness[max_fitness_idx] + sys.maxsize - 1

    return parents

def crossover(parents, generation_size):
    #gera crossover entre genitores
    #filhos terão h linhas X w colunas
    #parents: numpy array contendo genitores
    #generation_size: tupla de (h,w)

    #cria vetor de filhos
    offspring = np.empty(generation_size)
    
    #obtem ponto de corte
    crossover_point = np.uint8(generation_size[1]/2)#metade do número de colunas (tamanho do cromossomo)

    #iteração pelos genitores para gerar prole
    #itera pela quantidade de filhos a serem gerados
    for idx in range(generation_size[0]):
        #indice 1o genitor
        p1_idx = idx % parents.shape[0]
        #indice 2o genitor
        p2_idx = (idx+1) % parents.shape[0]

        #o novo filho terá a primeira metade de seus genes oriundo do 1o genitor
        #posição 0 até <crossover_point>, genes do 1o genitor inseridos
        offspring[idx, 0:crossover_point] = parents[p1_idx, 0:crossover_point] 

        #o novo filho terá a segunda metade de seus genes oriundo do 2o genitor
        #posição <crossover_point> até o fim, genes do 2o genitor inseridos
        offspring[idx, crossover_point:] = parents[p2_idx, crossover_point:]

    return offspring 

def mutation(offspring):
    #altera um gene aleatório de cada filho
    for idx in range(offspring.shape[0]):
        #gerar valor aleatório
        random_value = np.random.uniform(-1.0, 1.0, 1)
        #obtem gene aleatorio (valor entre 0 e nr de colunas)
        random_idx = np.random.randint(offspring.shape[1])

        #altera gene random_idx do filho idx
        offspring[idx, random_idx] = offspring[idx, random_idx] + random_value
    
    return offspring