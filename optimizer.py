import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, generations):
        self.population_size = population_size
        self.generations = generations
        
        # Genes: [Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x]
        
        self.population = np.random.uniform(-1.0, 1.0, (population_size, 6))
        
        # Theta: +/- 100, 50, 50
        self.population[:, 0] *= 100.0 # Kp_th
        self.population[:, 1] *= 50.0  # Ki_th
        self.population[:, 2] *= 50.0  # Kd_th
        
        # Position: +/- 50, 20, 20
        self.population[:, 3] *= 50.0  # Kp_x
        self.population[:, 4] *= 20.0  # Ki_x
        self.population[:, 5] *= 20.0  # Kd_x
        
        self.fitness_history = []
        self.population_history = [self.population.copy()]
        self.cost_history = []

    def select(self, population, fitness, elitism_count):
        """Returns elites and sorted population."""
        # Sort by cost 
        sorted_indices = np.argsort(fitness)
        sorted_pop = population[sorted_indices]
        
        elites = sorted_pop[:elitism_count]
        
        return elites, sorted_pop

    def crossover(self, parent1, parent2):
        """Mixes genes."""
        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

    def mutate(self, individual, mutation_rate):
        """Mutate individual with random noise"""
        if np.random.rand() < mutation_rate:
            # Noise
            noise = np.random.normal(0, 5.0, 6) 
            individual += noise
            
            # Theta
            individual[0] = np.clip(individual[0], -100, 100)
            individual[1] = np.clip(individual[1], -50, 50)
            individual[2] = np.clip(individual[2], -50, 50)
            # Position
            individual[3] = np.clip(individual[3], -50, 50)
            individual[4] = np.clip(individual[4], -20, 20)
            individual[5] = np.clip(individual[5], -20, 20)
            
        return individual

    def evolve(self, costs, elitism_rate=0.1, crossover_rate=0.8, mutation_rate=0.2, target_pop_size=None):
        """
        Performs one step of evolution with configurable rates.
        """
        if target_pop_size is None:
            target_pop_size = self.population_size

        elitism_count = max(1, int(elitism_rate * target_pop_size))
        
        # Select from CURRENT population
        elites, sorted_pop = self.select(self.population, costs, elitism_count)
        
        new_population = list(elites)
        
        # (Top 50% of CURRENT)
        pool_size = max(2, self.population_size // 2)
        pool = sorted_pop[:pool_size]
        
        while len(new_population) < target_pop_size:
            # Selection
            idx1, idx2 = np.random.randint(0, pool_size, 2)
            p1 = pool[idx1]
            p2 = pool[idx2]
            
            # Crossover
            if np.random.rand() < crossover_rate:
                child = self.crossover(p1, p2)
            else:
                # Replication (Direct copy of one parent if no crossover)
                child = p1.copy()
            
            # Mutation
            child = self.mutate(child, mutation_rate)
            
            new_population.append(child)
            
        self.population = np.array(new_population[:target_pop_size])
        self.population_size = target_pop_size # Update internal size
        
        best_cost = np.min(costs)
        self.fitness_history.append(best_cost)
        self.population_history.append(self.population.copy())
        self.cost_history.append(costs.copy()) # Store costs of this generation
        
        return self.population, best_cost
