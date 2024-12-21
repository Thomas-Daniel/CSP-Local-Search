import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import random
import matplotlib.pyplot as plt

# 1. Dataset Creation
# Parameters
n_samples = 1000
centers = 2
n_features = 4
random_state = 42

# Generate synthetic dataset
X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=random_state)
# Convert labels to binary (0 and 1)
y = y.reshape(-1, 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Neural Network Design
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Initialize weights and biases
        self.layers = []
        self.weights = []
        self.biases = []
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(self.layer_sizes)-1):
            # Initialize weights with small random values
            weight = np.random.uniform(-1, 1, (self.layer_sizes[i], self.layer_sizes[i+1]))
            bias = np.random.uniform(-1, 1, (1, self.layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X, weights, biases):
        a = X
        for w, b in zip(weights, biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
        return a

# 3. GA Implementation
def initialize_population(pop_size, genome_length):
    population = []
    for _ in range(pop_size):
        # Initialize genome with random values between -1 and 1
        genome = np.random.uniform(-1, 1, genome_length)
        population.append(genome)
    return population

def decode_genome(genome, layer_sizes):
    weights = []
    biases = []
    idx = 0
    for i in range(len(layer_sizes)-1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i+1]
        # Extract weights
        w_size = input_size * output_size
        w = genome[idx:idx+w_size].reshape(input_size, output_size)
        weights.append(w)
        idx += w_size
        # Extract biases
        b_size = output_size
        b = genome[idx:idx+b_size].reshape(1, output_size)
        biases.append(b)
        idx += b_size
    return weights, biases

def calculate_fitness(individual, nn, X, y):
    weights, biases = decode_genome(individual, nn.layer_sizes)
    predictions = nn.forward(X, weights, biases)
    mse = np.mean((predictions - y) ** 2)
    # Fitness is inversely related to MSE
    fitness = 1 / (mse + 1e-6)  # Add small value to prevent division by zero
    return fitness

def tournament_selection(population, fitnesses, tournament_size=3):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        # Randomly select individuals for the tournament
        participants = random.sample(range(pop_size), tournament_size)
        # Select the best among them
        best = participants[0]
        for p in participants[1:]:
            if fitnesses[p] > fitnesses[best]:
                best = p
        selected.append(population[best])
    return selected

def two_point_crossover(parent1, parent2):
    size = len(parent1)
    if size < 2:
        return parent1.copy(), parent2.copy()
    # Choose two crossover points
    pt1, pt2 = sorted(random.sample(range(size), 2))
    # Create offspring
    offspring1 = np.concatenate([parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]])
    offspring2 = np.concatenate([parent2[:pt1], parent1[pt1:pt2], parent2[pt2:]])
    return offspring1, offspring2

def mutate(individual, mutation_rate=0.1, mutation_scale=0.5):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Add Gaussian noise
            individual[i] += np.random.normal(0, mutation_scale)
    return individual

def run_genetic_algorithm(nn, X, y, population_size=100, generations=200, mutation_rate=0.1, elitism=10):
    # Determine genome length
    genome_length = 0
    for i in range(len(nn.layer_sizes)-1):
        genome_length += nn.layer_sizes[i] * nn.layer_sizes[i+1]  # Weights
        genome_length += nn.layer_sizes[i+1]  # Biases
    
    # Initialize population
    population = initialize_population(population_size, genome_length)
    
    best_mse_history = []
    best_individual = None
    best_mse = float('inf')
    
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [calculate_fitness(ind, nn, X, y) for ind in population]
        # Convert fitness to MSE
        mses = [1/f for f in fitnesses]
        # Track best individual
        min_mse = min(mses)
        min_index = mses.index(min_mse)
        if min_mse < best_mse:
            best_mse = min_mse
            best_individual = population[min_index]
        best_mse_history.append(best_mse)
        
        if (gen+1) % 10 == 0 or gen == 0:
            print(f"Generation {gen+1}: Best MSE = {best_mse}")
        
        # Selection
        selected = tournament_selection(population, fitnesses)
        
        # Crossover
        offspring = []
        for i in range(0, population_size - elitism, 2):
            parent1 = selected[i]
            parent2 = selected[i+1]
            child1, child2 = two_point_crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        
        # Mutation
        offspring = [mutate(child, mutation_rate=mutation_rate) for child in offspring]
        
        # Elitism: retain top elites
        elites_indices = np.argsort(mses)[:elitism]
        elites = [population[i] for i in elites_indices]
        
        # Create new population
        population = offspring + elites
    
    return best_individual, best_mse_history

# 4 & 5. GA Parameters and Optimization Process
# Initialize neural network
nn = NeuralNetwork(input_size=4, hidden_sizes=[16, 16], output_size=1)

# Run GA
best_genome, mse_history = run_genetic_algorithm(
    nn=nn,
    X=X_train,
    y=y_train,
    population_size=100,
    generations=200,
    mutation_rate=0.1,
    elitism=10
)

# Decode best genome
best_weights, best_biases = decode_genome(best_genome, nn.layer_sizes)

# Compute final MSE on training set
final_predictions = nn.forward(X_train, best_weights, best_biases)
final_mse = np.mean((final_predictions - y_train) ** 2)
print(f"\nFinal MSE on Training Set: {final_mse}")

# Compute MSE on test set
test_predictions = nn.forward(X_test, best_weights, best_biases)
test_mse = np.mean((test_predictions - y_test) ** 2)
print(f"Final MSE on Test Set: {test_mse}")

# Plot MSE over generations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mse_history)+1), mse_history, label='Training MSE')
plt.xlabel('Generation')
plt.ylabel('Mean Squared Error')
plt.title('MSE Over Generations')
plt.legend()
plt.grid(True)
plt.show()
