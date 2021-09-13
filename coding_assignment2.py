import numpy as np
import matplotlib.pyplot as plt

# taking inputs from user
ob_function = input('Enter objective function :')
n = int(input('Enter population size :'))
p_c = float(input('Enter crossover probability :'))
p_m = float(input('Enter mutation probability :'))

# declaration of empty matrix for plotting graphs
generations = np.zeros(100)

for i in range(len(generations)):
    generations[i] = generations[i] + i

mean_fitness = np.zeros(len(generations))
max_fitness = np.zeros(len(generations))
min_fitness = np.zeros(len(generations))

# Assigning design variables
x1min = 0.0
x1max = 0.5
x2min = 0.0
x2max = 0.5

# For two decimal points accuracy string length should be l1 + l2 = string length
l1 = 12      # length of sub string s1
l2 = 12      # length of sub string s2
string_len = l1 + l2
np.random.seed(1)
population = np.random.uniform(0, 1, (n, string_len))
population[population > 0.5] = 1
population[population <= 0.5] = 0
population = population.astype(int)
max_population = np.zeros([len(generations), string_len])


# getting decoded value
def decoded(arr, m):
    global x1min, x2min, x1max, x2max, l1, l2, string_len
    x_values = np.zeros([m, 2])
    x = np.zeros([m, 2], dtype=int)
    # converting binary to decimal and getting decoded s1 value
    for i in range(len(arr)):
        count = 0
        for j in range(l1 - 1, -1, -1):
            if arr[i][j] == 1:
                x[i][0] += 2 ** count
            count += 1

    # converting binary to decimal and getting decoded s2 value
    for i in range(len(arr)):
        count = 0
        for j in range(string_len - 1, l1 - 1, -1):
            if arr[i][j] == 1:
                x[i][1] += 2 ** count
            count += 1

    # getting x1 and x2 values
    for i in range(len(arr)):
        x_values[i][0] = x1min + ((x1max - x1min) / (pow(2, l1) - 1)) * x[i][0]

    for i in range(len(arr)):
        x_values[i][1] = x2min + ((x2max - x2min) / (pow(2, l2) - 1)) * x[i][1]

    return x_values


# Getting fitness values
def fitness(x_values, m):
    global ob_function
    fit = np.zeros((m, 1))
    for i in range(m):
        x1 = x_values[i][0]
        x2 = x_values[i][1]
        f = eval(ob_function)
        # converting minimization function to maximization
        fit[i][0] = 1 / (1 + (f * f))
    return fit


# For getting optimal solution
def final(fitness_value):
    global generations
    solutions = np.zeros(len(generations))
    for b in range(len(generations)):
        a = (1/fitness_value[b]) - 1
        solutions[b] = pow(a, 0.5) + solutions[b]
    return solutions


# reproduction using roulette wheel scheme
def reproduction(fit, m, population_sample):
    global string_len
    total = 0
    for i in range(m):
        total = total + fit[i][0]
    new_population = np.zeros([m, string_len])
    prob = np.zeros([m, 1])
    p = np.zeros([m, 1])
    # getting probability
    for i in range(len(prob)):
        prob[i][0] += fit[i][0] / total
        if i == 0:
            p[i][0] = 0
        else:
            p[i][0] = prob[i][0] + p[i - 1][0]

    # random selection
    r = np.random.rand(m)
    for i in range(m):
        for j in range(m):
            if r[i] <= p[j][0]:
                new_population[i, :] = population_sample[j - 1, :]
                break
        if r[i] >= p[m - 1][0]:
            new_population[i, :] = population_sample[m - 1, :]

    return new_population


# for swapping of binary digits
def two_point(s1, s2, ln):
    r = np.random.randint(low=0, high=ln, size=(2,))
    for i in range(min(r), max(r)):
        s1[i], s2[i] = s2[i], s1[i]
    return s1, s2


# using two point crossover method
def crossover(pool, m):
    global string_len, p_c
    new_pool = np.zeros([m, string_len])
    # shuffle the mating pool and selecting pair from top to bottom
    np.random.shuffle(pool)
    for i in range(0, m, 2):
        r = np.random.rand(1)
        if r <= p_c:
            new_pool[i, :], new_pool[i + 1, :] = two_point(pool[i, :], pool[i + 1, :], string_len)
        else:
            continue
    return new_pool


# using bit-wise mutation
def mutation(pool, m):
    global p_m, string_len
    for i in range(m):
        for j in range(string_len):
            r = np.random.rand(1)
            if r <= p_m:
                if pool[i][j] == 0:
                    pool[i][j] = 1
                else:
                    pool[i][j] = 0
    return pool


optimal = np.zeros(len(generations))
maximum = 0
# training of GA
for e in range(len(generations)):
    x = decoded(population, n)
    fitn = fitness(x, n)
    if max(fitn) > maximum:
        maximum = max(fitn)
    optimal[e] = maximum
    # maximum = max(fitn)
    mean_fitness[e] = mean_fitness[e] + np.mean(fitn)
    max_fitness[e] = max_fitness[e] + max(fitn)
    min_fitness[e] = min_fitness[e] + min(fitn)
    max_population[e, :] = population[np.where(fitn == max(fitn))[0][0], :]
    mating_pool = reproduction(fitn, n, population)
    new_mating_pool = crossover(mating_pool, n)
    new_mating_pool = new_mating_pool.astype(int)
    population = mutation(new_mating_pool, n)

# getting x1 and x2 values corresponding to maximum fitness value of each generation
xvalues = decoded(max_population, len(generations))
print('Solution :', end='\t')
print(xvalues[len(generations) - 1, :])
optimal_solution = final(optimal)
plt.plot(generations, optimal_solution)
plt.show()

# Plotting graphs of number of generations Vs mean, max and min fitness value
plt.plot(generations, mean_fitness, 'r')
plt.plot(generations, max_fitness, 'y')
plt.plot(generations, min_fitness, 'b')
plt.xlabel('Number of generations')
plt.ylabel('mean, max and min of fitness')
plt.legend(['mean fitness', 'maximum fitness', 'minimum fitness'], loc='upper right')

plt.show()
