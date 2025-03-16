import simpy
import numpy as np
import scipy.stats as stats
import random
import copy
import statistics
from math import exp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

###################################
# 1) Statistics Helper
###################################

def calculate_statistics(data, confidence_level=0.95):
    if len(data) == 0:
        return 0.0, (0.0, 0.0)
    mean_val = np.mean(data)
    df = len(data) - 1
    if df > 0:
        se = stats.sem(data)
        ci = stats.t.interval(confidence_level, df, mean_val, se)
    else:
        ci = (mean_val, mean_val)
    return mean_val, ci

###################################
# 2) DVS Server
###################################

class DVSServer:
    """
    DVS Server with states: OFF, SETUP, BUSY, SCALED, IDLE
    Collects response times & power usage => E(R) & E(P).
    """
    def __init__(self, env, arrival_rate, setup_time, alpha,
                 scale_speed_factor, turn_on_threshold, scale_threshold,
                 service_rate, idle_power):
        self.env = env
        self.arrival_rate = arrival_rate
        self.setup_time = setup_time
        self.alpha = alpha
        self.scale_speed_factor = scale_speed_factor
        self.turn_on_threshold = turn_on_threshold
        self.scale_threshold = scale_threshold
        self.service_rate = service_rate
        self.idle_power = idle_power
        
        # Metrics
        self.response_times = []
        self.server_off_counter = 0
        self.jobs_slow_speed_counter = 0
        self.jobs_fast_speed_counter = 0
        
        # State mgmt
        self.state = 'OFF'
        self.job_queue = simpy.PriorityStore(env)
        self.server_process_instance = env.process(self.server_process())
        self.job_arrival_process_instance = env.process(self.job_arrival_process())
        
        # Time & power
        self.state_times = {'OFF':0,'SETUP':0,'BUSY':0,'SCALED':0,'IDLE':0}
        self.last_change = 0
        self.total_energy = 0
        self.powers = []

    def calculate_power(self):
        if self.state == 'OFF':
            return 0.0
        elif self.state in ['SETUP','BUSY']:
            return 1.0
        elif self.state == 'SCALED':
            # Example:  (0.3)^2 = 0.09
            return 0.3**2
        elif self.state == 'IDLE':
            return self.idle_power

    def server_process(self):
        while True:
            power = self.calculate_power()
            self.powers.append(power)
            dt = self.env.now - self.last_change
            self.total_energy += power*dt
            self.state_times[self.state] += dt
            self.last_change = self.env.now

            if self.state == 'OFF':
                if len(self.job_queue.items) >= self.turn_on_threshold:
                    yield from self.transition_to_setup()
                else:
                    try:
                        yield self.env.timeout(1)
                    except simpy.Interrupt:
                        pass

            elif self.state == 'SETUP':
                self.transition_to_busy_or_scaled()

            elif self.state in ['BUSY','SCALED']:
                if len(self.job_queue.items) > 0:
                    yield from self.process_job()
                else:
                    self.transition_to_idle()

            elif self.state == 'IDLE':
                yield from self.idle_to_off_or_busy()

    def transition_to_setup(self):
        try:
            yield self.env.timeout(self.setup_time)
        except simpy.Interrupt:
            pass
        self.state = 'SETUP'

    def transition_to_busy_or_scaled(self):
        if len(self.job_queue.items) >= self.scale_threshold:
            self.state = 'SCALED'
        else:
            self.state = 'BUSY'

    def transition_to_idle(self):
        self.state = 'IDLE'

    def process_job(self):
        job_arrival_time, = yield self.job_queue.get()
        dur = self.get_service_duration()
        yield self.env.timeout(dur)

        resp = self.env.now - job_arrival_time
        self.response_times.append(resp)

        if self.state == 'BUSY':
            self.jobs_slow_speed_counter += 1
        else:  # SCALED
            self.jobs_fast_speed_counter += 1

    def get_service_duration(self):
        if self.state == 'SCALED':
            return np.random.exponential(1.0/(self.service_rate*self.scale_speed_factor))
        else:
            return np.random.exponential(1.0/self.service_rate)

    def idle_to_off_or_busy(self):
        try:
            yield self.env.timeout(1/self.alpha)
            self.state = 'OFF'
            self.server_off_counter += 1
        except simpy.Interrupt:
            self.transition_to_busy_or_scaled()

    def job_arrival_process(self):
        while True:
            yield self.env.timeout(np.random.exponential(1.0/self.arrival_rate))
            self.job_queue.put((self.env.now,))
            if self.state in ['OFF','IDLE']:
                self.server_process_instance.interrupt()

    def run_simulation(self, run_time, warm_up=100):
        self.env.run(until=run_time)
        self.state_times[self.state] += self.env.now - self.last_change
        # Compute stats after warm-up
        E_R, _ = calculate_statistics(self.response_times[warm_up:])
        E_P, _ = calculate_statistics(self.powers[warm_up:])
        return E_R, E_P, self.jobs_slow_speed_counter, self.jobs_fast_speed_counter


###################################
# 3) Parameter / Individual Helpers
###################################

PARAM_RANGES = {
    "arrival_rate": [0.2, 0.5, 0.95],
    "setup_time": [1, 5, 10, 20],
    "scale_speed_factor": [1.5, 2, 3],
    "turn_on_threshold": list(range(1, 21)),
    "scale_threshold": list(range(1, 21)),
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "service_rate": [1.0],  # fixed
    "idle_power": [0.6]     # fixed
}

def random_individual():
    return {
        "arrival_rate":       random.choice(PARAM_RANGES["arrival_rate"]),
        "setup_time":         random.choice(PARAM_RANGES["setup_time"]),
        "scale_speed_factor": random.choice(PARAM_RANGES["scale_speed_factor"]),
        "turn_on_threshold":  random.choice(PARAM_RANGES["turn_on_threshold"]),
        "scale_threshold":    random.choice(PARAM_RANGES["scale_threshold"]),
        "alpha":              random.choice(PARAM_RANGES["alpha"]),
        "service_rate":       1.0,
        "idle_power":         0.6
    }

def encode_ind(ind):
    # For building a feature vector
    return np.array([
        ind["arrival_rate"],
        ind["setup_time"],
        ind["scale_speed_factor"],
        ind["turn_on_threshold"],
        ind["scale_threshold"],
        ind["alpha"]
    ])

def simulate_individual(params, sim_time=15000, warm_up=100):
    env = simpy.Environment()
    server = DVSServer(env, **params)
    E_R, E_P, s_count, f_count = server.run_simulation(sim_time, warm_up)
    fit = E_R * E_P
    return fit, E_R, E_P, s_count, f_count

##############################################
# 4) Niche Classification & SA Acceptance
##############################################

def get_niche_type(slow_jobs, fast_jobs):
    """
    Classify the individual's usage niche based on fraction of fast jobs.
    Niche A: fraction fast < 0.1  (mostly slow)
    Niche B: fraction fast > 0.9  (mostly fast)
    Niche C: fraction fast in [0.1, 0.9]
    """
    total_jobs = slow_jobs + fast_jobs
    if total_jobs == 0:
        # degenerate case: no jobs processed, treat as mostly slow
        return 'A'
    frac_fast = fast_jobs / float(total_jobs)
    if frac_fast < 0.1:
        return 'A'
    elif frac_fast > 0.9:
        return 'B'
    else:
        return 'C'

def acceptance_probability(old_fitness, new_fitness, temperature):
    """
    Simulated annealing style acceptance:
    Accept better solutions always.
    Accept worse solutions with probability exp((old - new)/T).
    """
    if new_fitness < old_fitness:
        return 1.0
    return exp((old_fitness - new_fitness)/max(temperature,1e-9))

###################################
# 5) BPM-GA: Multi-Seed + Niches + SA
###################################

def bpm_ga_run_single_seed(seed=0,
                          pop_size=30,
                          generations=30,
                          sim_time=15000,
                          warm_up=100,
                          niche_capacity=10,
                          T0=1.0,
                          Tfinal=0.001):
    """
    Run a single GA "ecosystem" with niche management and simulated annealing acceptance.
    Returns the best individual found + its fitness.
    """
    random.seed(seed)
    np.random.seed(seed)

    # We'll maintain three niche subpopulations: A, B, C
    # Each is a list of (individual, fitness, slow_count, fast_count)
    niches = {'A': [], 'B': [], 'C': []}

    # Helper to add a new evaluated solution to a niche
    def add_to_niche(ind, fit, s_count, f_count):
        niche_type = get_niche_type(s_count, f_count)
        niches[niche_type].append((ind, fit, s_count, f_count))
        # If niche is over capacity, remove worst
        if len(niches[niche_type]) > niche_capacity:
            niches[niche_type].sort(key=lambda x: x[1])  # sort by fitness ascending
            niches[niche_type].pop()  # remove worst (last)

    # Initialize population across all niches
    total_init = pop_size
    for _ in range(total_init):
        person = random_individual()
        fit, _, _, sc, fc = simulate_individual(person, sim_time, warm_up)
        add_to_niche(person, fit, sc, fc)

    best_global_fit = float('inf')
    best_global_ind = None

    # Evolve
    for g in range(generations):
        # Compute temperature
        frac = g / float(generations-1 if generations>1 else 1)
        temperature = T0 * (Tfinal / T0)**(frac)  # geometric schedule

        # Flatten population
        flat_pop = niches['A'] + niches['B'] + niches['C']
        # Current best in population
        flat_pop.sort(key=lambda x: x[1])
        current_best_fit = flat_pop[0][1]
        if current_best_fit < best_global_fit:
            best_global_fit = current_best_fit
            best_global_ind = flat_pop[0][0]

        print(f"[Seed {seed}] Generation {g} | Best so far: {best_global_fit}")

        # Reproduction
        new_offspring = []
        while len(new_offspring) < pop_size:
            # 1) Select random parents from the entire population (flat)
            p1 = random.choice(flat_pop)
            p2 = random.choice(flat_pop)
            parent1, fit1 = p1[0], p1[1]
            parent2, fit2 = p2[0], p2[1]

            # 2) Crossover
            child1, child2 = crossover(parent1, parent2)

            # 3) Mutation
            if random.random() < 0.3:  # higher mutation to explore
                child1 = mutate(child1)
            if random.random() < 0.3:
                child2 = mutate(child2)

            # 4) Evaluate child1
            f_child1, _, _, sc1, fc1 = simulate_individual(child1, sim_time, warm_up)
            # SA acceptance vs parent1
            if random.random() < acceptance_probability(fit1, f_child1, temperature):
                new_offspring.append((child1, f_child1, sc1, fc1))

            # 5) Evaluate child2
            f_child2, _, _, sc2, fc2 = simulate_individual(child2, sim_time, warm_up)
            # SA acceptance vs parent2
            if random.random() < acceptance_probability(fit2, f_child2, temperature):
                new_offspring.append((child2, f_child2, sc2, fc2))

        # Insert new offspring into niches
        # Clear old niches (or you can keep some fraction of old pop)
        # We'll keep the best fraction as well: let's keep the top ~1/3
        # of each niche from old population
        keep_fraction = 0.33
        for niche_type in ['A','B','C']:
            niches[niche_type].sort(key=lambda x: x[1])
            keep_n = int(keep_fraction * len(niches[niche_type]))
            old_keep = niches[niche_type][:keep_n]
            niches[niche_type] = old_keep

        # Now add new offspring
        for (ind_new, fit_new, sc_new, fc_new) in new_offspring:
            add_to_niche(ind_new, fit_new, sc_new, fc_new)

    return best_global_ind, best_global_fit

def crossover(p1, p2):
    c1, c2 = {}, {}
    keys = p1.keys()
    for k in keys:
        if random.random() < 0.5:
            c1[k] = p1[k]
            c2[k] = p2[k]
        else:
            c1[k] = p2[k]
            c2[k] = p1[k]
    return c1, c2

def mutate(ind):
    mutated = copy.deepcopy(ind)
    # pick a random param to mutate
    # (excluding service_rate, idle_power)
    mutate_keys = ["arrival_rate","setup_time","scale_speed_factor",
                   "turn_on_threshold","scale_threshold","alpha"]
    mk = random.choice(mutate_keys)
    mutated[mk] = random.choice(PARAM_RANGES[mk])
    return mutated

########################################
# 6) Local Neighborhood Search
########################################

def local_search(base_ind, sim_time=15000, warm_up=100, steps=2):
    """
    Tries small perturbations around base_ind in threshold-related parameters
    to see if there's a better solution. If it finds an improvement, it updates.
    Repeat for 'steps' times. 
    """
    best_ind = copy.deepcopy(base_ind)
    best_fit, _, _, _, _ = simulate_individual(best_ind, sim_time, warm_up)

    param_keys = ["turn_on_threshold","scale_threshold","setup_time","alpha"]
    for _ in range(steps):
        improved = False
        for pk in param_keys:
            # We'll try +/- 1 for thresholds, +/- factor of ~2 for alpha
            candidates = []
            val = best_ind[pk]
            if pk in ["turn_on_threshold","scale_threshold","setup_time"]:
                # integer-based
                for d in [-1, 1]:
                    neighbor_val = val + d
                    # ensure in range
                    if pk in ["turn_on_threshold","scale_threshold"]:
                        if 1 <= neighbor_val <= 20:
                            c = copy.deepcopy(best_ind)
                            c[pk] = neighbor_val
                            f,_,_,_,_ = simulate_individual(c, sim_time, warm_up)
                            candidates.append((c,f))
                    else:  # setup_time
                        if neighbor_val in PARAM_RANGES["setup_time"]:
                            c = copy.deepcopy(best_ind)
                            c[pk] = neighbor_val
                            f,_,_,_,_ = simulate_individual(c, sim_time, warm_up)
                            candidates.append((c,f))
            else:
                # alpha
                # we'll try picking next-lower or next-higher alpha in PARAM_RANGES
                # relative to the current alpha
                alpha_list = sorted(PARAM_RANGES["alpha"])
                idx = alpha_list.index(val) if val in alpha_list else -1
                if idx >= 0:
                    neighbors_idx = [idx-1, idx+1]
                    for ni in neighbors_idx:
                        if 0 <= ni < len(alpha_list):
                            c = copy.deepcopy(best_ind)
                            c[pk] = alpha_list[ni]
                            f,_,_,_,_ = simulate_individual(c, sim_time, warm_up)
                            candidates.append((c,f))

            # Among all candidates, if any is better, pick the best
            if candidates:
                candidates.sort(key=lambda x: x[1])
                if candidates[0][1] < best_fit:
                    best_fit = candidates[0][1]
                    best_ind = candidates[0][0]
                    improved = True
        if not improved:
            break

    return best_ind, best_fit

########################################
# 7) Running Multiple Seeds + Final
########################################

def run_bpm_ga_multi_seed(num_seeds=5, pop_size=30, generations=30, sim_time=15000, warm_up=100):
    """
    1) Runs BPM-GA for multiple seeds,
    2) Does local search on each best solution,
    3) Picks the global best among them.
    """
    best_overall_fit = float('inf')
    best_overall_ind = None

    for seed in range(num_seeds):
        print(f"\n=== Starting seed {seed} ===")
        bi, bf = bpm_ga_run_single_seed(seed, pop_size, generations, sim_time, warm_up)
        # local search
        print(f"[Seed {seed}] Local search on best from GA: fit={bf}")
        improved_ind, improved_fit = local_search(bi, sim_time, warm_up)

        if improved_fit < best_overall_fit:
            best_overall_fit = improved_fit
            best_overall_ind = improved_ind
        print(f"[Seed {seed}] Final best after local search: {improved_fit}")

    return best_overall_ind, best_overall_fit

########################################
# 8) Validation: Re-test & Main Execution
########################################

def re_test_config(config, runs=5, sim_time=15000, warm_up=100):
    """
    Repeatedly test the best config to see average performance.
    """
    fitness_vals=[]
    E_Rs=[]
    E_Ps=[]
    slow_counts=[]
    fast_counts=[]
    for _ in range(runs):
        fit,eR,eP,sl,fs = simulate_individual(config,sim_time,warm_up)
        fitness_vals.append(fit)
        E_Rs.append(eR)
        E_Ps.append(eP)
        slow_counts.append(sl)
        fast_counts.append(fs)
    avg_fit = np.mean(fitness_vals)
    std_fit = np.std(fitness_vals)
    avg_er = np.mean(E_Rs)
    avg_ep = np.mean(E_Ps)
    avg_slow = np.mean(slow_counts)
    avg_fast = np.mean(fast_counts)
    return avg_fit, std_fit, avg_er, avg_ep, avg_slow, avg_fast

if __name__ == "__main__":
    # Hyperparameters for our new BPM-GA approach
    NUM_SEEDS = 5       # how many random seeds (ecosystems)
    POP_SIZE = 30       # population size per seed
    GENERATIONS = 30    # number of generations
    SIM_TIME = 15000
    WARM_UP = 100

    # Run the multi-seed BPM-GA
    best_ind, best_fit = run_bpm_ga_multi_seed(num_seeds=NUM_SEEDS,
                                              pop_size=POP_SIZE,
                                              generations=GENERATIONS,
                                              sim_time=SIM_TIME,
                                              warm_up=WARM_UP)

    print("\n=== BPM-GA Completed Across All Seeds ===")
    print("Best Config Found =>", best_ind)
    print("Single-run Fit:", best_fit)

    # 2) Single-run stats for that best config
    single_run_fit,eR,eP,sl,fs = simulate_individual(best_ind,SIM_TIME,WARM_UP)
    print("\nSingle-run Detailed Stats:")
    print(f"Fit: {single_run_fit:.4f}, E(R): {eR:.4f}, E(P): {eP:.4f}")
    print(f"Slow Jobs: {sl}, Fast Jobs: {fs}")

    # 3) Re-test multiple times
    avg_fit, std_fit, avg_er, avg_ep, avg_slow, avg_fast = re_test_config(best_ind, runs=5, sim_time=SIM_TIME, warm_up=WARM_UP)
    print("\n=== Re-testing Best Config over 5 runs ===")
    print(f"Avg Fitness: {avg_fit:.4f} ± {std_fit:.4f}")
    print(f"Avg E(R): {avg_er:.4f}, Avg E(P): {avg_ep:.4f}")
    print(f"Avg Slow: {avg_slow:.2f}, Avg Fast: {avg_fast:.2f}")

    # 4) Does it genuinely use both speeds?
    if (avg_slow < 1e-9) or (avg_fast < 1e-9):
        print("\nConclusion: The best solution found uses effectively ONE speed.")
    else:
        print("\nConclusion: The best solution found uses BOTH speeds.")
