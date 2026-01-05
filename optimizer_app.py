import numpy as np
import threading
import time
import mujoco
import mujoco.viewer
from environment import Simulation
from controller import DualPIDController
from optimizer import GeneticAlgorithm
import json

class OptimizerApp:
    def __init__(self):
        self.stop_event = threading.Event()
        self.ga = None
        self.best_genes = None
        self.worst_genes = None
        self.worst_cost = -1.0

    def evaluate_controller(self, genes, max_force=100.0, sim_time=10.0, dt=0.01):
        """Runs a simulation for a single set of PID gains."""
        # Unpack 6 genes
        Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x = genes
        pid = DualPIDController(Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x)
        
        try:
             sim = Simulation(render_mode=False, max_force=max_force)
        except Exception as e:
            print(f"Simulation Init Error: {e}")
            return 1e6 #high cost on failure

        state = sim.reset()
        total_cost = 0.0
        steps = int(sim_time / dt)
        
        pid.reset()
        
        for _ in range(steps):
            # state: [x, theta, x_dot, theta_dot]
            x = state[0]
            theta = state[1]
            
            error_x = 0.0 - x
            error_th = 0.0 - theta
            
            action = pid.get_action(error_th, error_x, dt)
            
            next_state, cost, done = sim.step(action)
            total_cost += cost
            
            state = next_state
            if done: 
                break
                
        return total_cost

    def visualize_controller(self, genes, max_force=100.0, duration=5.0):
        """Visualizes a controller using MuJoCo Viewer and returns trajectory."""
        Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x = genes
        pid = DualPIDController(Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x)
        sim = Simulation(max_force=max_force) 
        
        print(f"Vis: Th[{Kp_th:.1f}, {Ki_th:.1f}, {Kd_th:.1f}] X[{Kp_x:.1f}, {Ki_x:.1f}, {Kd_x:.1f}]")
        
        states = []
        actions = []
        
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            state = sim.reset()
            pid.reset()
            start_time = time.time()
            
            while viewer.is_running() and (time.time() - start_time < duration):
                step_start = time.time()
                
                # Capture State BEFORE step for plotting
                states.append(state)
                
                x = state[0]
                theta = state[1]
                
                error_x = 0.0 - x
                error_th = 0.0 - theta
                
                action = pid.get_action(error_th, error_x, 0.01) # Assuming fixed dt=0.01
                actions.append(action)
                
                state, _, _ = sim.step(action)
                
                viewer.sync()
                
                time_until_next_step = sim.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        return np.array(states), np.array(actions)

    def train(self, pop_size, generations, elitism_rate, crossover_rate, mutation_rate, max_force, 
              large_init_enabled=False, large_init_size=100, callback_epoch=None):
        """
        Runs the GA training loop.
        """
        self.stop_event.clear()
        print("Initializing GA...")
        
        # initial population size
        current_pop_size = large_init_size if large_init_enabled else pop_size
        
        self.ga = GeneticAlgorithm(current_pop_size, generations)
        
        for gen in range(generations):
            if self.stop_event.is_set():
                print("Training stopped by user.")
                break
                
            costs = []
            for individual in self.ga.population:
                cost = self.evaluate_controller(individual, max_force=max_force)
                costs.append(cost)
            
            # Target size for NEXT generation is always the standard pop_size
            # (unless we want to keep it large, but usually large init means shrink after - big initial population is expensive)
            target_size = pop_size
            
            # Evolve with dynamic parameters
            population, best_cost = self.ga.evolve(
                np.array(costs), 
                elitism_rate=elitism_rate,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                target_pop_size=target_size
            )
            
            # Find best of this generation
            best_idx = np.argmin(costs)
            best_genes = self.ga.population[best_idx] # Note: this is from NEW pop? No.
        
            best_idx = np.argmin(costs)

            pass 
            
    def train(self, pop_size, generations, elitism_rate, crossover_rate, mutation_rate, max_force, 
              large_init_enabled=False, large_init_size=100, callback_epoch=None):
        """
        Runs the GA training loop.
        """
        self.stop_event.clear()
        print("Initializing GA...")
        
        # Determine initial population size
        current_pop_size = large_init_size if large_init_enabled else pop_size
        
        self.ga = GeneticAlgorithm(current_pop_size, generations)
        
        for gen in range(generations):
            if self.stop_event.is_set():
                print("Training stopped by user.")
                break
                
            costs = []
            for individual in self.ga.population:
                cost = self.evaluate_controller(individual, max_force=max_force)
                costs.append(cost)
            
            # Find best of this generation 
            best_idx = np.argmin(costs)
            best_genes = self.ga.population[best_idx].copy()
            self.best_genes = best_genes
            best_cost = costs[best_idx]
            
            # Find WORST of this generation
            worst_idx = np.argmax(costs)
            worst_cost = costs[worst_idx]
            
            if self.worst_genes is None:
                self.worst_genes = self.ga.population[worst_idx].copy()
                self.worst_cost = worst_cost
            else:
                if worst_cost > self.worst_cost:
                    self.worst_genes = self.ga.population[worst_idx].copy()
                    self.worst_cost = worst_cost
            
            # Save best genes to file 
            try:
                with open("best_genes.txt", "w") as f:
                    json.dump(best_genes.tolist(), f)
            except Exception as e:
                print(f"Error saving best genes: {e}")
            
            target_size = pop_size
            
            # Evolve
            population, _ = self.ga.evolve(
                np.array(costs), 
                elitism_rate=elitism_rate,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                target_pop_size=target_size
            )
            
            print(f"Generation {gen+1}/{generations} - Best Cost: {best_cost:.2f}")
            
            if callback_epoch:
                callback_epoch(gen+1, best_cost, best_genes)

        # Save Population History as .npy
        try:
            np.save("results/population_history.npy", np.array(self.ga.population_history, dtype=object))
            print("Saved population history to results/population_history.npy")
        except Exception as e:
            print(f"Error saving npy history: {e}")

        # Save Population History as .csv
        try:
            import csv
            with open("results/population_history.csv", "w", newline='') as f:
                writer = csv.writer(f)
                # Header: Gen, Ind_ID, Cost, Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x
                writer.writerow(["Generation", "Individual_ID", "Cost", "Kp_th", "Ki_th", "Kd_th", "Kp_x", "Ki_x", "Kd_x"])
                
                for gen_idx, (population, costs) in enumerate(zip(self.ga.population_history, self.ga.cost_history)):
                    for ind_idx, (genes, cost) in enumerate(zip(population, costs)):
                        row = [gen_idx, ind_idx, cost] + genes.tolist()
                        writer.writerow(row)
            print("Saved population history to results/population_history.csv")
        except Exception as e:
            print(f"Error saving csv history: {e}")

        return self.ga.fitness_history, self.best_genes, self.ga.cost_history, self.worst_genes, self.ga.population_history
