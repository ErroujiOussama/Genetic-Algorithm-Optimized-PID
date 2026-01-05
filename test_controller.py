import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from environment import Simulation
from controller import DualPIDController

import json
import os

def load_best_genes():
    filename = "best_genes.txt"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                genes = json.load(f)
                print(f"Loaded Best Genes from {filename}: {genes}")
                return genes
        except Exception as e:
            print(f"Error loading genes: {e}")
    
    print("Optimization file not found or error. Using default fallback.")
    return [100.0, 1.0, 20.0, 10.0, 0.1, 5.0] # Fallback

BEST_GENES = load_best_genes()
# ==========================================

def test_realtime():
    print(f"Testing Controller with Gains: {BEST_GENES}")
    
    # Unpack Gains
    Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x = BEST_GENES
    
    # Initialize Controller and Simulation
    controller = DualPIDController(Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x)
    sim = Simulation(render_mode=False) # We allow viewer to be launched manually
    
    # Setup Real-time Plotting
    plt.ion() # Interactive Mode On
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot Lines
    line_x, = ax1.plot([], [], 'b-', linewidth=2, label="Cart Pos (m)")
    line_th, = ax2.plot([], [], 'r-', linewidth=2, label="Pole Angle (rad)")
    
    # Formatting
    ax1.set_title("Real-Time System Response")
    ax1.set_ylabel("Position (m)")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (rad)")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    
    # Data Buffers
    times = []
    xs = []
    ths = []
    
    # Simulation Loop
    start_time = time.time()
    dt = 0.01 # Simulation timestep
    
    print("Launching MuJoCo Viewer...")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        state = sim.reset()
        controller.reset()
        
        while viewer.is_running():
            step_start = time.time()
            current_time = step_start - start_time
            
            # --- Control Step ---
            # State: [x, theta, x_dot, theta_dot]
            x = state[0]
            theta = state[1]
            
            # Target is 0.0 for both
            error_x = 0.0 - x
            error_th = 0.0 - theta
            
            # Compute Action
            action = controller.get_action(error_th, error_x, dt)
            
            # Step Physics
            state, _, _ = sim.step(action)
            
            # --- Sync Viewer ---
            viewer.sync()
            
            # --- Update Plot ---
            times.append(current_time)
            xs.append(x)
            ths.append(theta)
            
            # Update plot every 5 steps to maintain performance
            if len(times) % 5 == 0:
                # Update Data
                line_x.set_data(times, xs)
                line_th.set_data(times, ths)
                
                # Adjust Limits (Dynamic Window)
                window_size = 5.0 # View last 5 seconds
                local_min_t = max(0, current_time - window_size)
                
                ax1.set_xlim(local_min_t, current_time + 0.5)
                ax2.set_xlim(local_min_t, current_time + 0.5)
                
                # Auto-scale Y with some padding
                if len(xs) > 0:
                    y_min, y_max = min(xs), max(xs)
                    margin = 0.5
                    if abs(y_max - y_min) < 0.1: margin = 1.0 # Default if flat
                    ax1.set_ylim(y_min - margin, y_max + margin)
                
                if len(ths) > 0:
                    y_min, y_max = min(ths), max(ths)
                    margin = 0.5
                    if abs(y_max - y_min) < 0.1: margin = 1.0
                    ax2.set_ylim(y_min - margin, y_max + margin)
                
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            # --- Timestep Sync ---
            time_until_next = sim.model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
                
    print("Simulation Finished.")
    plt.ioff()
    plt.show() 

if __name__ == "__main__":
    test_realtime()
