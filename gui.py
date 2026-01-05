import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from optimizer_app import OptimizerApp
import numpy as np
import os
class GA_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GA Inverted Pendulum Optimization")
        self.root.geometry("900x700") 
        
        self.optimizer = OptimizerApp()
        self.vis_queue = queue.Queue()
        self.vis_done_event = threading.Event()
        self.plot_buffer = [] # Buffer for aggregated plotse
        self.is_training = False
        
        # Plot Data
        self.generations = []
        self.costs = []
        
        self.setup_ui()
        self.check_visualization_queue()

    def setup_ui(self):
        # Top Frame
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side="top", fill="x", padx=10, pady=5)

        # Logo
        try:
            self.logo_img = tk.PhotoImage(file="logo_resized.png")
            logo_label = ttk.Label(top_frame, image=self.logo_img)
            logo_label.pack(side="left", padx=10)
        except Exception:
            pass # Ignore if logo missing

        
        # Parameters Frame (Expanded)
        param_frame = ttk.LabelFrame(top_frame, text="Training Parameters")
        param_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Row 0
        ttk.Label(param_frame, text="Pop Size:").grid(row=0, column=0, padx=5, pady=2)
        self.pop_size_var = tk.IntVar(value=50)
        ttk.Entry(param_frame, textvariable=self.pop_size_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Gens:").grid(row=0, column=2, padx=5, pady=2)
        self.generations_var = tk.IntVar(value=20)
        ttk.Entry(param_frame, textvariable=self.generations_var, width=8).grid(row=0, column=3, padx=5, pady=2)
        
        # Row 1 (Rates)
        ttk.Label(param_frame, text="Mut Rate:").grid(row=1, column=0, padx=5, pady=2)
        self.mutation_var = tk.DoubleVar(value=0.2)
        ttk.Entry(param_frame, textvariable=self.mutation_var, width=8).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="Elitism:").grid(row=1, column=2, padx=5, pady=2)
        self.elitism_var = tk.DoubleVar(value=0.1)
        ttk.Entry(param_frame, textvariable=self.elitism_var, width=8).grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(param_frame, text="Crossover:").grid(row=1, column=4, padx=5, pady=2)
        self.crossover_var = tk.DoubleVar(value=0.8)
        ttk.Entry(param_frame, textvariable=self.crossover_var, width=8).grid(row=1, column=5, padx=5, pady=2)
        
        # Row 2 (Force)
        ttk.Label(param_frame, text="Max Force:").grid(row=2, column=0, padx=5, pady=2)
        self.force_var = tk.DoubleVar(value=10.0)
        ttk.Entry(param_frame, textvariable=self.force_var, width=8).grid(row=2, column=1, padx=5, pady=2)
        
        # Row 3 (Large Init)
        ttk.Label(param_frame, text="Large Init?").grid(row=3, column=0, padx=5, pady=2)
        self.large_init_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, variable=self.large_init_var).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Init Size:").grid(row=3, column=2, padx=5, pady=2)
        self.large_init_size_var = tk.IntVar(value=100)
        ttk.Entry(param_frame, textvariable=self.large_init_size_var, width=8).grid(row=3, column=3, padx=5, pady=2)
        
        # Options Frame
        opt_frame = ttk.LabelFrame(top_frame, text="Options")
        opt_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        self.visualize_evolution_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Visualize Evolution", variable=self.visualize_evolution_var).pack(anchor="w", padx=5)

        self.show_responses_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Show Response Plots", variable=self.show_responses_var).pack(anchor="w", padx=5)

        self.save_results_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Save Results", variable=self.save_results_var).pack(anchor="w", padx=5)
        
        # Controls
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(side="left", fill="both", padx=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_training)
        self.start_btn.pack(side="top", pady=2, fill="x")
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side="top", pady=2, fill="x")
        
        self.manual_vis_btn = ttk.Button(control_frame, text="Vis. Best", command=self.visualize_best, state="disabled")
        self.manual_vis_btn.pack(side="top", pady=2, fill="x")
        
        self.manual_vis_btn_bad = ttk.Button(control_frame, text="Vis. Bad", command=self.visualize_bad, state="disabled")
        self.manual_vis_btn_bad.pack(side="top", pady=2, fill="x")

        # Plot Area (Middle)
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Cost Evolution")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Best Cost (Log Scale)")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, font=("Arial", 10)).pack(side="bottom", anchor="w", padx=10, pady=5)
        
        # Create results dir
        if not os.path.exists("results"):
            os.makedirs("results")

    def check_visualization_queue(self):
        try:
            # Queue item now assumed to be (gen, genes) or just genes?
            # Let's standardize on (gen, genes).
            
            item = self.vis_queue.get_nowait()
            
            # Unpack
            if isinstance(item, tuple):
                gen, genes = item
            else:
                gen, genes = 0, item # Fallback for older items, though (gen, genes) is preferred
            # Run visualization on main thread
            # unpack tuple
            result = self.optimizer.visualize_controller(genes, max_force=self.force_var.get(), duration=5.0)
            if isinstance(result, tuple) and len(result) == 2:
                states, actions = result
            else:
                states = result
                actions = np.zeros(len(states)) # Fallback
            
            if states is not None and len(states) > 0:
                if self.show_responses_var.get():
                    self.show_response_plots(states, actions)
                
                # Buffer for saving
                if self.save_results_var.get():
                    self.plot_buffer.append((gen, states, actions))
                    if len(self.plot_buffer) >= 10:
                        self.save_aggregated_plot()

            self.vis_done_event.set()
            self.status_var.set("Resuming Training...")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_visualization_queue)

    def show_response_plots(self, states, actions):
        """Opens a window to plot x, theta, and effort."""
        top = Toplevel(self.root)
        top.title("System Response")
        top.geometry("600x600")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        
        t = np.arange(len(states)) * 0.01 
        
        # Position
        ax1.plot(t, states[:, 0], label="Position (m)", color="blue")
        ax1.set_ylabel("Pos (m)")
        ax1.grid(True)
        ax1.legend(loc="upper right")
        
        # Angle
        ax2.plot(t, states[:, 1], label="Angle (rad)", color="orange")
        ax2.set_ylabel("Ang (rad)")
        ax2.grid(True)
        ax2.legend(loc="upper right")
        
        # Effort
        ax3.plot(t, actions, label="Effort (N)", color="green")
        ax3.set_ylabel("Force (N)")
        ax3.set_xlabel("Time (s)")
        ax3.grid(True)
        ax3.legend(loc="upper right")
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        def on_close():
            plt.close(fig)
            top.destroy()
        top.protocol("WM_DELETE_WINDOW", on_close)

    def save_aggregated_plot(self):
        """Saves last 10 generations: 5x2 grid + Summary row."""
        if not self.plot_buffer: return
        
        start_gen = self.plot_buffer[0][0]
        end_gen = self.plot_buffer[-1][0]
        
        # Increase figure size to accommodate more subplots
        fig = plt.figure(figsize=(20, 30)) 
        
        # Main Grid: 6 rows (5 for gens, 1 for summary), 2 columns
        # We need more height for the generation rows since they now hold 3 plots each
        gs = fig.add_gridspec(6, 2, height_ratios=[3, 3, 3, 3, 3, 2])
        
        # --- 1. Grid of 10 Individual Responses ---
        for i, (gen, states, actions) in enumerate(self.plot_buffer):
            row = i // 2
            col = i % 2
            
            if row < 5:
                # Create a sub-grid of 3 rows within this cell
                gs_sub = gs[row, col].subgridspec(3, 1, hspace=0.1)
                
                ax_x = fig.add_subplot(gs_sub[0])
                ax_th = fig.add_subplot(gs_sub[1], sharex=ax_x)
                ax_u = fig.add_subplot(gs_sub[2], sharex=ax_x)
                
                t = np.arange(len(states)) * 0.01
                
                # Position
                ax_x.plot(t, states[:, 0], color="blue", label="Pos")
                ax_x.set_ylabel("Pos (m)", fontsize=8)
                ax_x.grid(True, alpha=0.3)
                ax_x.tick_params(labelbottom=False) # Hide x labels for top plots
                ax_x.set_title(f"Gen {gen}", fontsize=10, pad=2)
                
                # Angle
                ax_th.plot(t, states[:, 1], color="orange", label="Ang")
                ax_th.set_ylabel("Ang (rad)", fontsize=8)
                ax_th.grid(True, alpha=0.3)
                ax_th.tick_params(labelbottom=False)
                
                # Effort
                ax_u.plot(t, actions, color="green", label="Force")
                ax_u.set_ylabel("Force (N)", fontsize=8)
                ax_u.grid(True, alpha=0.3)
                
                # Only show Legend on the very first cell to reduce clutter? 
                # Or small legend on all? separate plots mean separate contexts.
                # Let's put a small legend on each.
                # ax_x.legend(loc='upper right', fontsize='xx-small')

        # --- 2. Summary Row (Row 5) ---
        # Spans both columns? Or just separate?
        # Let's make the summary region span the whole bottom
        
        gs_summary = gs[5, :].subgridspec(1, 3)
        
        ax_sum_x = fig.add_subplot(gs_summary[0])
        ax_sum_th = fig.add_subplot(gs_summary[1])
        ax_sum_u = fig.add_subplot(gs_summary[2])
        
        cmap = plt.get_cmap("viridis")
        
        for i, (gen, states, actions) in enumerate(self.plot_buffer):
            t = np.arange(len(states)) * 0.01
            color = cmap(i / len(self.plot_buffer))
            alpha = 0.5 + (0.5 * (i / len(self.plot_buffer)))
            
            label = f"G{gen}" if (i==0 or i==9) else None
            
            ax_sum_x.plot(t, states[:, 0], color=color, alpha=alpha, label=label)
            ax_sum_th.plot(t, states[:, 1], color=color, alpha=alpha, label=label)
            ax_sum_u.plot(t, actions, color=color, alpha=alpha, label=label)
            
        ax_sum_x.set_title("Summary: Position")
        ax_sum_x.set_xlabel("Time (s)")
        ax_sum_x.grid(True)
        ax_sum_x.legend(fontsize='x-small')
        
        ax_sum_th.set_title("Summary: Angle")
        ax_sum_th.set_xlabel("Time (s)")
        ax_sum_th.grid(True)
        
        ax_sum_u.set_title("Summary: Effort")
        ax_sum_u.set_xlabel("Time (s)")
        ax_sum_u.grid(True)

        fig.suptitle(f"Analysis Gen {start_gen}-{end_gen}", fontsize=16)
        # plt.tight_layout() # distinct subgridspecs often fight with tight_layout
        
        filename = f"results/analysis_gens_{start_gen}_{end_gen}.png"
        fig.savefig(filename, bbox_inches='tight', dpi=1000) # Use bbox_inches to handle layout
        plt.close(fig)
        
        # Clear Buffer
        self.plot_buffer = []

    def start_training(self):
        if self.is_training: return
        
        try:
            pop_size = self.pop_size_var.get()
            gens = self.generations_var.get()
            mut_rate = self.mutation_var.get()
            elite_rate = self.elitism_var.get()
            cross_rate = self.crossover_var.get()
            max_force = self.force_var.get()
            lg_enabled = self.large_init_var.get()
            lg_size = self.large_init_size_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters.")
            return
            
        # Reset cost history for heatmap
        self.optimizer.cost_history_for_gui = []

        # Reset Plot
        self.generations = []
        self.costs = []
        self.ax.clear()
        self.ax.set_title("Cost Evolution")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Best Cost (Log Scale)")
        self.ax.grid(True, which="both", ls="-")
        self.canvas.draw()

        self.is_training = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.manual_vis_btn.config(state="disabled")
        self.manual_vis_btn_bad.config(state="disabled")
        
        self.training_thread = threading.Thread(
            target=self.run_optimizer, 
            args=(pop_size, gens, mut_rate, elite_rate, cross_rate, max_force, lg_enabled, lg_size)
        )
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_training(self):
        if self.is_training:
            self.optimizer.stop_event.set()

    def run_optimizer(self, pop_size, gens, mut_rate, elite_rate, cross_rate, max_force, lg_enabled, lg_size):
        def epoch_callback(gen, cost, best_genes):
            # Update GUI Data
            self.root.after(0, lambda: self.update_plot(gen, cost))
            
            # Visualization
            if self.visualize_evolution_var.get():
                self.vis_done_event.clear()
                # Pass Gen and Genes
                self.vis_queue.put((gen, best_genes))
                self.vis_done_event.wait()

        history, best_genes, cost_history, worst_genes, pop_history = self.optimizer.train(
            pop_size=pop_size, 
            generations=gens, 
            mutation_rate=mut_rate,
            elitism_rate=elite_rate,
            crossover_rate=cross_rate,
            max_force=max_force,
            large_init_enabled=lg_enabled,
            large_init_size=lg_size,
            callback_epoch=epoch_callback
        )
        
        # Store results for main thread plotting
        self.last_cost_history = cost_history
        self.last_pop_history = pop_history
        
        self.is_training = False
        self.root.after(0, self.training_finished)

    def update_plot(self, gen, cost):
        self.generations.append(gen)
        self.costs.append(cost)
        self.status_var.set(f"Training: Gen {gen} | Cost: {cost:.2f}")
        
        self.ax.clear()
        self.ax.set_title("Cost Evolution")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Best Cost (Log Scale)")
        self.ax.grid(True, which="both", ls="-")
        self.ax.semilogy(self.generations, self.costs, marker='o', linestyle='-')
        self.canvas.draw()
        
    def save_heatmap_plot(self, cost_history):
        """Generates and saves the Evolution Heatmap."""
        if not cost_history: return
        
        try:
            min_size = min(len(c) for c in cost_history)
            num_gens = len(cost_history)
            
            matrix = np.zeros((min_size, num_gens))
            
            for g, costs in enumerate(cost_history):
                # Sort individual costs for this generation
                sorted_costs = np.sort(costs)
                matrix[:, g] = sorted_costs[:min_size]
                
            # Apply Log Scale
            matrix = np.log10(matrix + 1e-6)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(matrix, aspect='auto', cmap='jet', interpolation='nearest', origin='upper')
            
            ax.set_title("Population Cost Distribution (Log Scale)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Individual Rank (0=Best)")
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Log10(Cost)")
            
            filename = "results/heatmap.png"
            fig.savefig(filename, dpi=300)
            plt.close(fig)
            print(f"Heatmap saved to {filename}")
        except Exception as e:
            print(f"Error saving heatmap: {e}")

    def save_3d_convergence_plot(self, pop_history):
        """Generates 3D scatter plots of PID gains over generations."""
        if not pop_history: return

        try:
            # Flatten data for plotting
            # We want X, Y, Z coordinates and Colormap based on Generation
            
            # Plot 1: Angle Gains (0, 1, 2)
            kp_th, ki_th, kd_th = [], [], []
            colors_th = []
            
            # Plot 2: Position Gains (3, 4, 5)
            kp_x, ki_x, kd_x = [], [], []
            colors_x = []
            
            num_gens = len(pop_history)
            
            for g, population in enumerate(pop_history):
                # population is (PopSize, 6)
                # Normalize generation for color (0.0 to 1.0)
                c_val = g / max(1, num_gens - 1)
                
                for ind in population:
                    # Angle
                    kp_th.append(ind[0])
                    ki_th.append(ind[1])
                    kd_th.append(ind[2])
                    colors_th.append(c_val)
                    
                    # Position
                    kp_x.append(ind[3])
                    ki_x.append(ind[4])
                    kd_x.append(ind[5])
                    colors_x.append(c_val)
            
            # Create Plot: Theta
            fig1 = plt.figure(figsize=(10, 8))
            ax1 = fig1.add_subplot(111, projection='3d')
            p1 = ax1.scatter3D(kp_th, ki_th, kd_th, c=colors_th, cmap='jet', alpha=0.6, s=10)
            ax1.set_xlabel('Kp_theta')
            ax1.set_ylabel('Ki_theta')
            ax1.set_zlabel('Kd_theta')
            ax1.set_title('Convergence of Angle PID Gains')
            cb1 = fig1.colorbar(p1, ax=ax1, shrink=0.5)
            cb1.set_label('Generation (Blue->Red)')
            fig1.savefig("results/convergence_theta.png", dpi=300)
            plt.close(fig1)
            
            # Create Plot: Position
            fig2 = plt.figure(figsize=(10, 8))
            ax2 = fig2.add_subplot(111, projection='3d')
            p2 = ax2.scatter3D(kp_x, ki_x, kd_x, c=colors_x, cmap='jet', alpha=0.6, s=10)
            ax2.set_xlabel('Kp_x')
            ax2.set_ylabel('Ki_x')
            ax2.set_zlabel('Kd_x')
            ax2.set_title('Convergence of Position PID Gains')
            cb2 = fig2.colorbar(p2, ax=ax2, shrink=0.5)
            cb2.set_label('Generation (Blue->Red)')
            fig2.savefig("results/convergence_x.png", dpi=300)
            plt.close(fig2)
            
            print("3D Convergence plots saved.")
            
                
        except Exception as e:
            print(f"Error saving 3D plots: {e}")

    def save_min_max_plot(self, cost_history):
        """Generates a plot of Min and Max costs per generation (Log Scale)."""
        if not cost_history: return
        
        try:
            min_costs = []
            max_costs = []
            gens = []
            
            for g, costs in enumerate(cost_history):
                # Filter out crazy values if any? No, show raw.
                # costs is an array
                min_c = np.min(costs)
                max_c = np.max(costs)
                
                min_costs.append(min_c)
                max_costs.append(max_c)
                gens.append(g)
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.semilogy(gens, min_costs, label="Min Cost", marker='o', color='blue')
            ax.semilogy(gens, max_costs, label="Max Cost", marker='x', color='red')
            
            ax.set_title("Cost Range per Generation (Log Scale)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Cost")
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.legend()
            
            filename = "results/min_max_cost.png"
            fig.savefig(filename, dpi=300)
            plt.close(fig)
            print(f"Min/Max plot saved to {filename}")
            
        except Exception as e:
            print(f"Error saving Min/Max plot: {e}")

    def training_finished(self):
        # Generate Plots on Main Thread
        if hasattr(self, 'last_cost_history'):
            self.save_heatmap_plot(self.last_cost_history)
        
        if hasattr(self, 'last_pop_history'):
            self.save_3d_convergence_plot(self.last_pop_history)
            
        if hasattr(self, 'last_cost_history'):
            self.save_min_max_plot(self.last_cost_history)

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.manual_vis_btn.config(state="normal")
        self.manual_vis_btn_bad.config(state="normal")
        self.status_var.set("Training Complete")
        messagebox.showinfo("Done", "Training Completed! Plots Saved.")

    def visualize_best(self):
        if self.optimizer.best_genes is not None:
            self.status_var.set("Visualizing Best...")
            states, actions = self.optimizer.visualize_controller(self.optimizer.best_genes)
            if self.show_responses_var.get() and states is not None:
                self.show_response_plots(states, actions)
            self.status_var.set("Ready")
        else:
            messagebox.showwarning("Warning", "No model trained yet.")

    def visualize_bad(self):
        if self.optimizer.worst_genes is not None:
            self.status_var.set("Visualizing Worst...")
            states, actions = self.optimizer.visualize_controller(self.optimizer.worst_genes)
            if self.show_responses_var.get() and states is not None:
                self.show_response_plots(states, actions)
            self.status_var.set("Ready")
        else:
            messagebox.showwarning("Warning", "No bad model record.")

if __name__ == "__main__":
    root = tk.Tk()
    app = GA_GUI(root)
    root.mainloop()
