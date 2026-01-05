import mujoco
import numpy as np
import os

class Simulation:
    def __init__(self, model_path="cartpole.xml", render_mode=False, max_force=100.0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")    
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None
        self.max_force = max_force
        
        self.Q = 10.0
        self.R = 0.01
        self.fall_penalty = 5000.0

    def reset(self):
        """Resets the simulation to a near-upright position."""
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[:] = np.random.uniform(-0.05, 0.05, self.model.nq)
        self.data.qvel[:] = np.random.uniform(-0.05, 0.05, self.model.nv)
        
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def get_state(self):
        """Returns the current state [qpos, qvel]."""
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def step(self, action):
        """
        Applies action, steps simulation, calculates cost.
        Returns: next_state, cost, done
        """        
        action = np.clip(action, -self.max_force, self.max_force)
        self.data.ctrl[0] = action
        
        mujoco.mj_step(self.model, self.data)
        
        cart_pos = self.data.qpos[0]
        theta = self.data.qpos[1] # Angle deviation from upright (0 if upright in 'capsule' fromto 000 001? Wait.)
        
        x = self.data.qpos[0]
        theta = self.data.qpos[1]
        x_dot = self.data.qvel[0]
        theta_dot = self.data.qvel[1]
        
        w_theta = 1000.0
        w_x = 100.0
        w_u = 0.001
        
        fall_penalty = 1e6       

        state_cost = (w_theta * (theta**2)) + (w_x * (x**2))
        action_cost = w_u * (action**2)
        
        current_penalty = 0.0
        if abs(theta) > 0.5:
             current_penalty = fall_penalty
            
        total_cost = state_cost + action_cost + current_penalty
    
        return self.get_state(), total_cost, False
