import numpy as np

class DualPIDController:
    def __init__(self, Kp_th, Ki_th, Kd_th, Kp_x, Ki_x, Kd_x):
        #Inner Loop 
        self.Kp_th = Kp_th
        self.Ki_th = Ki_th
        self.Kd_th = Kd_th
        
        #Outer Loop
        self.Kp_x = Kp_x
        self.Ki_x = Ki_x
        self.Kd_x = Kd_x
        
        self.reset()

    def reset(self):
        self.integral_th = 0.0
        self.prev_error_th = 0.0
        
        self.integral_x = 0.0
        self.prev_error_x = 0.0

    def get_action(self, error_th, error_x, dt):
        # Theta
        self.integral_th += error_th * dt
        derivative_th = (error_th - self.prev_error_th) / dt if dt > 0 else 0.0
        
        u_th = (self.Kp_th * error_th) + (self.Ki_th * self.integral_th) + (self.Kd_th * derivative_th)
        self.prev_error_th = error_th
        
        # Position
        self.integral_x += error_x * dt
        derivative_x = (error_x - self.prev_error_x) / dt if dt > 0 else 0.0
        
        u_x = (self.Kp_x * error_x) + (self.Ki_x * self.integral_x) + (self.Kd_x * derivative_x)
        self.prev_error_x = error_x
        
        # Sum 
        u = u_th + u_x
        
        return u
