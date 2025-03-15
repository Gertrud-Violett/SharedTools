#====================================================
#AI Generated PID Control Program 2025 MIT License makkiblog.com
#PID Controller Simulator Module

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        
    def compute(self, process_variable, dt):
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        # PID output
        output = p_term + i_term + d_term
        
        return output

class SecondOrderSystem:
    def __init__(self, gain, time_constant, damping_ratio):
        self.gain = gain
        self.time_constant = time_constant
        self.damping_ratio = damping_ratio
        
    def dynamics(self, y, t, u):
        # y[0] is the output
        # y[1] is the first derivative of output
        
        # Convert the second-order differential equation to two first-order equations
        dydt = np.zeros(2)
        
        # dy[0]/dt = y[1]
        dydt[0] = y[1]
        
        # dy[1]/dt = (-2*zeta*omega*y[1] - omega^2*y[0] + K*omega^2*u) / 1
        omega = 1 / self.time_constant
        dydt[1] = (self.gain * omega**2 * u - 2 * self.damping_ratio * omega * y[1] - omega**2 * y[0])
        
        return dydt

def run_simulation(input_csv, system_params, pid_params, dt=0.1, setpoint_col=None, output_csv=None):
    # Load input signal
    data = pd.read_csv(input_csv)
    
    # Extract time and input values
    if 'time' in data.columns:
        time = data['time'].values
    else:
        # If time column not provided, create it
        time = np.arange(0, len(data)) * dt
    
    # Get input column
    if 'input' in data.columns:
        input_signal = data['input'].values
    else:
        # Assume the first column (besides time) is the input
        input_col = [col for col in data.columns if col != 'time'][0]
        input_signal = data[input_col].values
    
    # Get setpoint column if provided, otherwise use input as setpoint
    if setpoint_col and setpoint_col in data.columns:
        setpoint_signal = data[setpoint_col].values
    else:
        setpoint_signal = input_signal
    
    # Initialize system and controller
    system = SecondOrderSystem(
        gain=system_params['gain'],
        time_constant=system_params['time_constant'],
        damping_ratio=system_params['damping_ratio']
    )
    
    pid = PIDController(
        kp=pid_params['kp'],
        ki=pid_params['ki'],
        kd=pid_params['kd']
    )
    
    # Initialize arrays for storing results
    n_steps = len(time)
    system_output = np.zeros(n_steps)
    control_action = np.zeros(n_steps)
    
    # Initial conditions
    y0 = [0, 0]  # [position, velocity]
    
    # Simulation loop
    for i in range(1, n_steps):
        dt_actual = time[i] - time[i-1]
        
        # Set the PID setpoint
        pid.setpoint = setpoint_signal[i]
        
        # Get current system output
        system_output[i-1] = y0[0]
        
        # Compute control action
        control_action[i-1] = pid.compute(system_output[i-1], dt_actual)
        
        # Apply control action and input to the system
        actuator_input = input_signal[i-1] + control_action[i-1]
        
        # Simulate system for this time step
        t_span = [time[i-1], time[i]]
        sol = odeint(system.dynamics, y0, t_span, args=(actuator_input,))
        
        # Update state for next iteration
        y0 = sol[-1]
    
    # Get final output
    system_output[-1] = y0[0]
    control_action[-1] = control_action[-2]  # Just copy the last control action
    
    # Create results dataframe
    results = pd.DataFrame({
        'time': time,
        'input': input_signal,
        'setpoint': setpoint_signal,
        'control_action': control_action,
        'system_output': system_output
    })
    
    # Save to CSV if a filename is provided
    if output_csv:
        results.to_csv(output_csv, index=False)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, setpoint_signal, 'r--', label='Setpoint')
    plt.plot(time, system_output, 'b-', label='System Output')
    plt.grid(True)
    plt.legend()
    plt.title('PID Controlled Second-Order System Response')
    plt.ylabel('Output')
    
    plt.subplot(2, 1, 2)
    plt.plot(time, input_signal, 'g-', label='Input Signal')
    plt.plot(time, control_action, 'm-', label='Control Action')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Control Signal')
    
    plt.tight_layout()
    plt.savefig('pid_simulation_results.png')
    plt.show()
    
    return results

class PIDSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PID Controller Simulator")
        self.root.geometry("600x550")
        
        # Set default values
        self.input_file = ""
        self.output_file = "pid_output.csv"
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # System Parameters Frame
        system_frame = ttk.LabelFrame(main_frame, text="System Parameters", padding="10")
        system_frame.pack(fill=tk.X, pady=10)
        
        # Gain
        ttk.Label(system_frame, text="Gain:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.gain_var = tk.DoubleVar(value=5.0)
        ttk.Entry(system_frame, textvariable=self.gain_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Time Constant
        ttk.Label(system_frame, text="Time Constant:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.time_constant_var = tk.DoubleVar(value=3.0)
        ttk.Entry(system_frame, textvariable=self.time_constant_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Damping Ratio
        ttk.Label(system_frame, text="Damping Ratio:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.damping_ratio_var = tk.DoubleVar(value=0.5)
        ttk.Entry(system_frame, textvariable=self.damping_ratio_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # PID Parameters Frame
        pid_frame = ttk.LabelFrame(main_frame, text="PID Parameters", padding="10")
        pid_frame.pack(fill=tk.X, pady=10)
        
        # Kp
        ttk.Label(pid_frame, text="Kp (Proportional):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.kp_var = tk.DoubleVar(value=0.5)
        ttk.Entry(pid_frame, textvariable=self.kp_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Ki
        ttk.Label(pid_frame, text="Ki (Integral):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ki_var = tk.DoubleVar(value=0.1)
        ttk.Entry(pid_frame, textvariable=self.ki_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Kd
        ttk.Label(pid_frame, text="Kd (Derivative):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.kd_var = tk.DoubleVar(value=0.2)
        ttk.Entry(pid_frame, textvariable=self.kd_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # Input/Output Frame
        io_frame = ttk.LabelFrame(main_frame, text="Input/Output Settings", padding="10")
        io_frame.pack(fill=tk.X, pady=10)
        
        # Input File
        ttk.Label(io_frame, text="Input CSV:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_file_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.input_file_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(io_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        # Output File
        ttk.Label(io_frame, text="Output CSV:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_file_var = tk.StringVar(value="pid_output.csv")
        ttk.Entry(io_frame, textvariable=self.output_file_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(io_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Setpoint Column
        ttk.Label(io_frame, text="Setpoint Column:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.setpoint_col_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.setpoint_col_var, width=30).grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(io_frame, text="(Optional)").grid(row=2, column=2, padx=5, pady=5)
        
        # Generate test input if needed checkbox
        self.generate_test_input_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(io_frame, text="Generate test input if no file is selected", 
                        variable=self.generate_test_input_var).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Simulation button
        ttk.Button(main_frame, text="Run Simulation", command=self.run_simulation, 
                   style="Accent.TButton").pack(pady=20)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Configure styles for buttons
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 11, "bold"))
        
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file_var.set(filename)
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_file_var.set(filename)
            
    def run_simulation(self):
        try:
            # Get parameters from GUI
            system_params = {
                'gain': self.gain_var.get(),
                'time_constant': self.time_constant_var.get(),
                'damping_ratio': self.damping_ratio_var.get()
            }
            
            pid_params = {
                'kp': self.kp_var.get(),
                'ki': self.ki_var.get(),
                'kd': self.kd_var.get()
            }
            
            input_csv = self.input_file_var.get()
            output_csv = self.output_file_var.get()
            setpoint_col = self.setpoint_col_var.get() if self.setpoint_col_var.get() else None
            
            # Update status
            self.status_var.set("Running simulation...")
            self.root.update()
            
            # Check if input file exists or if we should generate test data
            if not input_csv or not os.path.exists(input_csv):
                if self.generate_test_input_var.get():
                    # Create a simple step input signal
                    self.status_var.set("Input file not found. Creating a test signal...")
                    self.root.update()
                    
                    # Create time vector
                    t = np.arange(0, 100, 0.1)
                    
                    # Create input signal (multiple step changes)
                    input_signal = np.zeros_like(t)
                    input_signal[(t >= 10) & (t < 40)] = 1.0
                    input_signal[(t >= 40) & (t < 70)] = -1.0
                    input_signal[(t >= 70)] = 0.5
                    
                    # Create DataFrame and save to CSV
                    test_data = pd.DataFrame({
                        'time': t,
                        'input': input_signal
                    })
                    
                    test_input_csv = "test_input.csv"
                    test_data.to_csv(test_input_csv, index=False)
                    input_csv = test_input_csv
                    self.input_file_var.set(test_input_csv)
                else:
                    raise FileNotFoundError("Input file not found and test data generation is disabled.")
            
            # Run the simulation
            run_simulation(input_csv, system_params, pid_params, 
                          setpoint_col=setpoint_col, output_csv=output_csv)
            
            # Update status
            self.status_var.set("Simulation completed successfully!")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Simulation Error", str(e))

if __name__ == "__main__":
    # Create tkinter application
    root = tk.Tk()
    app = PIDSimulatorGUI(root)
    root.mainloop()