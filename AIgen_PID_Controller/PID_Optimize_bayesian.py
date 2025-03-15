#====================================================
#AI Generated PID Control Program 2025 MIT License makkiblog.com
#PID Parameter Optimization Module

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import time
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
from scipy.stats import norm

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        
    def reset(self):
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
        
        return output, error

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

def run_simulation(input_signal, time_vector, setpoint_signal, system_params, pid_params):
    # Initialize system and controller
    system = SecondOrderSystem(
        gain=system_params['gain'],
        time_constant=system_params['time_constant'],
        damping_ratio=system_params['damping_ratio']
    )
    
    pid = PIDController(
        kp=pid_params[0],  # Kp
        ki=pid_params[1],  # Ki
        kd=pid_params[2]   # Kd
    )
    
    # Initialize arrays for storing results
    n_steps = len(time_vector)
    system_output = np.zeros(n_steps)
    control_action = np.zeros(n_steps)
    error_values = np.zeros(n_steps)
    
    # Initial conditions
    y0 = [0, 0]  # [position, velocity]
    
    # Simulation loop
    for i in range(1, n_steps):
        dt_actual = time_vector[i] - time_vector[i-1]
        
        # Set the PID setpoint
        pid.setpoint = setpoint_signal[i]
        
        # Get current system output
        system_output[i-1] = y0[0]
        
        # Compute control action
        control_action[i-1], error_values[i-1] = pid.compute(system_output[i-1], dt_actual)
        
        # Apply control action and input to the system
        actuator_input = input_signal[i-1] + control_action[i-1]
        
        # Simulate system for this time step
        t_span = [time_vector[i-1], time_vector[i]]
        sol = odeint(system.dynamics, y0, t_span, args=(actuator_input,))
        
        # Update state for next iteration
        y0 = sol[-1]
    
    # Get final output
    system_output[-1] = y0[0]
    control_action[-1] = control_action[-2]
    error_values[-1] = setpoint_signal[-1] - system_output[-1]
    
    return system_output, control_action, error_values

def calculate_performance_metrics(time_vector, setpoint_signal, output_signal):
    # Find step changes in setpoint
    step_indices = np.where(np.abs(np.diff(setpoint_signal)) > 0.1)[0] + 1
    
    if len(step_indices) == 0:
        # No step changes found
        return None
    
    metrics_list = []
    
    for step_idx in step_indices:
        if step_idx + 100 >= len(time_vector):
            continue  # Skip if not enough data after step
            
        step_time = time_vector[step_idx]
        step_size = setpoint_signal[step_idx] - setpoint_signal[step_idx-1]
        
        if abs(step_size) < 0.1:
            continue  # Skip small steps
            
        # Get relevant slice of signals
        end_idx = min(step_idx + 500, len(time_vector))
        t_slice = time_vector[step_idx:end_idx] - step_time
        y_slice = output_signal[step_idx:end_idx]
        sp_slice = setpoint_signal[step_idx:end_idx]
        
        # Normalize output for consistent analysis
        y_norm = (y_slice - output_signal[step_idx-1]) / step_size
        sp_norm = (sp_slice - setpoint_signal[step_idx-1]) / step_size
        
        # Calculate metrics
        target = 1.0  # Normalized target
        
        # Rise time (time to reach 90% of setpoint)
        try:
            rise_indices = np.where(y_norm >= 0.9 * target)[0]
            rise_time = t_slice[rise_indices[0]] if len(rise_indices) > 0 else t_slice[-1]
        except:
            rise_time = t_slice[-1]
        
        # Settling time (time to stay within 2% of final value)
        try:
            settled = np.where(np.abs(y_norm - target) <= 0.02)[0]
            consecutive_settled = np.split(settled, np.where(np.diff(settled) != 1)[0] + 1)
            settling_time = t_slice[-1]  # Default if never settles
            for segment in consecutive_settled:
                if len(segment) > 10:  # Must be settled for at least 10 samples
                    settling_time = t_slice[segment[0]]
                    break
        except:
            settling_time = t_slice[-1]
        
        # Overshoot
        max_value = np.max(y_norm)
        overshoot = max(0, (max_value - target) * 100)
        
        # Steady-state error
        if len(t_slice) > 50:
            steady_state_error = abs(np.mean(y_norm[-20:]) - target) * 100
        else:
            steady_state_error = abs(y_norm[-1] - target) * 100
        
        metrics = {
            'rise_time': rise_time,
            'settling_time': settling_time,
            'overshoot': overshoot,
            'steady_state_error': steady_state_error
        }
        
        metrics_list.append(metrics)
    
    # Average metrics across all steps
    if not metrics_list:
        return None
        
    avg_metrics = {
        'rise_time': np.mean([m['rise_time'] for m in metrics_list]),
        'settling_time': np.mean([m['settling_time'] for m in metrics_list]),
        'overshoot': np.mean([m['overshoot'] for m in metrics_list]),
        'steady_state_error': np.mean([m['steady_state_error'] for m in metrics_list])
    }
    
    return avg_metrics

def objective_function(pid_params, time_vector, input_signal, setpoint_signal, system_params, weights):
    # Run simulation with these PID parameters
    output, control, errors = run_simulation(
        input_signal, time_vector, setpoint_signal, system_params, pid_params
    )
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(time_vector, setpoint_signal, output)
    
    if metrics is None:
        return float('inf')
    
    # Calculate objective value (lower is better)
    obj_value = (
        weights['rise_time'] * metrics['rise_time'] +
        weights['settling_time'] * metrics['settling_time'] +
        weights['overshoot'] * metrics['overshoot'] +
        weights['steady_state_error'] * metrics['steady_state_error'] +
        weights['control_effort'] * np.mean(np.abs(control))
    )
    
    # Add penalty for negative PID gains
    if any(p < 0 for p in pid_params):
        obj_value += 1000
    
    return obj_value

# Bayesian definition
def bayesian_optimization(objective_function, bounds, n_iterations=25, n_points=500, initial_points=5, xi=0.01):
    """
    Bayesian optimization for finding the optimal PID parameters.
    
    Parameters:
    -----------
    objective_function : callable
        Function to minimize
    bounds : list of tuples
        Bounds for each parameter (lower, upper)
    n_iterations : int
        Number of iterations to run optimization
    n_points : int
        Number of points to sample when maximizing acquisition function
    initial_points : int
        Number of random points to sample before starting optimization
    xi : float
        Exploration-exploitation trade-off parameter
    
    Returns:
    --------
    tuple
        Best parameters found and their score
    """
    start_time = time.time()
    
    # Dimensions
    dim = len(bounds)
    
    # Initialize storage for samples
    samples = []
    sample_values = []
    
    # Generate initial random samples
    print("Generating initial random samples...")
    for _ in range(initial_points):
        # Random point within bounds
        point = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)])
        samples.append(point)
        # Evaluate objective function
        value = objective_function(point)
        sample_values.append(value)
    
    # Convert to arrays
    samples = np.array(samples)
    sample_values = np.array(sample_values)
    
    # Find current best
    best_idx = np.argmin(sample_values)
    best_params = samples[best_idx]
    best_score = sample_values[best_idx]
    
    # Setup Gaussian Process with Matern kernel
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
    
    print(f"Starting Bayesian optimization for {n_iterations} iterations...")
    progress_bar = tqdm(total=n_iterations, desc="Bayesian Optimization")
    
    # Perform optimization iterations
    for i in range(n_iterations):
        iteration_start = time.time()
        
        # Fit GP with available data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(samples, sample_values)
        
        # Generate random points to evaluate acquisition function
        x_tries = np.array([np.random.uniform(bounds[j][0], bounds[j][1], size=n_points) for j in range(dim)]).T
        
        # Calculate mean and std at test points
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = gp.predict(x_tries, return_std=True)
        
        # Find current best
        current_best = np.min(sample_values)
        
        # Expected improvement
        with np.errstate(divide='ignore'):
            imp = current_best - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        # Find point with highest EI
        best_acq_idx = np.argmax(ei)
        next_sample = x_tries[best_acq_idx]
        
        # Evaluate objective function at next point
        next_value = objective_function(next_sample)
        
        # Add to samples
        samples = np.vstack((samples, next_sample))
        sample_values = np.append(sample_values, next_value)
        
        # Update best parameters if improved
        if next_value < best_score:
            best_score = next_value
            best_params = next_sample
            iteration_time = time.time() - iteration_start
            print(f"\nNew best: Kp={best_params[0]:.4f}, Ki={best_params[1]:.4f}, Kd={best_params[2]:.4f}, Score={best_score:.4f}, Iteration time: {iteration_time:.2f}s")
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    total_time = time.time() - start_time
    print(f"Bayesian optimization completed in {total_time:.2f} seconds")
    
    return best_params, best_score



def optimize_pid(input_csv, system_params, setpoint_col=None, output_csv=None,
                 initial_pid=None, optimize_method='grid_search', weights=None):
    """
    Optimize PID parameters for a given system and input signal.
    
    Parameters:
    -----------
    input_csv : str
        Path to CSV file containing input signal
    system_params : dict
        Dictionary with 'gain', 'time_constant', and 'damping_ratio'
    setpoint_col : str, optional
        Name of setpoint column in CSV. If None, input is used as setpoint.
    output_csv : str, optional
        Path to save output CSV. If None, no CSV is saved.
    initial_pid : list, optional
        Initial [kp, ki, kd] values for optimization
    optimize_method : str
        'grid_search', 'minimize', or 'bayesian'
    weights : dict, optional
        Weights for different performance metrics in objective function
        
    Returns:
    --------
    dict
        Optimized PID parameters and performance metrics
    """
    # Track total compute time
    start_time = time.time()
    
    # Default weights
    if weights is None:
        weights = {
            'rise_time': 1.0,
            'settling_time': 2.0,
            'overshoot': 3.0,
            'steady_state_error': 5.0,
            'control_effort': 0.1
        }
    
    # Default initial PID parameters
    if initial_pid is None:
        initial_pid = [1.0, 0.1, 0.1]
    
    # Load input signal
    data = pd.read_csv(input_csv)
    
    # Extract time and input values
    if 'time' in data.columns:
        time_vector = data['time'].values
    else:
        # If time column not provided, create it
        dt = 0.1
        time_vector = np.arange(0, len(data)) * dt
    
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
    
    # Optimize PID parameters
    best_pid = None
    best_metrics = None
    best_score = float('inf')
    
    optimization_start_time = time.time()  # Track optimization-specific time
    
    if optimize_method == 'grid_search':
        # Grid search over parameter space
        print("Starting grid search optimization...")
        
        # Define search space
        kp_values = np.linspace(0.1, 5.0, 10)
        ki_values = np.linspace(0.0, 1.0, 10)
        kd_values = np.linspace(0.0, 2.0, 10)
        
        total_combinations = len(kp_values) * len(ki_values) * len(kd_values)
        progress_bar = tqdm(total=total_combinations, desc="Optimizing PID")
        
        for kp in kp_values:
            for ki in ki_values:
                for kd in kd_values:
                    pid_params = [kp, ki, kd]
                    
                    # Calculate objective value
                    score = objective_function(
                        pid_params, time_vector, input_signal, setpoint_signal, system_params, weights
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_pid = pid_params
                        
                        # Run simulation with best parameters so far
                        output, control, errors = run_simulation(
                            input_signal, time_vector, setpoint_signal, system_params, best_pid
                        )
                        
                        # Calculate performance metrics
                        best_metrics = calculate_performance_metrics(
                            time_vector, setpoint_signal, output
                        )
                    
                    progress_bar.update(1)
        
        progress_bar.close()
        
    elif optimize_method == 'minimize':  # Use scipy.optimize.minimize
        print("Starting numerical optimization...")
        
        # Define bounds for parameters
        bounds = [(0.01, 10.0), (0.0, 5.0), (0.0, 5.0)]  # (kp, ki, kd)
        
        # Run optimization
        result = minimize(
            lambda params: objective_function(
                params, time_vector, input_signal, setpoint_signal, system_params, weights
            ),
            initial_pid,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )
        
        best_pid = result.x.tolist()
        
        # Run simulation with optimized parameters
        output, control, errors = run_simulation(
            input_signal, time_vector, setpoint_signal, system_params, best_pid
        )
        
        # Calculate performance metrics
        best_metrics = calculate_performance_metrics(
            time_vector, setpoint_signal, output
        )
        best_score = result.fun
        
    elif optimize_method == 'bayesian':  # Use Bayesian optimization
        print("Starting Bayesian optimization...")
        
        # Define bounds for parameters
        bounds = [(0.01, 10.0), (0.0, 5.0), (0.0, 5.0)]  # (kp, ki, kd)
        
        # Create objective function wrapper
        def objective_wrapper(params):
            return objective_function(
                params, time_vector, input_signal, setpoint_signal, system_params, weights
            )
        
        # Run Bayesian optimization
        best_pid, best_score = bayesian_optimization(
            objective_wrapper, 
            bounds,
            n_iterations=25,  # Number of iterations
            initial_points=5   # Number of initial random points
        )
        
        # Run simulation with optimized parameters
        output, control, errors = run_simulation(
            input_signal, time_vector, setpoint_signal, system_params, best_pid
        )
        
        # Calculate performance metrics
        best_metrics = calculate_performance_metrics(
            time_vector, setpoint_signal, output
        )
    
    # Calculate optimization time
    optimization_time = time.time() - optimization_start_time
    print(f"\nOptimization time: {optimization_time:.2f} seconds")
    
    # Run final simulation with optimized parameters
    print(f"\nOptimized PID parameters: Kp={best_pid[0]:.4f}, Ki={best_pid[1]:.4f}, Kd={best_pid[2]:.4f}")
    
    output, control, errors = run_simulation(
        input_signal, time_vector, setpoint_signal, system_params, best_pid
    )
    
    # Create results dataframe
    results = pd.DataFrame({
        'time': time_vector,
        'input': input_signal,
        'setpoint': setpoint_signal,
        'control_action': control,
        'system_output': output,
        'error': errors
    })
    
    # Save to CSV if a filename is provided
    if output_csv:
        results.to_csv(output_csv, index=False)
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_vector, setpoint_signal, 'r--', label='Setpoint')
    plt.plot(time_vector, output, 'b-', label='System Output')
    plt.grid(True)
    plt.legend()
    plt.title(f'Optimized PID Controller (Kp={best_pid[0]:.4f}, Ki={best_pid[1]:.4f}, Kd={best_pid[2]:.4f})')
    plt.ylabel('Output')
    
    plt.subplot(3, 1, 2)
    plt.plot(time_vector, input_signal, 'g-', label='Input Signal')
    plt.plot(time_vector, control, 'm-', label='Control Action')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Control Signal')
    
    plt.subplot(3, 1, 3)
    plt.plot(time_vector, errors, 'r-', label='Error')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    
    plt.tight_layout()
    plt.savefig('optimized_pid_results.png')
    plt.show()
    
    # Calculate total execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # Return optimized parameters and performance metrics
    return {
        'pid_params': {
            'kp': best_pid[0],
            'ki': best_pid[1],
            'kd': best_pid[2]
        },
        'performance': best_metrics,
        'objective_score': best_score,
        'optimization_time': optimization_time,
        'total_time': total_time
    }


def ziegler_nichols_tuning(system_params):
    """
    Calculate PID parameters using Ziegler-Nichols method.
    """
    # Convert system parameters to ultimate gain and period
    gain = system_params['gain']
    time_constant = system_params['time_constant']
    damping_ratio = system_params['damping_ratio']
    
    # For second-order system, calculate ultimate gain and period
    # These are approximate for a second-order system
    omega_n = 1 / time_constant
    
    # Calculate ultimate gain and period
    K_u = 1 / gain  # Approximate
    T_u = 2 * np.pi / omega_n
    
    # Calculate PID parameters using Ziegler-Nichols rules
    kp = 0.6 * K_u
    ki = 1.2 * K_u / T_u
    kd = 0.075 * K_u * T_u
    
    return {
        'kp': kp,
        'ki': ki,
        'kd': kd
    }

def cohen_coon_tuning(system_params):
    """
    Calculate PID parameters using Cohen-Coon method.
    This is an approximation for second-order systems.
    """
    # Convert system parameters
    K = system_params['gain']
    tau = system_params['time_constant']
    zeta = system_params['damping_ratio']
    
    # For second-order system, approximate first-order plus dead time
    theta = tau * (1 - zeta)  # Approximate dead time
    tau1 = tau  # Approximate time constant
    
    # Calculate ratios
    r = theta / tau1
    
    # Calculate PID parameters using Cohen-Coon rules
    kp = (1 / K) * (1.33 + (0.33 * r)) * (1 / (1 + r))
    ki = (1 / K) * (0.66 + (0.33 * r)) * (1 / (theta * (1 + r)))
    kd = (1 / K) * (0.33 * r) * (1 / (1 + r)) * theta
    
    return {
        'kp': kp,
        'ki': ki,
        'kd': kd
    }

# Example usage
if __name__ == "__main__":
    optimize_method = 'bayesian'  # 'grid_search', 'minimize', or 'bayesian'
    # Define system parameters
    system_params = {
        'gain': 5.0,         # System gain: def = 5.0
        'time_constant': 3.0,  # Time constant: def = 3.0
        'damping_ratio': 0.5   # Damping ratio: def = 0.5
    }
    
    # If no input CSV is available, create a simple step input
    try:
        # Try to load external CSV
        input_csv = "input_signal.csv"
        pd.read_csv(input_csv)  # Just to check if it exists
    except FileNotFoundError:
        # Create a simple step input signal
        print("Input file not found. Creating a test signal...")
        
        # Create time vector
        t = np.arange(0, 150, 0.1)
        
        # Create input signal (multiple step changes)
        input_signal = np.zeros_like(t)
        input_signal[(t >= 10) & (t < 40)] = 1.0
        input_signal[(t >= 40) & (t < 70)] = -1.0
        input_signal[(t >= 70) & (t < 100)] = 0.5
        input_signal[(t >= 100)] = 0.3
        
        # Create DataFrame and save to CSV
        test_data = pd.DataFrame({
            'time': t,
            'input': input_signal
        })
        test_data.to_csv("test_input.csv", index=False)
        input_csv = "test_input.csv"
    
    print("System Parameters:")
    print(f"Gain: {system_params['gain']}")
    print(f"Time Constant: {system_params['time_constant']}")
    print(f"Damping Ratio: {system_params['damping_ratio']}")
    print("\n")
    
    # Calculate PID parameters using classical methods
    zn_params = ziegler_nichols_tuning(system_params)
    cc_params = cohen_coon_tuning(system_params)
    
    print("Estimated PID parameters from classical methods:")
    print("Ziegler-Nichols method:")
    print(f"Kp: {zn_params['kp']:.4f}, Ki: {zn_params['ki']:.4f}, Kd: {zn_params['kd']:.4f}")
    print("\nCohen-Coon method:")
    print(f"Kp: {cc_params['kp']:.4f}, Ki: {cc_params['ki']:.4f}, Kd: {cc_params['kd']:.4f}")
    print("\n")
    
    # Define weights for optimization
    weights = {
        'rise_time': 1.0,
        'settling_time': 1.5,
        'overshoot': 3.0,
        'steady_state_error': 5.0,
        'control_effort': 0.1
    }
    
    # Choose optimization method
    #optimize_method = 'grid_search'  # 'grid_search' or 'minimize'
    
    # Use Ziegler-Nichols as initial guess
    initial_pid = [zn_params['kp'], zn_params['ki'], zn_params['kd']]
    
    print(f"Starting PID optimization using {optimize_method}...")
    print(f"Initial PID parameters: Kp={initial_pid[0]:.4f}, Ki={initial_pid[1]:.4f}, Kd={initial_pid[2]:.4f}")
    
    # Run optimization
    optimization_result = optimize_pid(
        input_csv=input_csv,
        system_params=system_params,
        setpoint_col=None,  # Use input as setpoint
        output_csv="optimized_pid_output.csv",
        initial_pid=initial_pid,
        optimize_method=optimize_method,
        weights=weights
    )
    
    # Display results
    print("\nOptimization completed!")
    print("\nOptimized PID Parameters:")
    print(f"Kp: {optimization_result['pid_params']['kp']:.4f}")
    print(f"Ki: {optimization_result['pid_params']['ki']:.4f}")
    print(f"Kd: {optimization_result['pid_params']['kd']:.4f}")
    
    print("\nPerformance Metrics:")
    if optimization_result['performance']:
        print(f"Rise Time: {optimization_result['performance']['rise_time']:.4f} seconds")
        print(f"Settling Time: {optimization_result['performance']['settling_time']:.4f} seconds")
        print(f"Overshoot: {optimization_result['performance']['overshoot']:.2f}%")
        print(f"Steady-state Error: {optimization_result['performance']['steady_state_error']:.4f}%")
    print(f"Objective Score: {optimization_result['objective_score']:.4f}")
    
    print("\nResults saved to 'optimized_pid_output.csv' and 'optimized_pid_results.png'")