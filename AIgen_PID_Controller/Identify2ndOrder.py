#====================================================
#AI Generated PID Control Program 2025 MIT License makkiblog.com
#Wave gen identification module WIP
#Code is incomplete

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

def second_order_step(t, K, zeta, wn, delay=0):
    """
    Second order step response function.
    
    Parameters:
    t: time vector
    K: DC gain
    zeta: damping ratio
    wn: natural frequency (rad/s)
    delay: time delay (optional)
    
    Returns:
    y: step response
    """
    t_shifted = np.maximum(t - delay, 0)
    
    if zeta < 1:  # Underdamped
        wd = wn * np.sqrt(1 - zeta**2)
        y = K * (1 - np.exp(-zeta * wn * t_shifted) * (np.cos(wd * t_shifted) + 
                                                      (zeta * wn / wd) * np.sin(wd * t_shifted)))
    elif zeta == 1:  # Critically damped
        y = K * (1 - np.exp(-wn * t_shifted) * (1 + wn * t_shifted))
    else:  # Overdamped
        alpha = zeta * wn
        beta = wn * np.sqrt(zeta**2 - 1)
        y = K * (1 - (np.exp(-alpha * t_shifted) * np.cosh(beta * t_shifted) + 
                      (alpha / beta) * np.exp(-alpha * t_shifted) * np.sinh(beta * t_shifted)))
        
    # Replace any NaN values with 0
    y = np.nan_to_num(y, nan=0.0)
    return y

def identify_system_parameters(csv_file):
    """
    Identify second-order system parameters from input-output data.
    
    Parameters:
    csv_file: Path to CSV file with Time, Input, Output columns
    
    Returns:
    K: DC gain
    tau: time constant (1/wn)
    zeta: damping ratio
    """
    # Load data
    data = pd.read_csv(csv_file)
    
    # Extract data
    t = data['Time'].values
    u = data['Input'].values
    y = data['Output'].values
    
    # Normalize input/output if needed
    u_steady = u[-1]  # Assuming the input reaches steady state
    if u_steady != 1.0 and u_steady != 0:
        y = y / u_steady
    
    # Initial parameter guesses
    # Estimate K from the steady-state gain
    K_guess = y[-1] / u[-1] if u[-1] != 0 else y[-1]
    
    # Estimate rise time and use it to guess wn
    rise_idx = np.argmax(y > 0.1 * y[-1])
    rise_time = t[rise_idx] if rise_idx > 0 else t[1]
    wn_guess = 1.8 / rise_time  # Approximate relationship
    
    # Estimate overshoot to guess damping ratio
    peak_idx = np.argmax(y)
    peak_value = y[peak_idx]
    steady_state = y[-1]
    overshoot = (peak_value - steady_state) / steady_state if steady_state != 0 else 0
    
    # If no overshoot, guess slightly underdamped
    zeta_guess = 0.7 if overshoot <= 0 else -np.log(overshoot) / np.sqrt(np.pi**2 + np.log(overshoot)**2)
    
    # Constrain zeta to be between 0 and 2
    zeta_guess = max(0.01, min(2.0, zeta_guess))
    
    # Initial guesses
    p0 = [K_guess, zeta_guess, wn_guess]
    bounds = ([0.1, 0.01, 0.1], [10.0, 2.0, 100.0])
    
    # Fit the model to the data
    try:
        # For step input
        if np.all(u[1:] == u[1]):
            popt, _ = curve_fit(second_order_step, t, y, p0=p0, bounds=bounds, maxfev=10000)
            K, zeta, wn = popt
        else:
            # For arbitrary input, we'd need to implement a convolution approach
            # This is a simplified approach for demonstration
            # In practice, you might want to use system identification tools like MATLAB's System Identification Toolbox
            raise NotImplementedError("Arbitrary input identification not yet implemented")
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        print("Using initial guesses instead")
        K, zeta, wn = p0
    
    # Calculate time constant
    tau = 1 / wn
    
    return K, tau, zeta, wn

def plot_results(csv_file, K, tau, zeta, wn):
    """
    Plot the original data and the fitted model response.
    
    Parameters:
    csv_file: Path to CSV file
    K, tau, zeta, wn: System parameters
    """
    # Load data
    data = pd.read_csv(csv_file)
    
    # Extract data
    t = data['Time'].values
    u = data['Input'].values
    y = data['Output'].values
    
    # Generate model response
    if np.all(u[1:] == u[1]):  # Step input
        y_model = second_order_step(t, K, zeta, wn)
        # Scale by input magnitude
        y_model = y_model * u[1]
    else:
        # For arbitrary input, need to convolve input with impulse response
        # This is simplified for demonstration
        y_model = np.zeros_like(y)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(t, u, 'b-', label='Input')
    plt.plot(t, y, 'r-', label='Measured Output')
    plt.plot(t, y_model, 'g--', label='Model Output')
    plt.grid(True)
    plt.legend()
    plt.title('System Identification Results')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(212)
    plt.plot(t, y - y_model, 'k-')
    plt.grid(True)
    plt.title('Error (Measured - Model)')
    plt.xlabel('Time')
    plt.ylabel('Error')
    
    plt.tight_layout()
    plt.savefig('system_identification_results.png')
    plt.show()

def main():
    # Get file path from user
    csv_file = input("Enter the path to your CSV file: ")
    
    try:
        # Identify system parameters
        K, tau, zeta, wn = identify_system_parameters(csv_file)
        
        # Print results
        print("\nIdentified System Parameters:")
        print(f"DC Gain (K): {K:.4f}")
        print(f"Time Constant (tau): {tau:.4f} seconds")
        print(f"Damping Ratio (zeta): {zeta:.4f}")
        print(f"Natural Frequency (wn): {wn:.4f} rad/s")
        
        # Check system type based on damping ratio
        if zeta < 1:
            system_type = "Underdamped"
        elif zeta == 1:
            system_type = "Critically Damped"
        else:
            system_type = "Overdamped"
        print(f"System Type: {system_type}")
        
        # Calculate additional characteristics
        if zeta < 1:
            # Underdamped characteristics
            wd = wn * np.sqrt(1 - zeta**2)  # Damped natural frequency
            overshoot = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
            peak_time = np.pi / wd
            settling_time_2p = 4 / (zeta * wn)
            
            print(f"Damped Natural Frequency (wd): {wd:.4f} rad/s")
            print(f"Percent Overshoot: {overshoot:.2f}%")
            print(f"Peak Time: {peak_time:.4f} seconds")
            print(f"2% Settling Time: {settling_time_2p:.4f} seconds")
        
        # Plot results
        plot_results(csv_file, K, tau, zeta, wn)
        
        # Show transfer function
        print("\nTransfer Function:")
        print(f"G(s) = {K:.4f} / (s^2 + {2*zeta*wn:.4f}s + {wn**2:.4f})")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()