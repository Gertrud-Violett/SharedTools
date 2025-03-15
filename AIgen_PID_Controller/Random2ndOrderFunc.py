#====================================================
#AI Generated PID Control Program 2025 MIT License makkiblog.com
#Wavegen Module


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim

# Define system parameters
K = 5  # Gain (ensuring it is ≤ 10)
tau = 2  # Time constant
zeta = 0.5  # Damping ratio


# Define the second-order system transfer function: H(s) = K / (tau^2 * s^2 + 2*zeta*tau*s + 1)
num = [K]
den = [tau**2, 2*zeta*tau, 1]
system = lti(num, den)

# Generate time vector to cover at least 10 wavelengths
t_end = 145  # Adjusted for at least 10 oscillations
t = np.linspace(0, t_end, 2000)  # Increase resolution

# Define an arbitrary input signal (piecewise changes)
u = np.piecewise(t, 
                 [t < 10, (t >= 10) & (t < 40), (t >= 40) & (t < 70), t >= 70], 
                 [0, 1, -1, 0.5])

# Simulate system response
t_out, y_out, _ = lsim(system, U=u, T=t)

# Save to CSV
df = pd.DataFrame({"Time (s)": t, "Input": u, "Output": y_out})
csv_filename = "./second_order_response.csv"
df.to_csv(csv_filename, index=False)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, u, label="Input (Aileron Angle)", linestyle="dashed")
plt.plot(t, y_out, label="Output (Pitch Angle)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Response")
plt.title("Simulated Second-Order System Response")
plt.legend()
plt.grid()
plt.show()

# Provide CSV file link
csv_filename
