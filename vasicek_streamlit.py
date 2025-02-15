#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Vasicek Model Implementation using Euler's Method
def vasicek(r0, a, b, sigma, T, num_steps, num_paths):
    dt = T / num_steps
    rates = np.zeros((num_steps + 1, num_paths))
    rates[0] = r0
    
    for t in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), num_paths)  # Brownian motion
        rates[t] = rates[t-1] + a * (b - rates[t-1]) * dt + sigma * dW
        
    return rates

# Streamlit UI
st.title("Vasicek Model - Interest Rate Simulation")

# User inputs
r0 = st.sidebar.slider("Initial Interest Rate (r0)", 0.001, 0.1, 0.01, 0.001)
a = st.sidebar.slider("Speed of Mean Reversion (a)", 0.1, 1.0, 0.4, 0.05)
b = st.sidebar.slider("Long-term Mean (b)", 0.001, 0.1, 0.02, 0.001)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 0.2, 0.05, 0.01)
T = st.sidebar.slider("Time Horizon (Years)", 1, 50, 20, 1)
num_steps = st.sidebar.slider("Number of Time Steps", 100, 20000, 10000, 500)
num_paths = st.sidebar.slider("Number of Simulated Paths", 1, 50, 15, 1)

# Simulate Vasicek model
simulated_rates = vasicek(r0, a, b, sigma, T, num_steps, num_paths)

# Time axis
time_axis = np.linspace(0, T, num_steps + 1)

# Expected values
average_rates = [r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t)) for t in time_axis]

# Standard deviation
std_dev = [(sigma**2 / (2 * a) * (1 - np.exp(-2 * a * t)))**0.5 for t in time_axis]

# Upper and lower bounds
upper_bound = [average_rates[i] + 2 * std_dev[i] for i in range(len(time_axis))]
lower_bound = [average_rates[i] - 2 * std_dev[i] for i in range(len(time_axis))]

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Vasicek Model - Simulated Interest Rate Paths')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Interest Rate')

for i in range(num_paths):
    ax.plot(time_axis, simulated_rates[:, i], linewidth=1)

ax.plot(time_axis, average_rates, color='black', linestyle='--', label='Average', linewidth=2)
ax.plot(time_axis, upper_bound, color='grey', linestyle='--', label='Upper Bound', linewidth=2)
ax.plot(time_axis, lower_bound, color='grey', linestyle='--', label='Lower Bound', linewidth=2)

ax.legend()
st.pyplot(fig)


# In[1]:


get_ipython().system('streamlit run vasicek_streamlit.py')


# In[ ]:




