import numpy as np
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_implied_volatility as impvol
import streamlit as st

# Streamlit title
st.title("Heston Model Simulator and Implied Volatility Surface")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S₀)", value=100.0)
nu0 = st.sidebar.number_input("Initial Volatility (ν₀)", value=0.2)
theta = st.sidebar.number_input("Long-term Volatility Mean (θ)", value=0.2)
kappa = st.sidebar.number_input("Mean Reversion Rate (κ)", value=3.0)
ksi = st.sidebar.number_input("Volatility of Volatility (ξ)", value=0.5)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.001)
rho = st.sidebar.slider("Correlation (ρ)", -1.0, 1.0, -0.65)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
steps = st.sidebar.number_input("Steps", value=252)
M = st.sidebar.number_input("Simulations (Paths)", value=1000)


# Simulate Heston Model
@st.cache_data
def heston_model(steps, r, rho, S0, nu0, theta, kappa, ksi, T, M):
    dt = T / steps
    mu = np.array([0, 0])
    sig = np.array([[1, rho], [rho, 1]])
    bminc = np.random.multivariate_normal(mu, sig, (steps, M))
    Wnuinc = np.squeeze(bminc[:, :, 0])
    WSinc = np.squeeze(bminc[:, :, 1])
    St = np.full((steps, M), S0)
    nut = np.full((steps, M), nu0)

    for i in range(1, steps):
        nut[i] = np.abs(nut[i - 1] + kappa * (theta - nut[i - 1]) * dt +
                        ksi * np.sqrt(nut[i - 1] * dt) * Wnuinc[i - 1, :])
        St[i] = St[i - 1] + r * St[i - 1] * dt + \
                np.sqrt(nut[i - 1] * dt) * St[i - 1] * WSinc[i - 1, :]

    return St, nut, Wnuinc, WSinc


# Run simulation
St, nut, Wnuinc, WSinc = heston_model(steps, r, rho, S0, nu0, theta, kappa, ksi, T, M)
t = np.linspace(0, T, steps)

# Plot results
st.subheader("Correlated Brownian Motions")
fig1, ax1 = plt.subplots()
ax1.plot(Wnuinc[:, 0].cumsum(), label="$W^\\nu_t$", color='blue')
ax1.plot(WSinc[:, 0].cumsum(), label="$W^S_t$", color='orange')
ax1.set_xlabel("Time Steps")
ax1.set_ylabel("Brownian Motion")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

st.subheader("Volatility Paths")
fig2, ax2 = plt.subplots()
ax2.plot(t, nut[:, :10])
ax2.axhline(theta, linestyle='--', color='black', label="θ (Long-term mean)")
ax2.set_title("Volatility over Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Volatility")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

st.subheader("Stock Price Paths")
fig3, ax3 = plt.subplots()
ax3.plot(t, St[:, :10])
ax3.set_title("Stock Price Paths")
ax3.set_xlabel("Time")
ax3.set_ylabel("Price")
ax3.grid(True)
st.pyplot(fig3)

# Compute implied volatilities
ST = St[-1, :]
K = np.arange(50, 200, 2)
callprices = np.array([np.mean(np.exp(-r * T) * np.maximum(ST - k, 0)) for k in K])
putprices = np.array([np.mean(np.exp(-r * T) * np.maximum(k - ST, 0)) for k in K])

callimpvols = impvol(callprices, S0, K, T, r, flag="c", q=0, return_as="numpy")
putimpvols = impvol(putprices, S0, K, T, r, flag="p", q=0, return_as="numpy")

st.subheader("Implied Volatility Smile")
fig4, ax4 = plt.subplots()
ax4.plot(K, callimpvols, label="Calls")
ax4.plot(K, putimpvols, label="Puts")
ax4.set_title("Implied Volatility vs Strike Price")
ax4.set_xlabel("Strike")
ax4.set_ylabel("Implied Volatility")
ax4.grid(True)
ax4.legend()
st.pyplot(fig4)
