import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return x**2

def grad(x):
    return 2*x

def header(x):
    s = 2*x/(np.sqrt(1 + 4*x**2))
    c = 1/(np.sqrt(1 + 4*x**2))

    return c, s

def header_2(x):
    s = np.sin(np.arctan(grad(x)))
    c = np.cos(np.arctan(grad(x)))

    return s, c

def main():
    st.title("Velocity and Gradient Vectors")

    # Input for initial parameter, learning rate, and momentum
    initial_x = st.text_input("Initial X:", "2.0")
    learning_rate = st.slider("Learning Rate:", 0.01, 1.0, 0.1, step=0.01)
    momentum = st.slider("Momentum:", 0.0, 1.0, 0.9, step=0.01)

    # Initialize variables using st.session_state
    st.session_state.x_param = st.session_state.get("x_param", float(initial_x))
    st.session_state.x_velocity = st.session_state.get("x_velocity", 0)
    st.session_state.step = st.session_state.get("step", 0)

    if st.button("Reset"):
        st.session_state.x_param = float(initial_x)
        st.session_state.x_velocity = 0
        st.session_state.step = 0

    if "param_history" not in st.session_state:
        st.session_state.param_history = []

    # Calculate the gradient before using it in the next step
    grad_x = grad(st.session_state.x_param)

    if st.button("Next Step"):
        # Compute the gradients (you can replace this with your own function)
        st.session_state.x_velocity = (momentum * st.session_state.x_velocity - learning_rate * grad_x)
        st.session_state.x_param += st.session_state.x_velocity
        st.session_state.step += 1
        # Append the current parameter values to the history
        st.session_state.param_history.append((st.session_state.x_param, function(st.session_state.x_param)))

    fig, ax = plt.subplots()
    x_np_values = np.linspace(-5, 5, 400)
    y_np_values = function(x_np_values)
    ax.plot(x_np_values, y_np_values, label="Function")
    line, = ax.plot([], [], "ro", label="Current Position")

    # Calculate tangent vector (unit vector)
    head_x, head_y = header(st.session_state.x_param)

    # Plot the velocity vector with arrowhead
    ax.arrow(st.session_state.x_param, function(st.session_state.x_param),
             head_x * st.session_state.x_velocity, head_y * st.session_state.x_velocity,
             color='green', width=0.03, head_width=0.1, head_length=0.1, label='Velocity Vector')


    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 10)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Optimization Path with Velocity and Gradient Vectors")
    plt.legend()
    line.set_data([st.session_state.x_param], [function(st.session_state.x_param)])
    st.pyplot(fig)

if __name__ == "_main_":
    main()
