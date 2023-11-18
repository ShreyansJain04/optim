import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import base64
from matplotlib.animation import FuncAnimation


class AdamOptimizer:
    def __init__(self, func, learning_rate, x_start):
        self.params = {
            "x": torch.tensor([x_start], dtype=torch.float32, requires_grad=True)
        }
        self.func = func
        self.optimizer = torch.optim.Adam([self.params["x"]], lr=learning_rate)

    def step(self):
        self.optimizer.zero_grad()
        loss = self.func(self.params["x"])
        loss.backward()
        self.optimizer.step()
        return self.params["x"].item(), loss.item()


def function(x):
    return (x**6 / 12) - (9 * x**5 / 10) + (13 * x**4 / 4) - 4 * x**3 + 4.5


def generate_animation(learning_rate):
    optimizer = AdamOptimizer(function, learning_rate=learning_rate, x_start=-0.6)
    fig, ax, line = init_plot()
    anim = FuncAnimation(
        fig, animate, fargs=(optimizer, line), frames=400, interval=100, blit=True
    )
    anim.save("optimization_animation.mp4", writer="ffmpeg", fps=30)
    return optimizer  # Return optimizer object


def init_plot():
    fig, ax = plt.subplots()
    x_np_values = np.linspace(-5, 5, 400)
    y_np_values = function(x_np_values)
    ax.plot(x_np_values, y_np_values, label="Function")
    (line,) = ax.plot([], [], "ro", label="Current Position")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 10)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Optimization Path using Adam in PyTorch")
    plt.legend()
    return fig, ax, line


def animate(i, optimizer, line):
    x, z = optimizer.step()
    line.set_data([x], [function(x)])
    return [line]  # Return a list of Artist objects


def main():
    markdown_content = """
    # Optimizers in Machine Learning

    In the optimization world, optimizers play a crucial role in improving machine learning models. They adjust the model's settings to reduce the prediction errors, which are measured by the loss function.

    Optimizers come in two main types:

    *First-Order Optimizers:* These use partial derivative values to guide the optimizer. They indicate if the loss is increasing or decreasing at a particular point, like a compass guiding the optimizer to minimize the loss.

    *Second-Order Optimizers:* These go further by considering both first-order and second-order derivatives. Delving deeper into the curvature of the loss function, they can, in some cases, achieve more efficient optimization. This provides deeper insights into the shape of the loss function, making optimization more efficient.

    Choosing the right optimizer can make a significant difference in the training speed and performance of machine learning models. For example, Gradient Descent is a widely used first-order optimizer, while more complex tasks may require second-order optimizers like Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS).

    Selecting the right optimizer is crucial in machine learning model development, as each one has its strengths and weaknesses.

    A non-exhaustive list of first-order optimizers are:
    - Gradient Descent
    - Stochastic Gradient Descent
    - Mini Batch Gradient Descent
    - Momentum
    - Nesterov Accelerated Gradient
    - Adagrad
    - Adadelta
    - RMSprop
    - Adam
    - AdaMax
    - Nadam

    In this blog, we shall be limiting our discussion to the following optimizers:
    - Gradient Descent
    - Stochastic Gradient Descent
    - Mini Batch Gradient Descent
    - Momentum
    - Nesterov Accelerated Gradient
    - RMSprop
    - Adam.

    ## Gradient Descent (Vanilla Gradient Descent)

    *Concept:* This optimizer takes steps proportional to the negative gradient of the function at the current point. Think of it as trying to find the lowest point in a valley by walking in the direction where the slope is steepest.

    In the vanilla gradient descent, all the training data is used to compute the gradient of the loss function w.r.t the parameters. So if we have 10000 datapoints, all of them undergo a forwards pass in a neural network and then the loss is computed. The loss is then backpropagated through the network to compute the gradients w.r.t the parameters. We will have 10000 forward passes and one backward pass in this case.

    *Advantages:*

        Deterministic Updates: Since the gradients are computed using all the training 
        data, the updates to the parameters are deterministic. This means that the updates 
        are not dependent on the order of the training data. This is a very desirable 
        property as we want the model to converge to the same minima regardless of the 
        order of the training data. This also results in a smooth convergence.

        Suitable for small datasets: Since this uses all the datapoints to calculate the 
        gradients, this leads to a more stable convergence.

    *Disadvantages:*

        Resource Intensive: Since we use all the datapoints to calculate the gradients, 
        all the values have to be stored in the memory. This can be a problem for large 
        datasets.

    ### Stochastic Gradient Descent

    *Concept:* Instead of the entire dataset, SGD uses a single data point at each iteration to move towards the minima. So we will have 10000 forward passes and 10000 backward passes. This is because the gradients are calculated using only one datapoint, and hence the direction of the gradient is not very accurate. This leads to a very noisy convergence as shown.

    *Advantages:*

        Less Resource Intensive: Since we use only one datapoint to calculate the gradients, 
        we need to store only one datapoint in the memory. This is very useful for large 
        datasets.
        
        Faster Convergence: More frequent updates lead to quicker convergence as compared 
        to vanilla gradient descent.

    *Disadvantages:*

        Noisy Convergence: Since we use only one datapoint to calculate the gradients, 
        the direction of the gradient is not very accurate. This leads to a very noisy 
        convergence.

        Non-Deterministic Updates: Since the gradients are calculated using only one 
        datapoint, the updates to the parameters are not deterministic. This means that 
        the updates are dependent on the order of the training data. This is not a very 
        desirable property as we want the model to converge to the same minima regardless 
        of the order of the training data. Sometimes it can also lead to non-convergence 
        of the model.

    ### Mini Batch Gradient Descent

    *Concept:* A middle ground between SGD and Vanilla Gradient Descent, it uses batches of data points for each update. In Mini Batch Gradient Descent, we use a batch of datapoints to calculate the gradients. So for each epoch, we will have 10000/batch size forward passes and 1 backward pass.

    *Advantages:*

        Less Resource Intensive: Since we use only a batch of datapoints to calculate the 
        gradients, we need to store only a batch of datapoints in the memory. This is very 
        useful for large datasets.

        Faster Convergence: More frequent updates lead to quicker convergence as compared 
        to vanilla gradient descent.

        As compared to SGD, it has a more stable convergence as shown.

    *Disadvantages:*

        Noisy Convergence:* Since we use only a batch of datapoints to calculate the 
        gradients, the direction of the gradient is not very accurate. This leads to 
        a very noisy convergence as shown in Fig 1.

    We observed that SGD and Mini Batch Gradient Descent have a noisy convergence as compared to Vanilla Gradient Descent. This is because the direction of the gradient is not very accurate.

    What if we want a faster convergence? One way is to reduce the observed sharp edges in SGD and Mini Batch SGD. This can be done by using a momentum term. This is where Momentum comes in.

    ## Momentum

    Momentum's Intuition is that it confidently speeds up the convergence by accumulating the gradient of the past steps. It is a method that helps accelerate SGD in the relevant direction and dampens oscillations as can be seen in Fig 1. It does this by adding a fraction of the update vector of the past time step to the current update vector. This fraction is called the momentum coefficient and is usually set to 0.9 or a similar value.

    It's akin to a ball rolling down a hill, gathering speed – unlike standard SGD, which feels more step-by-step.

    This also gives us the intuition that if we have a very noisy convergence, we can use momentum to dampen the oscillations and get a smoother convergence. But it's not all good. Momentum can also lead to overshooting of the minima and lots of to and fro oscillations.

    Another question that can arise is if the term momentum is actually related to physics?
    The answer is yes. The momentum term is actually related to the momentum term in physics.

    Inertia is the property of an object to resist changes in its state of motion.
    In optimization:
    Inertia is the property of an object to resist changes in its state of motion.
    The learning rate acts as the acceleration here and the force is the gradient. So we can see that the momentum term in SGD is actually related to the momentum term in physics.
    """

    st.markdown(markdown_content)
    st.latex(r"v_{t+1} = \mu \cdot v_t - \alpha \cdot \nabla J(\theta_t)")
    st.latex(r"\theta_{t+1} = \theta_t + v_{t+1}")

    file_ = open("./media/animation_1.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    # Set the width and height attributes as needed
    width = 800
    height = 300

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="{width}" height="{height}">',
        unsafe_allow_html=True,
    )


    st.title("SGD with Momentum and Contour Plot")

    # Input for initial parameter, learning rate, and momentum
    initial_x = st.text_input("Initial X:", "2.0")
    initial_y = st.text_input("Initial Y:", "2.0")
    learning_rate = st.slider("Learning Rate:", 0.01, 1.0, 0.1, step=0.01)
    momentum = st.slider("Momentum:", 0.0, 1.0, 0.9, step=0.01)

    # Initialize variables using st.session_state
    st.session_state.x_param = st.session_state.get("x_param", float(initial_x))
    st.session_state.y_param = st.session_state.get("y_param", float(initial_y))
    st.session_state.x_velocity = st.session_state.get("x_velocity", 0)
    st.session_state.y_velocity = st.session_state.get("y_velocity", 0)
    st.session_state.step = st.session_state.get("step", 0)

    if "param_history" not in st.session_state:
        st.session_state.param_history = []

    if st.button("Next Step"):
        # Compute the gradients (you can replace this with your own function)
        grad_x = 2 * st.session_state.x_param
        grad_y = 2 * st.session_state.y_param
        st.session_state.x_velocity = (
            momentum * st.session_state.x_velocity - learning_rate * grad_x
        )
        st.session_state.y_velocity = (
            momentum * st.session_state.y_velocity - learning_rate * grad_y
        )
        st.session_state.x_param += st.session_state.x_velocity
        st.session_state.y_param += st.session_state.y_velocity
        st.session_state.step += 1
        # Append the current parameter values to the history
        st.session_state.param_history.append(
            (st.session_state.x_param, st.session_state.y_param)
        )

    # Create a grid for the contour plot
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Display the parameter updates
    st.header("SGD with Momentum Parameter Updates:")
    # st.write(f"Initial X: {initial_x}, Initial Y: {initial_y}, Learning Rate: {learning_rate}, Momentum: {momentum}, Step: {st.session_state.step}")
    # st.write(f"Initial X: {initial_x}, Initial Y: {initial_y}, Learning Rate: {learning_rate}, Momentum: {momentum}, Step: {st.session_state.step}")
    st.text(
        f"Initial X: {initial_x}, Initial Y: {initial_y}, Learning Rate: {learning_rate}, Momentum: {momentum}, Step: {st.session_state.step}"
    )

    # Plot contour and parameter updates
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=np.logspace(0, 2, 20), cmap="viridis")

    # Plot the parameter path as a line
    if len(st.session_state.param_history) > 1:
        param_history = np.array(st.session_state.param_history)
        plt.plot(
            param_history[:, 0],
            param_history[:, 1],
            marker="o",
            linestyle="-",
            color="r",
            label="Parameter Path",
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    st.pyplot(plt)

    st.markdown( """
    ## Nesterov Accelerated Gradient:
    *Concept:* While momentum only takes into account the previous gradients, NAG looks ahead and corrects direction, leading to better adjustements in the steps.
    The Math and the intuition behind it: 
    In the regular momentum update, we first compute the gradient at the current position and then take a big jump in the direction of the accumulated gradient (momentum). For NAG instead of calculating the gradient at the current position, we calculate the gradient after the momentum update, this is the look ahead feature and gives a better approximation to the gradient in the next step.

    The math behind it:

    *Initialization*:"""
    )

    st.write(
        r"""
    - Set the initial parameters: $\theta_0$
    - Set the learning rate: $\eta$
    - Set the momentum parameter: $\mu
    """
    )

    st.markdown(
        """
    *Nesterov Update Rule*:
    - Compute the gradient of the loss function at the predicted future parameter values:

    """
    )

    st.latex(r"\nabla L(\theta_t + \mu v_t)")

    st.markdown(
        """
    - Update the velocity vector:
    """
    )

    st.latex(r"v_{t+1} = \mu v_t - \eta \nabla L(\theta_t + \mu v_t)")

    st.markdown(
        """
    - Update the parameters using the velocity:
    """
    )

    st.latex(r"\theta_{t+1} = \theta_t + v_{t+1}")

    st.write(
        """
    Here,
    - $\\theta_t$ represents the current parameter values at iteration $t$.
    - $\\eta$ is the learning rate, controlling the step size.
    - $\\mu$ is the momentum parameter, which determines how much of the previous velocity to keep.
    """
    )

    st.write(
        """
    Nesterov Accelerated Gradient has been shown to converge faster than the standard gradient descent in many cases, making it a popular choice for optimization in deep learning and other machine learning applications.

    Imagine hiking down a foggy mountain trail to reach a base camp quickly and safely. In this challenging setting, Standard Momentum is like a hiker who can only see the path directly in front of them due to the fog. They make decisions based on what's immediately visible, risking overshooting or getting stuck when the path changes unexpectedly.

    In contrast, NAG is like a hiker who can anticipate the path slightly ahead despite the fog. They periodically assess the terrain in advance, enabling more informed choices and efficient progress.


    *Advantages:*

        Overshooting Reduction: Consider a scenario where the current gradient points in a 
        certain direction, but due to noise or complex curvature of the loss landscape, 
        it may not be the best direction to move in the long term. Standard Momentum would 
        keep accumulating velocity in the current direction, potentially overshooting the 
        optimal solution. NAG anticipates that the parameters will move in the direction 
        of the lookahead point, which allows it to "course-correct" by reducing the 
        accumulated velocity in the original direction. This anticipatory adjustment helps 
        in reducing overshooting.

        Noise Handling: When gradients are noisy, they can cause erratic updates in 
        standard Momentum. The lookahead mechanism of NAG dampens the impact of noisy 
        gradients. By considering the gradient at the lookahead point, NAG effectively 
        filters out some of the noise, resulting in more stable and reliable updates.

    *Disadvantages:*

        More computationally complex: It requires an additional gradient computation at 
        each iteration, which can be expensive for large models.

    *Additional hyperparameter to tune:* The momentum parameter $\\mu$ needs to be tuned in addition to the learning rate $\\eta$.

    *Adagrad*
    Adagrad is an optimization algorithm that adapts the learning rate for each parameter during training, allowing for more effective updates.

    ## Adagrad

    Adagrad is an optimization algorithm that adapts the learning rate for each parameter during training, allowing for more effective updates.

    ### Mathematical Initialization:
    """
    )

    st.write(
        """
    - Set the initial parameters: $\\theta_0$
    - Define the learning rate: $\\eta$
    """
    )

    st.markdown(
        """
    ### Adagrad Update Rule:
    """
    )

    st.write(
        """
    - Compute the gradient of the loss function: $\\nabla L(\\theta_t)$           

    """
    )

    st.markdown(
        """
    - Adapt the learning rate for each parameter based on the historical gradient information:
    """
    )

    st.latex(
        r"""

    i: \frac{\eta}{\sqrt{G_{ii} + \epsilon}}
            """
    )

    st.write(
        """

    Here, $G_{ii}$ represents the sum of the squares of historical gradients for parameter $i$, and $\epsilon$ is a small constant to prevent division by zero.


    """
    )

    st.markdown("- Update the parameters:")

    st.latex(
        r"""

    \theta_{t+1} = \theta_t - i \cdot \nabla L(\theta_t)

    """
    )

    st.markdown(
        """
    Advantages:
                
        Learning Rate Adaptation: Adagrad dynamically adjusts the learning rate for each
        parameter, scaling it inversely with the square root of the sum of past squared
        gradients. This adaptability allows Adagrad to allocate larger learning rates to
        parameters with infrequent updates and smaller learning rates to parameters with
        frequent updates. As a result, it can effectively handle different convergence
        rates across parameters.
                
        No Manual Learning Rate Tuning: Adagrad eliminates the need to manually tune
        the learning rate, which can be challenging to get right in practice.

    Disadvantages:

        Accumulation of Gradients: The denominator starts accumulating squares, which
        can lead to a very small learning rate and effectively stop the learning process.
                
        Memory Intensive: Adagrad accumulates the squares of the gradients in the 
        denominator, which can become very large for large models and lead to memory
        issues.

    A point to note: Although it can correct learning rate according to the need to a good extent, a very poor choice of learning rate can still lead to a bad convergence.

    """
    )

    st.write(
        """
    ## RMSProp

    RMSProp is an optimization algorithm that builds upon the Adagrad method by addressing some of its limitations. It adapts the learning rate for each parameter during training, offering more stable and efficient updates.
    Mathematical Initialization:

    - Set the initial parameters: $\\theta_0$
    - Define the learning rate: $\\eta$
    - Specify the decay parameter: $\\rho$ (typically close to 1)

    ### RMSProp Update Rule:

    - Compute the gradient of the loss function: $\\nabla L(\\theta_t)$
    - Calculate the exponentially moving average of the squared gradients for each parameter:
    """
    )

    st.latex(
        r"""

    E[G_{ii}] = \rho E[G_{ii}] + (1 - \rho)(\nabla L(\theta_t))^2
    """
    )

    st.write(
        """        
        Here, $E[G_{ii}]$ represents the exponentially moving average of the squared gradients for parameter i.
    """
    )

    st.markdown(
        """
    - Adapt the learning rate for each parameter:
    """
    )

    st.latex(
        r"""

    i: \frac{\eta}{\sqrt{E[G_{ii}] + \epsilon}}
    """
    )

    st.write(
        """
    Here, $\epsilon$ is a small constant to prevent division by zero.
    """
    )

    st.markdown(
        """
    - Update the parameters:
    """
    )

    st.latex(
        r"""

    \theta_{t+1} = \theta_t -  i \cdot \nabla L(\theta_t)
    """
    )

    st.write(
        """

    Advantages:

        Stability: RMSProp addresses the issue of rapidly decreasing learning rates by
        introducing an exponentially moving average for the squared gradients. This 
        results in a more stable and adaptive learning rate.

        No Manual Learning Rate Tuning: Similar to Adagrad, RMSProp eliminates the need 
        for manual tuning of the learning rate, making it a convenient choice for 
        practical applications.

    Disadvantages:

        Memory Intensive: RMSProp still accumulates historical gradient information, 
        which can be memory-intensive for large models, though to a lesser extent than 
        Adagrad.

        Hyperparameter Tuning: RMSProp introduces the decay parameter ρ, which needs 
        to be tuned along with the learning rate. Finding the right values for these 
        hyperparameters can be a trial-and-error process.

    In summary, RMSProp overcomes some of the issues of Adagrad, particularly the problem of a decreasing learning rate. It provides more stable training and is often used in deep learning models. However, it still requires careful tuning of hyperparameters and can be memory-intensive for very large models.


    ### Adam (Adaptive Moment Estimation)

    Adam is a popular optimization algorithm that combines elements of both RMSProp and momentum. It adapts the learning rate and keeps track of past gradients' moving averages, providing efficient and stable updates.

    #### Mathematical Initialization:

    - Set the initial parameters: $\\theta_0$
    - Define the learning rate: $\\eta$
    - Specify the decay parameters: $\\beta_1$ and $\\beta_2$ (both typically close to 1)
    """
    )

    st.markdown(
        """

    #### Adam Update Rule:

    - Compute the gradient of the loss function: $\\nabla L(\\theta_t)$

    """
    )

    st.markdown(
        """
    - Calculate the first and second moments (moving averages) of the gradients for each parameter:
    """
    )

    st.latex(
        r"""

    m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)  
        """
    )

    st.latex(
        r"""
    v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla L(\theta_t))^2
    """
    )

    st.write(
        """
    Here, $m_t$ and $v_t$ represent the first and second moments of the gradients, respectively.
    """
    )

    st.markdown(
        """
    - Correct the moments for bias:
    """
    )

    st.latex(
        r"""

    \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
            """
    )

    st.latex(
        r"""
    \hat{v}_t = \frac{v_t}{1 - \beta_2^t}

    """
    )

    st.markdown(
        """
    - Adapt the learning rate (i) for each parameter:
    """
    )

    st.latex(
        r"""

    i: \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}
    """
    )

    st.write(
        """
    Here, $\\epsilon$ is a small constant to prevent division by zero.
    """
    )

    st.markdown("- Update the parameters:")

    st.latex(
        r"""

    \theta_{t+1} = \theta_t - i \cdot \hat{m}_t

    """
    )

    st.markdown(
        """
    *Advantages*:

        Efficiency: Adam combines the benefits of adaptive learning rates (like RMSProp) 
        and momentum. It effectively handles different convergence rates for parameters 
        and speeds up convergence.

        Stability: The correction for bias in the moments ensures that the algorithm 
        remains stable even during the initial iterations.

    *Disadvantages*:

        Hyperparameter Sensitivity: Like RMSProp, Adam introduces decay parameters (betas), 
        which require tuning for optimal performance. Incorrect settings can affect 
        convergence.
        
        Memory Usage: Adam accumulates moments for each parameter, which can lead to 
        increased memory usage, especially in deep learning models.

    Adam is widely used in deep learning due to its efficiency and stability, but it does require careful tuning of its hyperparameters for different tasks and models.

    Lets look at an example of how the different optimizers perform on a complex loss surface.
    """
    )

    st.title("Optimization Animation")
    st.write("This generates an optimization animation using the Adam optimizer.")

    learning_rate = st.slider("Learning Rate", 0.01, 0.99, 0.5, step=0.01)

    st.subheader("Optimization Animation")
    st.video(f"/media/{learning_rate:.2f}.mp4")

if __name__ == "__main__":
    main()
