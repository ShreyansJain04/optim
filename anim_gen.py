import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import base64

class AdamOptimizer:
    def __init__(self, func, learning_rate, x_start):
        self.params = {'x': torch.tensor([x_start], requires_grad=True)}
        self.func = func
        self.optimizer = torch.optim.Adam(self.params.values(), lr=learning_rate)

    def step(self):
        self.optimizer.zero_grad()
        loss = self.func(self.params['x'])
        loss.backward()
        self.optimizer.step()
        return self.params['x'].item(), loss.item()

def function(x):
    return (x**6 / 12) - (9 * x**5 / 10) + (13 * x**4 / 4) - 4 * x**3 + 4.5

def generate_animation(learning_rate):
    optimizer = AdamOptimizer(function, learning_rate=learning_rate, x_start=-0.6)
    fig, ax, line = init_plot()
    anim = FuncAnimation(fig, animate, fargs=(optimizer, line), frames=400, interval=100, blit=True)
    animation_name = f'optimization_animation_lr_{learning_rate}.mp4'
    anim.save(animation_name, writer='ffmpeg', fps=30)
    return optimizer, animation_name

def init_plot():
    fig, ax = plt.subplots()
    x_np_values = np.linspace(-5, 5, 400)
    y_np_values = function(x_np_values)
    ax.plot(x_np_values, y_np_values, label='Function')
    line, = ax.plot([], [], 'ro', label='Current Position')
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 10)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimization Path using Adam in PyTorch')
    plt.legend()
    return fig, ax, line

def animate(i, optimizer, line):
    x, z = optimizer.step()
    line.set_data([x], [function(x)])
    return [line]  # Return a list of Artist objects

def main():
    # Define a range of learning rates
    learning_rates = np.arange(0.01, 1.01, 0.01)

    for learning_rate in learning_rates:
        optimizer, animation_name = generate_animation(learning_rate)
        final_x, final_value = optimizer.step()

if __name__ == '__main__':
    main()
