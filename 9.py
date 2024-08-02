#Program: Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.

import numpy as np
from ipywidgets import interact
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.io import push_notebook

def local_regression(x0, X, Y, tau):
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]
    xw = X.T * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))
    beta = np.linalg.pinv(xw @ X) @ xw @ Y
    return x0 @ beta

n = 1000
X = np.linspace(-3, 3, num=n)
Y = np.log(np.abs(X ** 2 - 1) + .5) + np.random.normal(scale=.1, size=n)

def plot_lwr(tau):
    domain = np.linspace(-3, 3, num=300)
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plot = figure(width=400, height=400, title=f'tau={tau}')
    plot.scatter(X, Y, alpha=.3)
    plot.line(domain, prediction, line_width=2, color='red')
    return plot

show(gridplot([[plot_lwr(10.), plot_lwr(1.)], [plot_lwr(0.1), plot_lwr(0.01)]]))

domain = np.linspace(-3, 3, num=100)
def interactive_update(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    model.data_source.data['y'] = prediction
    push_notebook()

plot = figure()
plot.scatter(X, Y, alpha=.3)
model = plot.line(domain, [local_regression(x0, X, Y, 1.) for x0 in domain], line_width=2, color='red')
handle = show(plot, notebook_handle=True)

interact(interactive_update, tau=(0.01, 3., 0.01))
