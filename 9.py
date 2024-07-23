import numpy as np
from ipywidgets import interact
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.io import push_notebook



def local_regression(x0, X, Y, tau):

    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]
  
    xw = X.T * radial_kernel(x0, X, tau)
    beta = np.linalg.pinv(xw @ X) @ xw @ Y

    return x0 @ beta

def radial_kernel(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))

n = 1000

X = np.linspace(-3, 3, num=n)
Y = np.log(np.abs(X ** 2 - 1) + .5)
# Jitter X
X += np.random.normal(scale=.1, size=n)

def plot_lwr(tau):
 
    domain = np.linspace(-3, 3, num=300)
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plot = figure(width=400, height=400)
    plot.title.text = 'tau=%g' % tau
    plot.scatter(X, Y, alpha=.3)
    plot.line(domain, prediction, line_width=2, color='red')
    return plot


p1 = plot_lwr(10.)
p2 = plot_lwr(1.)
p3 = plot_lwr(0.1)
p4 = plot_lwr(0.01)

show(gridplot([[p1, p2], [p3, p4]]))


def interactive_update(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    model.data_source.data['y'] = prediction
    push_notebook()

domain = np.linspace(-3, 3, num=100)
prediction = [local_regression(x0, X, Y, 1.) for x0 in domain]
plot = figure()
plot.scatter(X, Y, alpha=.3)
model = plot.line(domain, prediction, line_width=2, color='red')

handle = show(plot, notebook_handle=True)
interact(interactive_update, tau=(0.01, 3., 0.01))
