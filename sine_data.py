import numpy as np
from scipy.signal import argrelextrema


def csine(start, end, points=400):
    """ sin curve composition that starts at starts its period at `start` and
        ends it at `end`. One period each.
    """
    x = np.linspace(start, end, points)
    pi = np.pi
    curves = []
    curves.append( np.sin(x * 2*pi/(end/0.5)) )
    curves.append( np.sin(x * 2*pi/(end/1)) )
    curves.append( np.sin(x * 2*pi/(end/4)) )
    return (x, np.sum(curves, axis=0), curves)


def generate_y(sine_x):
    sine_extrema = [
        argrelextrema(sine_x[i], np.greater)[0] for i in range(len(sine_x))
    ]

    # y is the local global maximum so that we only have one (hopefully :))
    # we predict the index of the highest local maximum (0 to 399)
    sine_y = np.array([
        e[sine_x[i,e].argmax()] for i,e in enumerate(sine_extrema)
    ])

    return sine_y


def valid_dataset(points=100):
    sine_x = np.array([
        csine(55, 70, points=points)[1],
        csine(33, 55, points=points)[1],
    ])

    sine_y = generate_y(sine_x)
    sine_x = sine_x.reshape((*sine_x.shape, 1))

    return (sine_x, sine_y)


def train_dataset(points=100):
    sine_x = np.array([
        csine( 0, 20, points=points)[1],
        csine( 7, 14, points=points)[1],
        csine(12, 33, points=points)[1],
        csine(44, 70, points=points)[1],
        csine(74, 98, points=points)[1],
    ])

    sine_y = generate_y(sine_x)
    sine_x = sine_x.reshape((*sine_x.shape, 1))

    return (sine_x, sine_y)

