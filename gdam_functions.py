"""Functions used in Exercise 8 of Geol 197 GDAM"""

"""
Collection of functons to calculate the statistics asked.

Authors:
    Kirk Ramos
    Paula Jatulan
    26.10.2024
"""

# Import any modules needed in your functions here
import math as m
import numpy as np

# Define your new functions below
def mean(x):
    """
    Calculates the average of volcanoes' elevation values.
    
    Parameter
    ---------
    x: <numerical>
        Elevation of a volcano in m.
    
    Returns
    -------
    <numerical>
        Calculated average elevation.
    """
    
    return (sum(x)/len(x))
    
def std_dev(x):
    """
    Calculates the standard deviation of volcanoes' elevation.
    
    Parameter
    ---------
    x: <numerical>
        Elevation of a volcano in m.
    
    Returns
    -------
    <numerical>
        Calculated standard deviation.
    """
    
    return (np.sqrt((sum((j - mean(x)) ** 2 for j in x)) / (len(x)-1)))
    
def std_err(x):
    """
    Calculates the standard error of a statistic.
    
    Parameter
    ---------
    x: <numerical>
        Elevation of a volcano in m.
    
    Returns
    -------
    <numerical>
        Calculated standard error.
    """    
    
    return (std_dev(x) / np.sqrt(len(x)))

def gaussian(m, sd, x):
    """
    Calculates the Gaussian value for a given x.
    
    Parameters
    ----------
    m: <numerical>
        Mean elevation of volcanoes per region
    sd: <numerical>
        Standard deviation of the elevation values of volcanoes per 
    x: <numerical>
        Elevation of volcanoes

    Returns
    -------
    <numerical>
        Calculated Gaussian value
    """
    
    if isinstance(x, list):
        return [gaussian(m, sd, i) for i in x]
    else:
        return (1 / (sd * np.sqrt(2 * np.pi))) * (np.exp(-((x - m) ** 2) / (2 * sd ** 2)))

def linregress(x, y):
    """
    Returns the slope and y-intercept for a regression line for data x and y.

    Parameters
    ----------
    x: <array-like>
        Independent variable data
    y: <numerical>
        Dependent variable data

    Returns
    -------
    Calculated slope and y-intercept of the regression line.
    """

    # Count the number of data points
    n = len(x)

    # Calculate the sums needed for the equations
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(xi ** 2 for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    # Calculate the delta
    delta = n * sum_x2 - sum_x ** 2

    # Calculate B (slope)
    B = (n * sum_xy - sum_x * sum_y) / delta

    # Calculate A (y-intercept)
    A = (sum_x2 * sum_y - sum_x * sum_xy) / delta

    # Return both A (y-intercept) and B (slope)
    return A, B

def pearson(x, y):
    """
    Calculates the correlation coefficient (r).

    Parameters
    ----------
    x: <array-like>
        Independent variable data
    y: <array-like>
        Dependent variable data

    Returns
    -------
    Calculated correlation coefficient (r).
    """

    # Convert x and y into numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Calculate the means of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate the numerator
    numerator = sum((x - mean_x) * (y - mean_y))

    # Calculate the denominator
    denominator = np.sqrt(sum((x - mean_x) ** 2) * sum((y - mean_y) ** 2))

    # Return the Pearson correlation coefficient
    return numerator / denominator

def chi_squared(o,e,s):
    """
    Calculates the goodness-of-fit using the reduced chi-squared function.

    Parameters
    ----------
    o: <array-like>
        Observed values
    e: <array-like>
        Expected values
    s: <array-like>
        Standard deviation of each observed value

    Returns
    -------
    Calculated goodness-of-fit.
    """

    # Convert o, e, and s into numpy arrays
    o = np.array(o)
    e = np.array(e)
    s = np.array(s)

    # Count the number of data points
    n = len(o)

    # Calculate the numerator
    numerator = (o - e) ** 2

    # Calculate the denominator
    denominator = s ** 2

    return (1/n) * sum(numerator / denominator)