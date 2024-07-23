# Helper functions
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math 

def get_unique_combinations(s, num):
    # Generate all combinations of length 'num'
    all_combs = itertools.combinations(s, num)
    # Convert each combination to a sorted tuple to remove duplicates
    unique_combs = set(tuple(sorted(c)) for c in all_combs)
    # Convert each sorted tuple back to a list
    return [list(c) for c in unique_combs]

def H(theta, k=0.1, theta_0=45):
    return 1 / (1 + np.exp(k * (theta - theta_0)))

def sigmoid(x, k=.1, x_0=0.5):

    return 1 / (1 + np.exp(-k * (x - x_0)))


def compute_my_metric(data):
    # Compute histogram
    counts, bin_edges = np.histogram(data, bins=180, range=(-90, 90))

    # Calculate bin midpoints
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter to get non-zero counts
    non_zero_bins = counts > 0

    # Ensure non-zero bins mask matches the length of bin_midpoints
    filtered_bin_midpoints = bin_midpoints[non_zero_bins]
    counts = counts[non_zero_bins]

    hist_dict = dict(zip(filtered_bin_midpoints, counts))
    num = 2
    unique_perms = get_unique_combinations(filtered_bin_midpoints, num)
    metric_list = []
    weight_list = []

    for perm in unique_perms:
        angle1 = perm[0]
        angle2 = perm[1]
        A = max(angle1, angle2)
        B = min(angle1, angle2)
        diff = min((180 - (A - B)), A - B)
        weight_list.append((hist_dict[angle1] * hist_dict[angle2]))
        metric_list.append((90-diff)/90)
    metric_list = ((np.array(weight_list)/max(weight_list)) * np.array(metric_list))
    return np.mean(metric_list)

def alignment_metric(data_in_hexagon):

    angles_in_radians = np.deg2rad(data_in_hexagon)

    # Calculate the components of the order parameter
    
    # Calculate the components of the order parameter
    cos_sum = np.sum(np.cos(2*angles_in_radians))
    sin_sum = np.sum(np.sin(2*angles_in_radians))

    # Calculate the magnitude of the order parameter
    order_parameter = np.sqrt(cos_sum**2 + sin_sum**2) / len(data_in_hexagon)

    return order_parameter


if __name__ == "__main__":

    # Prepare x and y data
    y = []
    x = []
    for i in range(4, 180):
        # Generate data
        data = np.random.uniform(-90, -90+i, 5000)
        val = float(alignment_metric(data))
        y.append(val)
        x.append(i)

    x1 = []
    y1 = []
    for j in range(1, 90):
        # Generate data
        data1 = [-j, j] * 10000
        val = float(alignment_metric(data1))
        y1.append(val)
        x1.append(j)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the histogram
    ax1.hist(data, bins=180, range=(-90, 90), edgecolor='black')
    ax1.set_xlabel('Theta (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of (-90,90) with Every Degree as a Bin', fontsize=10)  # Smaller title font
    ax1.grid(True)

    # Plot x vs. y
    ax2.scatter(x, y, marker='o', color='b', label='Metric Data')
    ax2.set_xlabel('i (degrees)')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Metric as a Function of a uniform distribution of agles between (-90, -90 + i) Degrees', fontsize=10)  # Smaller title font
    ax2.grid(True)
    ax2.legend()

    # Plot x vs. y
    ax3.scatter(x1, y1, marker='o', color='b', label='Metric Data')
    ax3.set_xlabel('i (degrees)')
    ax3.set_ylabel('Metric Value')
    ax3.set_title('Metric as a Function of Data with only Angles [-i, i]', fontsize=10)  # Smaller title font
    ax3.grid(True)
    ax3.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()



    # # Create the figure and axes
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

    # # Plot the transfer function H(theta)
    # theta_values = np.linspace(0, 90, 100)
    # H_values = H(theta_values)
    # ax1.plot(theta_values, H_values, color='b')
    # ax1.set_xlabel('Theta (degrees)')
    # ax1.set_ylabel('H(Theta)')
    # ax1.set_title('Sigmoid Transfer Function H(Theta) for angle differences', fontsize=10)
    # ax1.grid(True)

    # # Plot the sigmoid function
    # x_values = np.linspace(0, 1, 100)
    # y_values = sigmoid(x_values, k=10, x_0=0.5)
    # ax2.plot(x_values, y_values, color='r')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('Sigmoid(x)')
    # ax2.set_title('Sigmoid Function for Weights', fontsize=10)
    # ax2.grid(True)

    # # Adjust layout
    # plt.tight_layout()
    # plt.show()
