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

def list_combinations(list_a, list_b):
    combinations = [(b, a) for a in list_a for b in list_b]
    return combinations

def H(theta, k=0.1, theta_0=45):
    return 1 / (1 + np.exp(k * (theta - theta_0)))


def compute_my_metric(data_dict):
    if len(data_dict.keys()) > 1:
        # Compute histogram
        segments = data_dict.keys()
        combinations = get_unique_combinations(segments, 2)

        pair_metric = []
        pair_weights = []

        for pair in combinations:
            pair0 = pair[0]
            pair1 = pair[1]
            counts0, bin_edges0 = np.histogram(data_dict[pair0][0], bins=180, range=(-90, 90))
            counts1, bin_edges1 = np.histogram(data_dict[pair1][0], bins=180, range=(-90, 90))

            # Calculate bin midpoints
            bin_midpoints0 = (bin_edges0[:-1] + bin_edges0[1:]) / 2
            hist_dict0 = dict(zip(bin_midpoints0, counts0))
            sorted_items0 = sorted(hist_dict0.items(), key=lambda item: item[1], reverse=True)
            num_to_retain0 = max(1, len(sorted_items0) * 40 // 100)
            hist_dict0 = dict(sorted_items0[:num_to_retain0])

            bin_midpoints1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
            hist_dict1 = dict(zip(bin_midpoints1, counts1))
            sorted_items1 = sorted(hist_dict1.items(), key=lambda item: item[1], reverse=True)
            num_to_retain1 = max(1, len(sorted_items1) * 40 // 100)
            hist_dict1 = dict(sorted_items1[:num_to_retain1])

            pixel_combinations = list_combinations(hist_dict0.keys(), hist_dict1.keys())
            metric_list = []
            weight_list = []

            for combo in pixel_combinations:
                angle1 = combo[0]
                angle2 = combo[1]
                A = max(angle1, angle2)
                B = min(angle1, angle2)
                diff = min((180 - (A - B)), A - B)
                
                # Check if angle1 and angle2 are in hist_dict0 and hist_dict1 respectively
                if angle1 in hist_dict0 and angle2 in hist_dict1:
                    weight = hist_dict0[angle1] * hist_dict1[angle2]
                    metric = H(diff)
                    weight_list.append(weight)
                    metric_list.append(metric)
                else:
                    # Handle the case where angle1 or angle2 is not in the dictionary
                    weight_list.append(0)
                    metric_list.append(0)

            # Calculate the weighted sum
            pair_sum = sum(w * m for w, m in zip(weight_list, metric_list))

            # Calculate the sum of weights
            sum_weights = sum(weight_list)

            # Calculate the weighted average
            if sum_weights != 0:
                pair_metric.append(pair_sum / sum_weights)
            else:
                pair_metric.append(0)
            
            pair_weights.append(data_dict[pair0][1] * data_dict[pair1][1])

        # Calculate the weighted sum
        total_sum = sum(w * m for w, m in zip(pair_metric, pair_weights))

        # Calculate the sum of weights
        sum_weights = sum(pair_weights)
        
        if sum_weights != 0:
            return 10*(total_sum / sum_weights)-1
        else:
            return 0
    else: 
        return 0






def alignment_metric(data_in_hexagon):

    angles_in_radians = np.deg2rad(data_in_hexagon)

    # Calculate the components of the order parameter
    
    # Calculate the components of the order parameter
    cos_sum = np.sum(np.cos(2*angles_in_radians))
    sin_sum = np.sum(np.sin(2*angles_in_radians))

    # Calculate the magnitude of the order parameter
    order_parameter = np.sqrt(cos_sum**2 + sin_sum**2) / len(data_in_hexagon)

    return order_parameter







#____________________________________________________________________________________________________________________________________________________________________
if __name__ == "__main__":

    # Prepare x and y data
    y = []
    x = []
    for i in range(4, 180):
        # Generate data
        data = np.random.uniform(-90, -90+i, 100000)
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
