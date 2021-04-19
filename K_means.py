# K-Means Algorithm

# Imports
import math
import random                
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Settings           
pd.options.display.max_rows = 500
pd.options.display.max_columns = 10

# Read the csv file, display the dataframe, return a tuple of data
def read_csv_pd(filename):
    data_frame = pd.read_csv(filename, delimiter = ',')                                 # Read from the .csv to create a data frame
    print(data_frame)
    print("")
    countries = data_frame[["Countries"]].values                                        # Use data frame to create an array of shape (392,1)
    birth_life = data_frame[[data_frame.columns[1], data_frame.columns[2]]].values      # Use data frame to create an array of shape (392,2)
    return birth_life, countries

# Calculate the distance between each cluster centroid & each datapoint, append distances to a list
def euclidean_distance(cent, datapoints):
    distance_list = []
    for centroid in cent:
        for datapoint in datapoints:
            distance_list.append(math.sqrt(math.pow(datapoint[0] - centroid[0], 2) + math.pow(datapoint[1] - centroid[1], 2)))
    return distance_list

# Step 2: Assign each datapoint to the nearest cluster
def assign_to_cluster(blc, centroids_in, n_clusters):
    # Calculate the distances between each datapoint and each centroid, reshape the list data
    distance_reshaped = np.reshape(euclidean_distance(centroids_in, blc[0]), (len(centroids_in), len(blc[0])))

    # Find the shortest distance of a datapoint to each centroid. The index tells us to which cluster the datapoint belongs
    belongs_to_cluster = []
    distances_shortest = []
    for column in zip(*distance_reshaped):
        distances_shortest.append(min(column))
        belongs_to_cluster.append(np.argmin(column) + 1)

    # Assign each datapoint to the nearest cluster using a dictionary 
    clusters = {}
    for i in range(0, n_clusters):
        clusters[i + 1] = []
    for d_point, clus in zip(blc[0], belongs_to_cluster):
        clusters[clus].append(d_point)
    return clusters, belongs_to_cluster
    
# Step 3: Calculate the new cluster mean (centroids)
def mean_centroid(clusters_in):
    for i, cluster in enumerate(clusters_in):
        reshaped = np.reshape(clusters_in[cluster], (len(clusters_in[cluster]), 2))
        print("Cluster " + str(cluster) + " datapoints of shape: " + str(reshaped.shape))
        print(reshaped)
        print("")
        centroids[i][0] = sum(reshaped[:,0])/len(reshaped[:,0])
        centroids[i][1] = sum(reshaped[:,1])/len(reshaped[:,1])
    print("The new centroids calculated are: " + str(centroids))
    print("")


# Algorithm starts

# Read the csv file
#blc_tuple = read_csv_pd("data1953.csv")                     # The value returned is a tuple
#blc_tuple = read_csv_pd("data2008.csv")
blc_tuple = read_csv_pd("dataBoth.csv")

# Get the number of clusters & number of iterations 
k = int(input("Please enter the number of clusters: "))
iterations = int(input("Please enter the number of iterations that the algorithm must run: "))
print("")

# Access the tuple & create a list of lists
birth_life_list = np.ndarray.tolist(blc_tuple[0][0:, :])    # The value returned is a list of lists

# Step 1: Randomly select k amount of centroids
centroids = random.sample(birth_life_list, k)
print("The initial random centroids are:")
print(centroids)
print("")

# Main loop
for iteration in range(0, iterations):
    print("Iteration: " + str(iteration + 1))
    print("")
    
    assign = assign_to_cluster(blc_tuple, centroids, k)
    mean_centroid(assign[0])

    # Create a new data frame
    cluster_data = pd.DataFrame({'Birth Rate': blc_tuple[0][:,0], 'Life Expectancy': blc_tuple[0][:,1], 'Country': blc_tuple[1][:,0], 'Cluster': assign[1]})
    print("Cluster Data Frame:")
    print(cluster_data)
    print("")

    # Additional : Manipulating the data frame
    group_by_cluster = cluster_data[['Birth Rate', 'Life Expectancy', 'Country', 'Cluster']].groupby('Cluster')
    print("List of countries belonging to each cluster for iteration: " + str(iteration + 1))
    print(list(group_by_cluster))
    print("")
    count_clusters = group_by_cluster.count()
    print("Number of countries belonging to each cluster for iteration: " + str(iteration + 1))
    print(count_clusters)
    print("")
    print("Averages of Birth Rates and Life Expectancies for each cluster for iteration: " + str(iteration + 1))
    print(cluster_data.groupby(['Cluster']).mean())
    print("")

#########

# Objective Function (Sanity Check) checks for convergence using the sum of distances

    # Additional : Assign the cluster dictionary to cluster_dict
    cluster_dict = assign[0]

    # Additional : Dictionary for distances from each point to the closest center
    distance_dict = {}
    for i in range(0, k):
        distance_dict[i + 1] = []

    # Additional : cluster_dict contains the datapoints belonging to each cluster
    for index, key in enumerate(cluster_dict):
        cluster_points = np.array(cluster_dict[key])
        cluster_points = np.reshape(cluster_points, (len(cluster_points), 2))

        # Use cluster_points to calculate the mean
        birth_rate_avg = sum(cluster_points[:,0])/len(cluster_points[:,0])
        life_exp_avg = sum(cluster_points[:,1])/len(cluster_points[:,1])

        # Use cluster_points and averages to populate the distance_dict
        for dp in cluster_points:
            distance = math.sqrt(math.pow(birth_rate_avg - dp[0], 2) + math.pow(life_exp_avg - dp[1], 2))
            distance_dict[index + 1].append(distance)

    # Additional : Calculate the total distance in each cluster
    total_distance = []
    for s in distance_dict:
        total_distance.append(sum(distance_dict[s]))

    print("The total distance of each clusters respectively: {}".format(total_distance))
    print("The total distance of all clusters: {}".format(sum(total_distance)))
    print("")

##########
        
    # Reshape the centroids list
    centr = np.reshape(centroids, (k, 2))

    # Plot the data
    sns.lmplot(data = cluster_data, x = 'Birth Rate', y = 'Life Expectancy', hue = 'Cluster', fit_reg = False, legend = False, legend_out = False)
    plt.plot(centr[:,0], centr[:,1], c = 'black', marker = '*', markersize = 15, linestyle = None, linewidth = 0)
    plt.legend(loc = 'upper right')
    plt.title('Iteration: ' + str(iteration + 1) + "\nTotal distance of all clusters: " + str(round(sum(total_distance), 0)))
    
plt.show()  
