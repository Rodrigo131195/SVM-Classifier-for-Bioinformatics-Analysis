from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the training data from file
data = np.loadtxt('Train_call.txt', skiprows=1)

# Load the target labels from file as strings
with open('Train_clinical.txt', 'r') as file:
    target_lines = file.readlines()[1:]  # Skip the first row (header)
    y_str = [line.split()[1] for line in target_lines]

# Convert string labels to integer labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_str)

# Try different values of k
k_range = range(1, 21x)
sse = []

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)  # Inertia is the sum of squared distances to the nearest centroid

# Plot SSE against k
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method')
plt.show()
