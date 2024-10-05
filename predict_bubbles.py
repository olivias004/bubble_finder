import numpy as np
import matplotlib.pyplot as plt

def predict_new_bubble_positions(transformed_bubbles, stellar_data, time_interval=10):
    """
    Predicts the new positions of the bubbles based on stellar velocities from the TIPSY data.
    
    Parameters:
    transformed_bubbles (list of arrays): Each array is a cluster of physical positions (x, y, z).
    stellar_data (ndarray): Stellar data containing velocity information from TIPSY.
    time_interval (int): Time interval in Myr for the prediction (default is 10 Myr).
    
    Returns:
    predicted_positions (list): List of new bubble positions after applying the stellar velocity influence.
    """
    # Extract the stellar velocities (assuming they are in 'velocity' key)
    stellar_velocities = np.asarray(stellar_data['vel'])
    
    # Calculate the average velocity of stars, assuming this influences bubble movement
    mean_velocity = np.mean(stellar_velocities, axis=0)  # (vx, vy, vz)
    
    print(f"Mean velocity: {mean_velocity}")
    
    predicted_positions = []
    
    for cluster in transformed_bubbles:
        # Predict the new positions by adding the velocity influence
        shifted_positions = cluster + mean_velocity * time_interval
        predicted_positions.append(shifted_positions)
    
    return predicted_positions

def visualize_bubbles_projection(predicted_bubble_positions, grid_size=(200, 200)):
    """
    Projects the predicted bubble positions onto a 2D x-y plane and visualizes them with random colors.
    
    Parameters:
    predicted_bubble_positions (list of arrays): List of predicted positions of each bubble cluster.
    grid_size (tuple): Size of the projection grid (for visualization purposes).
    """
    x_size, y_size = grid_size
    
    # Create a 2D projection (x-y plane)
    projection = np.zeros((x_size, y_size), dtype=int)

    # Create random colors for each bubble
    np.random.seed(42)
    colors = np.random.randint(1, len(predicted_bubble_positions) + 1, size=len(predicted_bubble_positions))

    # Automatically scale positions to the grid without limits
    all_x_positions = np.concatenate([cluster[:, 0] for cluster in predicted_bubble_positions])
    all_y_positions = np.concatenate([cluster[:, 1] for cluster in predicted_bubble_positions])
    
    # Find the min and max of x and y positions to scale them to grid indices
    x_min, x_max = np.min(all_x_positions), np.max(all_x_positions)
    y_min, y_max = np.min(all_y_positions), np.max(all_y_positions)

    for i, cluster in enumerate(predicted_bubble_positions):
        for pos in cluster:
            x_physical, y_physical = pos[0], pos[1]
            
            # Convert physical coordinates back to grid indices based on min/max scaling
            x_idx = int((x_physical - x_min) / (x_max - x_min) * (x_size - 1))
            y_idx = int((y_physical - y_min) / (y_max - y_min) * (y_size - 1))
            
            # Ensure indices are within bounds
            if 0 <= x_idx < x_size and 0 <= y_idx < y_size:
                projection[x_idx, y_idx] = colors[i]

    # Visualize the 2D projection
    plt.figure(figsize=(10, 10))
    cmap = plt.cm.get_cmap('tab20', np.max(projection) + 1)
    plt.imshow(projection.T, cmap=cmap, origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.title("Predicted 2D Projection of Bubble Clusters")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Example usage:
# Assuming transformed_bubbles is a list of bubble clusters that are already in physical scale
# and s.star contains stellar data with velocity information
predicted_bubbles = predict_new_bubble_positions(transformed_bubbles, s.star)
visualize_bubbles_projection(predicted_bubbles)














import numpy as np
import matplotlib.pyplot as plt

def predict_new_bubble_positions(transformed_bubbles, stellar_data, time_interval=10, supernova_threshold=300, merge_distance=1.0, disappearance_prob=0.05):
    """
    Predicts the new positions of the bubbles based on stellar velocities from the TIPSY data.
    Takes into account bubble formation from supernovae, merging of close bubbles, and bubble disappearance.
    
    Parameters:
    transformed_bubbles (list of arrays): Each array is a cluster of physical positions (x, y, z).
    stellar_data (ndarray): Stellar data containing velocity information from TIPSY.
    time_interval (int): Time interval in Myr for the prediction (default is 10 Myr).
    supernova_threshold (float): Velocity threshold to consider for supernova-induced bubble formation.
    merge_distance (float): Maximum distance between bubble centers to trigger a merge.
    disappearance_prob (float): Probability that a bubble may disappear.
    
    Returns:
    predicted_positions (list): List of new bubble positions after applying the stellar velocity influence.
    """
    # Extract the stellar velocities
    stellar_velocities = np.asarray(stellar_data['vel'])
    
    # Calculate the average velocity of stars, assuming this influences bubble movement
    mean_velocity = np.mean(stellar_velocities, axis=0)  # (vx, vy, vz)
    
    predicted_positions = []

    # Step 1: Predict the new positions of existing bubbles
    for cluster in transformed_bubbles:
        # Predict the new positions by adding the velocity influence
        shifted_positions = cluster + mean_velocity * time_interval
        predicted_positions.append(shifted_positions)
    
    # Step 2: Introduce new bubbles based on supernova events
    supernovae = stellar_velocities[np.linalg.norm(stellar_velocities, axis=1) > supernova_threshold]
    for sn_velocity in supernovae:
        new_bubble_position = np.mean(stellar_data['pos'], axis=0) + sn_velocity * time_interval
        # Create a small bubble around the supernova event
        new_bubble = np.random.normal(loc=new_bubble_position, scale=0.1, size=(20, 3))  # Random small bubble
        predicted_positions.append(new_bubble)
    
    # Step 3: Handle bubble merging based on distance between bubble centers
    bubble_centers = np.array([np.mean(cluster, axis=0) for cluster in predicted_positions])
    merged_positions = []
    merged = np.zeros(len(predicted_positions), dtype=bool)  # Track which bubbles have merged
    
    for i, center in enumerate(bubble_centers):
        if merged[i]:
            continue
        to_merge = [i]
        for j, other_center in enumerate(bubble_centers[i+1:], start=i+1):
            if np.linalg.norm(center - other_center) < merge_distance:
                to_merge.append(j)
                merged[j] = True
        # Merge all bubbles that are close together
        merged_bubble = np.concatenate([predicted_positions[idx] for idx in to_merge], axis=0)
        merged_positions.append(merged_bubble)
    
    # Step 4: Apply bubble disappearance probability
    final_positions = []
    for bubble in merged_positions:
        if np.random.rand() > disappearance_prob:  # Keep the bubble
            final_positions.append(bubble)
    
    return final_positions

def visualize_bubbles_projection(predicted_bubble_positions, grid_size=(200, 200)):
    """
    Projects the predicted bubble positions onto a 2D x-y plane and visualizes them with random colors.
    
    Parameters:
    predicted_bubble_positions (list of arrays): List of predicted positions of each bubble cluster.
    grid_size (tuple): Size of the projection grid (for visualization purposes).
    """
    x_size, y_size = grid_size
    
    # Create a 2D projection (x-y plane)
    projection = np.zeros((x_size, y_size), dtype=int)

    # Create random colors for each bubble
    np.random.seed(42)
    colors = np.random.randint(1, len(predicted_bubble_positions) + 1, size=len(predicted_bubble_positions))

    # Automatically scale positions to the grid without limits
    all_x_positions = np.concatenate([cluster[:, 0] for cluster in predicted_bubble_positions])
    all_y_positions = np.concatenate([cluster[:, 1] for cluster in predicted_bubble_positions])
    
    # Find the min and max of x and y positions to scale them to grid indices
    x_min, x_max = np.min(all_x_positions), np.max(all_x_positions)
    y_min, y_max = np.min(all_y_positions), np.max(all_y_positions)

    for i, cluster in enumerate(predicted_bubble_positions):
        for pos in cluster:
            x_physical, y_physical = pos[0], pos[1]
            
            # Convert physical coordinates back to grid indices based on min/max scaling
            x_idx = int((x_physical - x_min) / (x_max - x_min) * (x_size - 1))
            y_idx = int((y_physical - y_min) / (y_max - y_min) * (y_size - 1))
            
            # Ensure indices are within bounds
            if 0 <= x_idx < x_size and 0 <= y_idx < y_size:
                projection[x_idx, y_idx] = colors[i]

    # Visualize the 2D projection
    plt.figure(figsize=(10, 10))
    cmap = plt.cm.get_cmap('tab20', np.max(projection) + 1)
    plt.imshow(projection.T, cmap=cmap, origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.title("Predicted 2D Projection of Bubble Clusters")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Example usage:
# Assuming transformed_bubbles is a list of bubble clusters that are already in physical scale
# and s.star contains stellar data with velocity information and supernova data can be derived
predicted_bubbles = predict_new_bubble_positions(transformed_bubbles, s.star)
visualize_bubbles_projection(predicted_bubbles)




