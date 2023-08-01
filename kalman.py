import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
dt = .1 # Time step
df=pd.read_csv('kalmann.txt',delimiter=',')
data_points=df.to_numpy()
var_x=df['x'].var()
var_y=df['y'].var()
print(var_x)
print(var_y)

# Define the state transition matrix A
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Define the measurement matrix H
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
print(H.shape)
# Define the process noise covariance matrix Q
Q = np.array([[var_y, 0, 0, 0],
              [0, var_y, 0, 0],
              [0, 0, var_y, 0],
              [0, 0, 0, var_y]])
 

# Define the measurement noise covariance matrix R
R = np.array([[var_x/30, 0],
              [0, var_y/30]])
print(R)
print(Q)
# Initialize the state vector [x, y, vx, vy]
state = np.array([[3.686804471625727e-06],
                  [372.99815102559614],
                  [0],
                  [0]])

# Initialize the covariance matrix P
P = np.eye(4)
print(P)
#Initialize the error in
W=np.zeros(4)

print(W)
# Create an empty list to store the estimated positions
estimated_positions = []

for i in range(len(data_points)):

    
    #Get the measurement from the data points
    measurement = np.array([[data_points[i][0]],
                            [data_points[i][1]]])
    velocity = np.array([[data_points[i][2]],
                         [data_points[i][3]]])

    #Update the state transition matrix A with velocity information
    A[0, 2] = dt * velocity[0]
    A[1, 3] = dt * velocity[1]
    # Predict
    state = np.dot(A, state)+W
    P = np.dot(A, np.dot(P, A.T)) + Q

    # Update
    y = measurement - np.dot(H, state)
    S = np.dot(H, np.dot(P, H.T)) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))

    state += np.dot(K, y)
    P = np.dot((np.eye(4) - np.dot(K, H)), P)

    # Store the estimated position
    estimated_positions.append((state[0][0], state[1][0]))
    print("Iteraion :: "+str(i))
    print("Estimated Position")
    print(state[0][0], state[1][0])
    print("Error")
    print(P)


    
    
estimated_positions=np.array(estimated_positions)
plt.plot(data_points[:,0],data_points[:,1],label="measured")
plt.plot(estimated_positions[:,0],estimated_positions[:,1],label="predicted")
plt.legend()
plt.show()
