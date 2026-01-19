import sys
import os
import json
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../PYTHON_MOTION_PLANNING-MASTER/")


from python_motion_planning import *
import numpy as np
import matplotlib.pyplot as plt

def curvature(path):
    """
    Calculate the curvature of a 2D path defined by x and y coordinates.
    
    Parameters:
        x (array-like): Array of x-coordinates of the path.
        y (array-like): Array of y-coordinates of the path.
        
    Returns:
        curvature (numpy.ndarray): Array containing curvature values corresponding to each point on the path.
    """
    x,y = np.array([x for (x,y) in path]), np.array([y for (x,y) in path])
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**(3/2)

    avg_curvature = np.mean(curvature)
    max_curvature = np.max(curvature)
    
    return avg_curvature, max_curvature

def rotations(path):
    rotation = 0.0
    for i in range(1, len(path)-1):
        vector1 = path[i] - path[i - 1]
        vector2 = path[i + 1] - path[i] 
        cos_angle = max(min(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)), 1), -1)
        angle = np.arccos(cos_angle)
        # print(min(np.abs(np.degrees(angle)), 180 - np.abs(np.degrees(angle))))
        rotation += min(np.abs(np.degrees(angle)), 180 - np.abs(np.degrees(angle)))

    return rotation

with open("config.json") as f:
    config = json.load(f)

configuration = config["Configuration"]
robot = config["Robot"]


map_filepath = configuration["map_filepath"]
algorithm = configuration["algorithm"]
processed = configuration["processed"]
if processed == "Y":
    processed = True
else:
    processed = False

rr = robot["robot_radius"]
params = {}
params["MAX_V"] = robot["maximum_linear_velocity"]
params["MAX_W"] = robot["maximum_angular_velocity"]

if algorithm in ["a_star", "dijkstra", "aco", "theta_star"]:
    map_instance = "G"
elif algorithm in ["rrt", "rrt_star", "prm"]:
    map_instance = "M"
else:
    print("ENTER CORRECT CODE FOR THE ALGORITHM")

image = cv2.imread(map_filepath)
# image_loader = ImageProcessor(map_instance, rr, processed=processed)
# image_loader.load_image(map_filepath)
# start, goal, env, image = image_loader.process()
# image = image.astype(np.uint8)
# image = image.transpose()[::-1,:]
# height, width = image.shape
# rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
# rgb_image[:, :, 0] = image  
# rgb_image[:, :, 1] = image  
# rgb_image[:, :, 2] = image


# rgb_image[50 - int(start[1]), int(start[0])] = np.array([0,0,255])
# rgb_image[50 - int(goal[1]), int(goal[0])] = np.array([204, 85, 17])

# cv2.imwrite('Processed_image.png', rgb_image)
import numpy as np
import json

# Open and read the JSON file
with open('schedule.json', 'r') as file:
    data = json.load(file)
names = ['agent1', 'agent2', 'agent3',]

states = []
for name in names:
    state =[]
    ss =[]
    for a in  data['schedule'][name]:
        s = a['t'],a['x'],a['y']
        ss.append(s)

        # if len(ss) == 0 or (a['x'] != ss[-1][1] or a['y'] != ss[-1][2]):
        #     ss.append(s)
        # else:
        #     if len(ss) > 1:
        #         state.append(np.array(ss))
        #     ss = []
    state.append(ss)
    states.append(state)
    # states.append(np.array(state))
    # state2 = []
    # for ii,s in enumerate(state):
    #     state2.append(s)
    #     if ii+1 < len(state):
    #         ss = (s[0],(state[ii][1]+state[ii+1][1])/2,(state[ii][2]+state[ii+1][2])/2)
    #         state2.append(ss)
    # states.append(np.array(state2))

def moving_average(noisy_points, width):
    smoothed_points = np.zeros_like(noisy_points)
    weights = np.ones(width)/width

    for dim in range(noisy_points.shape[1]):
        column_data = noisy_points[:, dim]
        smoothed_points[:, dim] = np.convolve(column_data, 
                                              weights, 
                                              mode='same')

    return smoothed_points

for state in states[1]:
    # path = [(pos[1],pos[2]) for pos in moving_average(state,min(len(state),4))]
    path = [(pos[1],pos[2]) for pos in state]

    start = (path[0][0], path[0][1], 0)
    goal = (path[-1][0], path[-1][1], 0)
    env = Grid(150, 80,1,)


    # ------------- planner-------------
    plt = PID(start=start, goal=goal, env=env, algorithm=algorithm, params=params)
    plt.path  = path
    try:
        path, cost = plt.run()
        np.savetxt('path.txt', path, fmt="%f", delimiter=",")

    except:
        print("Cannot plan path/No path found. \nEither the corridors are too narrow for the robot given the robot radius plus the error margin, or the goal and start are separated by an obstacle, and can't be reached.")
        exit(0)

    out_params = {}
    out_params["Algorithm"] = algorithm
    out_params["Distance travelled by the robot"] = cost
    out_params["Time taken (in sim time)"] = plt.tottime
    out_params["Total angle rotates (in degrees)"] = rotations(path)

    vel = []
    ang_vel = []
    for i in plt.vel_data:
        vel.append(i[0][0])
        ang_vel.append(i[1][0])
    vel = np.array(vel)[1:len(vel) - 1]
    acc = np.gradient(vel)

    ang_vel = np.array(ang_vel)[1:len(vel) - 1]
    ang_acc = np.gradient(ang_vel)

    out_params["Maximum linear acceleration"] = np.max(np.abs(acc))
    out_params["Maximum angular acceleration"] = np.max(np.abs(ang_acc))
    out_params["Mean curvature of the path (in m^-1)"] = curvature(path)[0]

    # print(curvature(path))
    
    
    print("Distance travelled by the robot: " + str(cost))
    print("Time taken (in sim time): " + str(plt.tottime))
    print("Maximum linear acceleration: " + str(np.max(np.abs(acc))))
    print("Maximum angular acceleration: " + str(np.max(np.abs(ang_acc))))
    print("Mean curvature of the path (in m^-1): " + str(curvature(path)[0]))
    print("Total angle rotates (in degrees): " + str(rotations(path)))


# with open("results.json", "w") as f:
#     json.dump(out_params, f, indent = 4)
