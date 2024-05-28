import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

parents = [-1,
            0, 1, 2,
            0, 4, 5,
            0, 7, 7,
            6,
            10, 11, 12, 13,
            10, 15, 16, 17,
            10, 19, 20, 21,
            10, 23, 24, 25,
            10, 27, 28, 29,
            3,
            31, 32, 33, 34,
            31, 36, 37, 38,
            31, 40, 41, 42,
            31, 44, 45, 46,
            31, 48, 49, 50]

joints = ['Neck',
            'RShoulder', 'RElbow', 'RWrist',
            'LShoulder', 'LElbow', 'LWrist',
            'Nose', 'REye', 'LEye',
            'LHandRoot',
            'LHandThumb1', 'LHandThumb2', 'LHandThumb3', 'LHandThumb4',
            'LHandIndex1', 'LHandIndex2', 'LHandIndex3', 'LHandIndex4',
            'LHandMiddle1', 'LHandMiddle2', 'LHandMiddle3', 'LHandMiddle4',
            'LHandRing1', 'LHandRing2', 'LHandRing3', 'LHandRing4',
            'LHandLittle1', 'LHandLittle2', 'LHandLittle3', 'LHandLittle4',
            'RHandRoot',
            'RHandThumb1', 'RHandThumb2', 'RHandThumb3', 'RHandThumb4',
            'RHandIndex1', 'RHandIndex2', 'RHandIndex3', 'RHandIndex4',
            'RHandMiddle1', 'RHandMiddle2', 'RHandMiddle3', 'RHandMiddle4',
            'RHandRing1', 'RHandRing2', 'RHandRing3', 'RHandRing4',
            'RHandLittle1', 'RHandLittle2', 'RHandLittle3', 'RHandLittle4'
    ]

def show_file_content(file_path):
    df = pd.read_hdf(file_path)
    print(df)

def inspect_h5(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        def print_attrs(name, obj):
            print(f"{name}: {obj}")
        
        h5_file.visititems(print_attrs)

def export_to_csv(file_path):
    with h5py.File(file_path, 'r') as f:
        pose_data = f['pose/data'][:]
        pose_normalize = f['pose/normalize'][:]
        pose_confidence = f['pose/confidence'][:]
        
    # Convert XY coordinates into dataframes
    pose_data_df = pd.DataFrame(pose_data)
    pose_normalize_df = pd.DataFrame(pose_normalize)
    pose_confidence_df = pd.DataFrame(pose_confidence)
    
    # Export dataframes to CSV files
    pose_data_df.to_csv('D:\\UoA\\SOFTENG 700A\\mix-stage-master\\src\\data\\visualization\\pose_data.csv', index=False)
    pose_normalize_df.to_csv('D:\\UoA\\SOFTENG 700A\\mix-stage-master\\src\\data\\visualization\\pose_normalize.csv', index=False)
    pose_confidence_df.to_csv('D:\\UoA\\SOFTENG 700A\\mix-stage-master\\src\\data\\visualization\\pose_confidence.csv', index=False)

def get_num_of_frame(file_path):
    with h5py.File(file_path, 'r') as f:
        pose_data = f['pose/normalize'][:]
    return len(pose_data)

def extract_coordinates(file_path, frame):
    with h5py.File(file_path, 'r') as f:
        pose_data = f['pose/normalize'][:]

    # Select frame 0 - 108
    # Each row represents a frame or a snapshot in that interval
    # Each pair of columns (i.e., columns 0 and 52, 1 and 53, 2 and 54, etc.) represents the X and Y coordinates of a specific keypoint
    new_data = pose_data[frame].reshape(2, 52)

    # Extract XY coordinates
    x_coords = new_data[0]
    y_coords = new_data[1]

    # Rescale neck coordinate to 0
    x_coords[0] = 0
    y_coords[0] = 0

    return x_coords, y_coords

def plot_keypoints(file_path):
    x_coords, y_coords = extract_coordinates(file_path, 0)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, color='r')

    for i, (joint, parent) in enumerate(zip(joints, parents)):
        if i == 0:
            plt.text(x_coords[i], y_coords[i], joint, fontsize=9, ha='right')
        if parent != -1:
            plt.plot([x_coords[parent], x_coords[i]], [y_coords[parent], y_coords[i]], 'b-')

    plt.gca().invert_yaxis()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Skeleton Key Points')
    plt.show()

def plot_animation(file_path):
    def update(frame):
        x_coords, y_coords = extract_coordinates(file_path, frame)
        plt.cla()
        plt.scatter(x_coords, y_coords, color='r')

        for i, (joint, parent) in enumerate(zip(joints, parents)):
            if i == 0:
                plt.text(x_coords[i], y_coords[i], joint, fontsize=9, ha='right')
            if parent != -1:
                plt.plot([x_coords[parent], x_coords[i]], [y_coords[parent], y_coords[i]], 'b-')

        plt.gca().invert_yaxis()
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Skeleton Key Points')
        plt.xlim(500, -500)
        plt.ylim(500, -500)

    fig = plt.figure(figsize=(8, 8))
    num_of_frame = get_num_of_frame(file_path)
    ani = animation.FuncAnimation(fig, func=update, frames=num_of_frame, interval=30)

    # Shut down the terminal when figure is closed
    def close_figure(event):
        if event.canvas.figure == fig:
            sys.exit()     
    fig.canvas.mpl_connect('close_event', close_figure)

    plt.show()

if __name__ == '__main__':
    # file_path = "D:\\UoA\\SOFTENG 700A\\mix-stage-master\\src\\data\\visualization\\features\\cmu0000033570.h5" # noah
    file_path = "D:\\UoA\SOFTENG 700A\\mix-stage-master\\src\\data\\visualization\\features\\100912.h5" # oliver
    # show_file_content(file_path)
    # inspect_h5(file_path)
    # export_to_csv(file_path)
    # plot_keypoints(file_path)
    plot_animation(file_path)