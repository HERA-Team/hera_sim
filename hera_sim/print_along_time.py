import os
import numpy as np

def print_along_time(arr):
    for i in range(len(arr)):
        print(arr[i,0,0,0])

for filename in os.listdir("."):
    if filename.endswith(".npy"):
        print("-----------------------------------"+filename+"----------------------------------------------")
        print_along_time(np.load(filename))
