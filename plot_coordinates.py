import numpy as np
import matplotlib.pyplot as plt

FILE_LOCATION = "Data/copter.csv"
# FILE_LOCATION = "Data/turtlebot.csv"

def main():
    data = np.genfromtxt(FILE_LOCATION, delimiter=",")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    series = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    cbar = fig.colorbar(series, ax=ax, extend='both')
    cbar.minorticks_on()

    plt.show()
    A = 1

if __name__ == "__main__":
    main()