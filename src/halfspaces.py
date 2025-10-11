import numpy as np
import matplotlib.pyplot as plt

class halfspaceBuilder:
    """
    This script is designed to determine hyperplane constraints of shepherding boundaries.

    By default, it will show the rectangle boundary of the AIMS lab enviroment, which is
    [-2.4, 2.4] x [-1.8, 1.6] in meters on x-y directions. The initial location of each
    agents and the optimal trajectories under initally guessed patameters will be displayed
    for references to choose shepherding boundaries.

    Instructions:
    Left button: Decide points that determine halfspaces
        1st click: The first point for a line
        2nd click: The second point to decide a line
        3rd click: Click to choose which side has negative values.
    Right button: Undo
        + If the hyperplane is pending construction: Delete the last one point
        + Else: Delete the last halfspace we decided. 
    Close the window: Terminate the process and generate Zeta and iota for the constraints
    """

    def __init__(self):
        self.saved = []  # list of (A_row (2,), b_scalar)
        self.current_pts = []  # points for current halfspace construction
        # plotting
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(self._title_text())
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal', 'box')
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.line_artists = []  # plotted constraint lines
        self.pt_artists = []  # plotted points for current halfspace
        self.saved_artists = []  # markers for saved side points (optional)

