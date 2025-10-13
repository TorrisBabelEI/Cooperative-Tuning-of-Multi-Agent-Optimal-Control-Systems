import numpy as np
import matplotlib.pyplot as plt
import math

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

    def __init__(self, initial_traj, initial_states, terminal_states,
                 xlim = [-2.4, 2.4], ylim = [-1.8, 1.6], numAgent = 1, legendFlag = False):
        self.saved = []
        self.current_pts = []
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_title("Left click to pick: p1, p2, side. Right click to Undo. Close window to terminate.")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal', 'box')

        self.numAgent = numAgent

        for idx in range(self.numAgent):
            self.ax.plot(initial_traj[idx][:,0], initial_traj[idx][:,1], color='blue', alpha=.75)
            self.ax.scatter(initial_states[idx, 0], initial_states[idx, 1], marker="o", color="magenta")
            self.ax.scatter(terminal_states[idx, 0], terminal_states[idx, 1], marker="^", color="green")

            self._plot_arrow(initial_states[idx, :])
            self._plot_arrow(initial_traj[idx][-1, :])

        if legendFlag:
            labels = ["Start", "Goal"]
            marker = ["o", "^"]
            colors = ["magenta", "green"]
            f = lambda m,c: plt.plot([], [], marker=m, color=c, ls="none")[0]
            handles = [f(marker[i], colors[i]) for i in range(len(labels))]
            handles.append(plt.plot([],[], color="blue", linewidth=2)[0])
            labels.append("Trajectory")
            handles.append(plt.plot([],[], linestyle=None, color="red", linewidth=2)[0])
            labels.append("Shepherding Bondary [Infeasible Shaded]")
            plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left", framealpha=1)

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)

        self.line_artists = []
        self.fill_artists = []
        self.pt_artists = []

    def _plot_arrow(self, stateNow):
        magnitude = 0.1
        dx = magnitude * math.cos(stateNow[2])
        dy = magnitude * math.sin(stateNow[2])
        # width = 0.03
        plt.arrow(stateNow[0], stateNow[1], dx, dy, alpha=0.5, color="green")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # left
            self._add_point(event)
        elif event.button == 3:  # right
            self._undo()

    def _add_point(self, event):
        x, y = event.xdata, event.ydata
        self.current_pts.append(np.array([x, y]))
        self._update_temp_points()

        if len(self.current_pts) == 3:
            p1, p2, side = self.current_pts
            Arow, b = self._pts_to_halfspace(p1, p2, side)
            self.saved.append((Arow, b))
            self._plot_halfspace(Arow, b)
            self._clear_current()

    def _undo(self):
        if self.current_pts:
            self.current_pts.pop()
            self._update_temp_points()
        elif self.saved:
            self.saved.pop()
            if self.line_artists:
                self.line_artists.pop().remove()
            if self.fill_artists:
                self.fill_artists.pop().remove()
            self.fig.canvas.draw_idle()

    def _update_temp_points(self):
        for p in self.pt_artists:
            p.remove()
        self.pt_artists = [self.ax.plot(pt[0], pt[1], 'ko', markersize=6)[0] for pt in self.current_pts]
        self.fig.canvas.draw_idle()

    def _clear_current(self):
        for p in self.pt_artists:
            p.remove()
        self.pt_artists = []
        self.current_pts = []
        self.fig.canvas.draw_idle()

    def _pts_to_halfspace(self, p1, p2, side_pt):
        v = p2 - p1
        n = np.array([v[1], -v[0]])
        if np.linalg.norm(n) < 1e-8:
            raise ValueError("p1 is too close to p2.")
        b = np.dot(n, p1)
        if np.dot(n, side_pt) > b:
            n = -n
            b = -b
        n /= np.linalg.norm(n)
        b /= np.linalg.norm(n)
        return n, b

    def _plot_halfspace(self, Arow, b):
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        a1, a2 = Arow
        if abs(a2) > 1e-8:
            x_vals = np.linspace(xmin, xmax, 200)
            y_vals = (b - a1*x_vals)/a2
        else:
            x_vals = np.full(2, b/a1)
            y_vals = np.array([ymin, ymax])

        line = self.ax.plot(x_vals, y_vals, 'r-', lw=2)[0]
        self.line_artists.append(line)

        # shade Ax <= b region
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
        Z = Arow[0]*X + Arow[1]*Y - b
        fill = self.ax.contourf(X, Y, Z, levels=[-1e9, 0], colors=['#ff6666'], alpha=0.5)
        self.fill_artists.append(fill)
        self.fig.canvas.draw_idle()

    def on_close(self, event):
        A = np.vstack([a for a, _ in self.saved]) if self.saved else np.zeros((0,2))
        b = np.array([b for _, b in self.saved]) if self.saved else np.zeros((0,))
        print("\n=== Decided Constraints ===")
        for i, (a, bi) in enumerate(zip(A, b)):
            print(f"{i}: A = [{a[0]:.4f}, {a[1]:.4f}], b = {bi:.4f}")
        self.A, self.b = A, b

def halfspace_io(initial_traj, initial_state, xlim = [-2.4, 2.4], ylim = [-1.8, 1.6], numAgent = 1):
    builder = halfspaceBuilder(initial_traj, initial_state, xlim, ylim, numAgent)
    plt.show()
    return builder.A, builder.b