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

    Constraints are available in real-time via get_constraints() method. You can also
    add/remove constraints programmatically using add_constraint() and remove_constraint().

    Instructions:
    Left button: Decide points that determine halfspaces
        1st click: The first point for a line
        2nd click: The second point to decide a line
        3rd click: Click to choose which side has violated constraints.
    Right button: Undo
        + If the hyperplane is pending construction: Delete the last one point
        + Else: Delete the last halfspace we decided. 
    """

    def __init__(self, initial_traj, initial_states, terminal_states,
                 xlim = [-2.4, 2.4], ylim = [-1.8, 1.6], numAgent = 1, legendFlag = False):
        self.saved = []  # List of (Arow, b) tuples for constraints
        self.current_pts = []
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_title("Left click to pick: p1, p2, violated side. Right click to Undo. \n Close window to terminate.")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.xlim = xlim
        self.ylim = ylim
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
            handles.append(plt.plot([],[], linestyle='-.', color="red", linewidth=2)[0])
            labels.append("Shepherding Bondary")
            plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left", framealpha=1)

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.line_artists = []
        self.fill_artists = []
        self.pt_artists = []
        
        # Initialize A and b as empty arrays for real-time access
        self.A = np.zeros((0, 2))
        self.b = np.zeros((0,))

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
            self.add_constraint(Arow, b, plot=True)
            self._clear_current()

    def _undo(self):
        if self.current_pts:
            self.current_pts.pop()
            self._update_temp_points()
        elif self.saved:
            self.remove_constraint(plot=True)  # Remove last constraint

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
        norm_n = np.linalg.norm(n)
        n /= norm_n
        b /= norm_n
        return n, b

    def _plot_halfspace(self, Arow, b):
        xmin, xmax = self.xlim[0], self.xlim[1]
        ymin, ymax = self.ylim[0], self.ylim[1]
        a1, a2 = Arow
        if abs(a2) > 1e-8:
            x_vals = np.linspace(xmin, xmax, 3)
            y_vals = (b - a1*x_vals)/a2
        else:
            x_vals = np.full(2, b/a1)
            y_vals = np.array([ymin, ymax])

        line = self.ax.plot(x_vals, y_vals, 'r-.', lw=2)[0]
        self.line_artists.append(line)

        # shade Ax <= b region
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 200),
                           np.linspace(ymin, ymax, 200))
        Z = Arow[0]*X + Arow[1]*Y - b
        Region = ( Z <= 0 )
        fill = self.ax.contourf(X, Y, Region, levels=[0.5, 1], colors=['#ff6666'], alpha=0.3)
        self.fill_artists.append(fill)
        self.fig.canvas.draw_idle()

    def _update_constraint_matrices(self):
        """Update A and b matrices from the saved constraints list."""
        if self.saved:
            self.A = np.vstack([a for a, _ in self.saved])
            self.b = np.array([b for _, b in self.saved])
        else:
            self.A = np.zeros((0, 2))
            self.b = np.zeros((0,))
    
    def get_constraints(self):
        """
        Get current constraint matrices A and b in real-time.
        
        Returns:
            A: numpy array of shape (n_constraints, 2) representing constraint normals
            b: numpy array of shape (n_constraints,) representing constraint offsets
               Constraints are of the form: A @ x <= b
        """
        self._update_constraint_matrices()
        return self.A, self.b
    
    def add_constraint(self, Arow, b, plot=False):
        """
        Add a constraint programmatically.
        
        Args:
            Arow: numpy array of shape (2,) representing the constraint normal vector
            b: scalar representing the constraint offset (constraint: Arow @ x <= b)
            plot: bool, if True and GUI is active, plot the constraint
        """
        # Ensure Arow is a numpy array and normalized
        Arow = np.array(Arow)
        if Arow.shape != (2,):
            raise ValueError(f"Arow must be of shape (2,), got {Arow.shape}")
        
        # Normalize if not already normalized
        norm = np.linalg.norm(Arow)
        if norm > 1e-8:
            Arow = Arow / norm
            b = b / norm
        
        self.saved.append((Arow, b))
        self._update_constraint_matrices()
        
        if plot and hasattr(self, 'ax'):
            self._plot_halfspace(Arow, b)

    
    def remove_constraint(self, plot=False):
        """
        Remove a constraint by index.
        
        Args:
            plot: bool, if True and GUI is active, update the plot
        """
        if self.saved:
            self.saved.pop()
            self._update_constraint_matrices()
            
            if plot and hasattr(self, 'ax'):
                # Replot all remaining constraints to maintain consistency
                self._replot_all_constraints()
    
    def clear_constraints(self, plot=False):
        """
        Clear all constraints.
        
        Args:
            plot: bool, if True and GUI is active, clear the plot
        """
        self.saved = []
        self._update_constraint_matrices()
        
        if plot and hasattr(self, 'ax'):
            # Remove all plot elements
            for line in self.line_artists:
                line.remove()
            for fill in self.fill_artists:
                fill.remove()
            self.line_artists = []
            self.fill_artists = []
            self.fig.canvas.draw_idle()
    
    def _replot_all_constraints(self):
        """Replot all constraints (used after removing a constraint)."""
        # Clear existing plots
        for line in self.line_artists:
            line.remove()
        for fill in self.fill_artists:
            fill.remove()
        self.line_artists = []
        self.fill_artists = []
        
        # Replot all constraints
        for Arow, b in self.saved:
            self._plot_halfspace(Arow, b)

def halfspaceIO(initial_traj, initial_states, terminal_states, xlim = [-2.4, 2.4], ylim = [-1.8, 1.6],
                 numAgent = 1, legendFlag = False):
    builder = halfspaceBuilder(initial_traj, initial_states, terminal_states, xlim, ylim, numAgent, legendFlag)
    plt.show()
    return builder.A, builder.b