#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PDP import PDP
from Halfspaces import halfspaceIO


class MultiPDP:
    numAgent: int  # number of agents
    optMethodStr: str  # a string for optimization method

    def __init__(self, listOcSystem: list, adjacencyMat, graphPeriodicFlag=False,
                 xlim = [-2.4, 2.4], ylim = [-1.8, 1.6], sigma = 1, alpha = None,
                 rho = 1.0, legendFlag=False):

        self.listOcSystem = listOcSystem
        self.numAgent = len(listOcSystem)
        self.configDict = listOcSystem[0].configDict
        self.graphPeriodicFlag = graphPeriodicFlag
        self.xlim = xlim
        self.ylim = ylim
        self.zeta = None  # Transposed halfspace matrix (zeta^T y <= iota)
        self.iota = None  # Halfspace vector
        self.sigma = sigma    # Softplus parameter
        self.alpha = alpha    # Leaky parameter for softplus function
        if rho < 0 or rho > 1:
            raise ValueError('rho should be in [0, 1].')
        self.rho = 1.0 if graphPeriodicFlag else rho    # rho in [0, 1], trading-off between shepherding and edge agreement
                                                        # for periodic graph, only shepherding is considered for now due to complexity
        self.formationRadius = 1.0
        self.formationRotation = 0.0
        self.legendFlag = legendFlag
        self.edges = []
        self.incidenceMat = []
        self.relativePosition = []
        if not graphPeriodicFlag:
            self.adjacencyMat = adjacencyMat
            # self.generateMetropolisWeight(adjacencyMat)
            self.generateRegularEdgeAgreement(adjacencyMat, radius=self.formationRadius, rotation=self.formationRotation)
        else:
            self.adjacencyMatList = adjacencyMat
        self.listPDP = list()
        for idx in range(self.numAgent):
            self.listPDP.append(PDP(OcSystem=self.listOcSystem[idx]))

    def generateMetropolisWeight(self, adjacencyMat):
        """
        Generate a Metropolis Weight matrix, whose entry in i-th row and j-th column is the weight for receiver i and sender j
        """
        # self.weightMat[i][j] is the weight for receiver i and sender j
        self.weightMat = np.zeros((self.numAgent, self.numAgent))
        # each row with index i is a vector of weights given this receiver i

        # for adjacencyMat, row i is the adjacency for agent i
        # i-th element in dArray is d_i, namely the number of neighbors (incuding itself)
        dArray = np.sum(adjacencyMat, axis=1)

        for row in range(self.numAgent):
            for col in range(self.numAgent):
                # not compute self.weightMat[row][col] yet
                if row != col:
                    # if col (j) is a neighbor of row (i), do calculation; otherwise 0
                    if adjacencyMat[row][col] > 0.5:
                        self.weightMat[row][col] = 1 / (max(dArray[row], dArray[col]))

        # sum all the non-diagonal weights, then distracted by a vector with ones
        weightMatSelf = np.ones((self.numAgent)) - np.sum(self.weightMat, axis=1)
        # allocate weights for diagonal
        for idx in range(self.numAgent):
            self.weightMat[idx][idx] = weightMatSelf[idx]
    
    def generateRegularEdgeAgreement(self, adjacencyMat, radius = 1.0, rotation = 0.0):
        # Generate incidence matrix based on the adjacency matrix of an undirected graoh
        symmetric_flag = np.array_equal(adjacencyMat, adjacencyMat.T)
        if not symmetric_flag:
            raise ValueError('Invalid Adjacency Matrix.')

        numAgent = adjacencyMat.shape[0]
        edges = []
        for i in range(numAgent):
            for j in range(i + 1, numAgent):
                if adjacencyMat[i, j] != 0:
                    edges.append((i, j))

        incidenceMat = np.zeros((numAgent, len(edges)))

        for k, (i, j) in enumerate(edges):
            incidenceMat[i, k] = 1
            incidenceMat[j, k] = -1

        # Generate relative state variables difference for edge agreement 
        angles = np.linspace(0, 2*np.pi, numAgent, endpoint=False) + rotation
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pos = np.vstack((x, y))

        self.edges = edges
        self.incidenceMat = incidenceMat
        self.relativePosition = pos @ incidenceMat


    def generateRandomInitialTheta(self, radius: float, center=[0.0, 0.0], headingRange=[-3.14, 3.14]):
        """
        Randomly generate initial theta for multiple agents, where the position is randomly distributed on a circle with given radius and center.

        Inputs:
            radius: the radius of the circle
            center; 1d lsit, the position of center of the circle, [px0, py0]
            headingRange: 1d list, the random range of heading angle, [lower bound, upper bound]; to be a deterministic value when the list size is just 1
        
        Outputs:
            initialThetaAll: 2d numpy array, i-th row is the initial theta for agent-i
        """

        initialThetaAll = np.zeros((self.numAgent, self.listOcSystem[0].DynSystem.dimParameters))
        for idx in range(self.numAgent):
            angle = np.random.uniform(-3.14, 3.14)
            px = center[0] + radius * round(math.cos(angle), 2)
            py = center[1] + radius * round(math.sin(angle), 2)
            if len(headingRange) > 1:
                heading = round(np.random.uniform(headingRange[0], headingRange[1]), 2)
            else:
                heading = headingRange[0]
            initialThetaAll[idx, :] = np.array([px, py, heading])
        return initialThetaAll

    def generateRandomInitialState(self, initialThetaAll: np.array, radius: float, seedNo = 114):
        # Need to change this initial state generation function, which doesn't depend on the theta.
        """
        Randomly generate initial state for multiple agents, see more details in each dynamical system object.

        Inputs:
            initialThetaAll: 2d numpy array, i-th row is the initial theta for agent-i
            radius: float, each initial position is generated along a circle, where the center is the associated theta position and the argument radius

        Outputs:
            initialStateAll: 2d numpy array, i-th row is the initial state for agent-i
        """
        initialStateAll = np.zeros((self.numAgent, self.listOcSystem[0].DynSystem.dimStates))
        for idx in range(self.numAgent):
            x0 = self.listOcSystem[0].DynSystem.generateRandomInitialState(initialThetaAll[idx, :], radius, center=initialStateAll[idx, 0:2])
            initialStateAll[idx, :] = x0
        return initialStateAll

    def solve(self, initialStateAll, initialThetaAll, paraDict: dict):
        """
        
        Inputs:
            initialStateAll: 2d numpy array, i-th row is an initial state for agent i
            initialThetaAll: 2d numpy array, i-th row is an initial theta for agent i
        """
        # initialize the problem
        resultDictList = list()
        for idx in range(self.numAgent):
            resultDictList.append(self.listOcSystem[idx].solve(initialStateAll[idx], initialThetaAll[idx]))

        # acquire shepherding boundaries
        self.defineHalfspaces(resultDictList, initialStateAll, initialThetaAll, legendFlag=self.legendFlag)

        for pdp in self.listPDP:
            pdp.setConstraints(self.zeta, self.iota, self.sigma, self.alpha)
            # See the function in PDP.py for details

        thetaNowAll = initialThetaAll
        lossTraj = list()
        thetaAllTraj  = list()
        thetaErrorTraj = list()
        formationErrorTraj = list()
        maxIter = paraDict["maxIter"]
        rho = self.rho
        stepSize = paraDict["stepSize"]
        for idxIter in range(int(paraDict["maxIter"])):
            # for dynamic periodic graph
            if self.graphPeriodicFlag:
                idxGraph = int(idxIter % len(self.adjacencyMatList))
                # self.generateMetropolisWeight(self.adjacencyMatList[idxGraph])
                self.generateRegularEdgeAgreement(self.adjacencyMatList[idxGraph], radius=self.formationRadius, rotation=self.formationRotation)

            # error among theta (not using consensus)
            # thetaErrorTraj.append(self.computeThetaError(thetaNowAll))
            
            # compute the gradients (this also returns the trajectories for formation loss)
            shepherdingLossNow, _, shepherdingGradientMatNow, resultDictList = self.computeShepherdingLossGradient(initialStateAll, thetaNowAll)
            
            # compute formation loss and its gradient using the already computed trajectories
            formationLoss, formationGradient = self.computeFormationLossAndGradient(resultDictList, thetaNowAll)
            formationErrorTraj.append(formationLoss)
            
            # combine gradients with rho weighting
            totalGradient = (1 - rho) * shepherdingGradientMatNow + rho * formationGradient
            
            # exchange information and update theta
            if idxIter < maxIter:
                # thetaNextAll = np.matmul(self.weightMat, thetaNowAll) - stepSize * shepherdingGradientMatNow
                thetaNextAll = thetaNowAll - stepSize * np.exp(-idxIter/50) * totalGradient
            else:
                # thetaNextAll = np.matmul(self.weightMat, thetaNowAll)
                thetaNextAll = thetaNowAll

            totalLoss = rho*shepherdingLossNow + (1-rho)*formationLoss
            lossTraj.append(totalLoss)
            thetaAllTraj.append(thetaNowAll)
            gradientNorm = np.linalg.norm(totalGradient, axis=1).sum()
            thetaNowAll = thetaNextAll
            
            if thetaErrorTraj:
                printStr = 'Iter:' + str(idxIter) + ', mean loss:' + str(totalLoss) + ', grad norm:' + str(gradientNorm) + ', theta error:' + str(thetaErrorTraj[idxIter])
            else:
                printStr = 'Iter:' + str(idxIter) + ', mean loss:' + str(totalLoss) + ', grad norm:' + str(gradientNorm)
            print(printStr)

            # if (gradientNorm <= 0.01) and (thetaErrorTraj[idxIter] <= 0.001):
            if gradientNorm <= 1e-4:
                break

        # final computation using the last iteration's trajectories
        lossVec = np.zeros((self.numAgent))
        for idx in range(self.numAgent):
            lossVec[idx] = self.listPDP[idx].lossFun(resultDictList[idx]["xi"], thetaNowAll[idx]).full()[0, 0]

        print('Iter:', idxIter + 1, ' loss:', lossVec.sum())

        for idx in range(self.numAgent):
            print(f'Final Trajectory of Agent {idx}: \n', resultDictList[idx]["xTraj"])

        print(f'Final Theta of All Agents: \n {thetaNowAll}')

        # plot the loss
        self.plotLossTraj(lossTraj, thetaErrorTraj, blockFlag=False)

        # visualize
        self.visualize(resultDictList, initialStateAll, thetaNowAll, legendFlag=self.legendFlag)

        plt.show()

    def computeShepherdingLossGradient(self, initialStateAll, thetaNowAll):
        shepherdingLossVec = np.zeros((self.numAgent))
        # i-th row is the full gradient for agent-i
        gradientMat = np.zeros((self.numAgent, self.listOcSystem[0].DynSystem.dimParameters))
        resultDictList = list()
        for idx in range(self.numAgent):
            resultDict = self.listOcSystem[idx].solve(initialStateAll[idx], thetaNowAll[idx])
            resultDictList.append(resultDict)
            lqrSystem = self.listPDP[idx].getLqrSystem(resultDict, thetaNowAll[idx])
            resultLqr = self.listPDP[idx].solveLqr(lqrSystem)
            shepherdingLossVec[idx] = self.listPDP[idx].lossFun(resultDict["xi"], thetaNowAll[idx]).full()[0, 0]
            shepherdingLossVec /= self.numAgent
            dLdXi = self.listPDP[idx].dLdXiFun(resultDict["xi"], thetaNowAll[idx])
            dXidTheta = np.vstack((np.concatenate(resultLqr["XTrajList"], axis=0),
                np.concatenate(resultLqr["UTrajList"], axis=0)))
            # this is partial derivative
            dLdTheta = self.listPDP[idx].dLdThetaFun(resultDict["xi"], thetaNowAll[idx])

            # this is full derivative
            gradientMat[idx, :] = np.array(np.dot(dLdXi, dXidTheta) + dLdTheta).flatten()
            gradientMat /= self.numAgent

        return shepherdingLossVec.sum(), shepherdingLossVec, gradientMat, resultDictList

    def computeThetaError(self, thetaNowAll):
        error = 0.0
        for i in range(self.numAgent):
            for j in range(self.numAgent):
                error += np.linalg.norm(thetaNowAll[i, :] - thetaNowAll[j, :]) ** 2
        return error

    def computeFormationLossAndGradient(self, resultDictList, thetaNowAll):
        """
        Compute formation loss and its gradient with respect to theta parameters using LQR system.
        
        Args:
            resultDictList: List of result dictionaries from each agent's optimal control solution
            thetaNowAll: Current theta values for all agents
            
        Returns:
            formation_loss: Total formation loss across all agents and edges
            formation_gradient: Gradient matrix of shape (numAgent, dimParameters)
        """
        formation_loss = 0.0
        formation_gradient = np.zeros((self.numAgent, self.listOcSystem[0].DynSystem.dimParameters))
        
        # Pre-compute LQR systems for all agents
        dXidTheta_list = []
        for idx in range(self.numAgent):
            lqrSystem = self.listPDP[idx].getLqrSystem(resultDictList[idx], thetaNowAll[idx])
            resultLqr = self.listPDP[idx].solveLqr(lqrSystem)
            
            # Get sensitivity matrices for agent idx
            dXidTheta = np.vstack((np.concatenate(resultLqr["XTrajList"], axis=0),
                                   np.concatenate(resultLqr["UTrajList"], axis=0)))
            dXidTheta_list.append(dXidTheta)
        
        # Iterate directly through edges
        for edge_idx, (i, j) in enumerate(self.edges):
            # Get trajectories for both agents
            traj_i = resultDictList[i]["xTraj"]  # Shape: (horizon+1, dimStates)
            traj_j = resultDictList[j]["xTraj"]  # Shape: (horizon+1, dimStates)
            
            # Get desired relative position from incidence matrix
            desired_relative_pos = self.relativePosition[:, edge_idx]  # Shape: (2,)
            
            # Compute formation loss and gradient over the entire trajectory
            for t in range(traj_i.shape[0]):
                # Extract position components (x, y) from state
                pos_i = traj_i[t, :2]  # Position of agent i at time t
                pos_j = traj_j[t, :2]  # Position of agent j at time t
                
                # Compute actual relative position
                actual_relative_pos = pos_i - pos_j
                
                # Compute error: difference between actual and desired relative positions
                diff_pos = actual_relative_pos - desired_relative_pos
                
                # Add to formation loss (using L2 norm)
                formation_loss += np.linalg.norm(diff_pos) ** 2
                
                # Compute gradient contributions for both agents i and j
                # For agent i: d/dtheta_i ||diff_pos||^2 = 2 * diff_pos^T * d/dtheta_i pos_i
                # For agent j: d/dtheta_j ||diff_pos||^2 = -2 * diff_pos^T * d/dtheta_j pos_j
                
                # Agent i gradient
                state_idx = t * self.listOcSystem[0].DynSystem.dimStates
                pos_sensitivity_i = dXidTheta_list[i][state_idx:state_idx+2, :]  # Shape: (2, dimParameters)
                gradient_contribution_i = 2 * np.dot(diff_pos, pos_sensitivity_i)
                formation_gradient[i, :] += gradient_contribution_i
                
                # Agent j gradient (note the negative sign)
                pos_sensitivity_j = dXidTheta_list[j][state_idx:state_idx+2, :]  # Shape: (2, dimParameters)
                gradient_contribution_j = -2 * np.dot(diff_pos, pos_sensitivity_j)
                formation_gradient[j, :] += gradient_contribution_j
        
        # Normalize by number of edges
        num_edges = len(self.edges)
        return formation_loss / num_edges, formation_gradient / num_edges

    def plotLossTraj(self, lossTraj, thetaErrorTraj, blockFlag=True):
        if thetaErrorTraj:
            _, (ax1, ax2) = plt.subplots(2, 1)
        else:
            _, ax1 = plt.subplots(1, 1)
            
        lossTraj = lossTraj / lossTraj[0]
        ax1.plot(np.arange(len(lossTraj), dtype=int), lossTraj, color="blue", linewidth=2)
        ax1.set_ylabel("Total Loss (Relative)")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        if thetaErrorTraj:
            ax2.plot(np.arange(len(thetaErrorTraj), dtype=int), thetaErrorTraj, color="blue", linewidth=2)
            # ax2.set_title("Theta error")
            # ax2.legend(["error"])
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Error")
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show(block=blockFlag)

    def plotArrow(self, stateNow):
        magnitude = 0.1
        dx = magnitude * math.cos(stateNow[2])
        dy = magnitude * math.sin(stateNow[2])
        # width = 0.03
        # plt.arrow(stateNow[0], stateNow[1], dx, dy, alpha=0.5, color="green", width=width, head_width=7*width, head_length=3*width)
        plt.arrow(stateNow[0], stateNow[1], dx, dy, alpha=0.5, color="green")

    def plotAHalfspace(self, ax, zetaRow, iota):
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        zeta1, zeta2 = zetaRow

        # Boundary line (Ax = b)
        if abs(zeta2) > 1e-8:
            x_vals = np.linspace(xmin, xmax, 10)
            y_vals = (iota - zeta1 * x_vals) / zeta2
        else:
            x_vals = np.full(2, iota / zeta1)
            y_vals = np.array([ymin, ymax])
        ax.plot(x_vals, y_vals, 'r-.', lw=2)

        # Shaded Area (Ax <= b)
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 200),
                           np.linspace(ymin, ymax, 200))
        Region = (zeta1*X + zeta2*Y - iota <= 0.0)
        ax.contourf(X, Y, Region, levels=[0.5, 1],
                    colors=['#ff6666'], alpha=0.3)


    def defineHalfspaces(self, resultDictList, initialStateAll, thetaAll, legendFlag=False):
        self.zeta, self.iota = halfspaceIO(
                                initial_traj = [resultDictList[idx]["xTraj"] for idx in range(self.numAgent)],
                                initial_states = initialStateAll, 
                                terminal_states = thetaAll,
                                xlim = self.xlim, ylim = self.ylim, 
                                numAgent = self.numAgent, legendFlag = legendFlag)
        

    def visualize(self, resultDictList, initialStateAll, thetaAll, blockFlag=True, legendFlag=True):
        _, ax1 = plt.subplots(1, 1)

        for idx in range(self.numAgent):
            ax1.plot(resultDictList[idx]["xTraj"][:,0], resultDictList[idx]["xTraj"][:,1], color="blue", linewidth=2)
            ax1.scatter(initialStateAll[idx, 0], initialStateAll[idx, 1], marker="o", color="magenta")
            ax1.scatter(thetaAll[idx, 0], thetaAll[idx, 1], marker="^", color="green")

        # plot arrows for heading angles
        for idx in range(self.numAgent):
            self.plotArrow(initialStateAll[idx, :])
            self.plotArrow(resultDictList[idx]["xTraj"][-1, :])

        if self.iota is not None:
            for Ai, bi in zip(self.zeta, self.iota):
                self.plotAHalfspace(ax1, Ai, bi)

        # Plot ideal formation polygon
        if self.relativePosition.any():
            # Calculate center of final positions (thetaAll)
            center_x = np.mean(thetaAll[:, 0])
            center_y = np.mean(thetaAll[:, 1])
            
            # Generate ideal formation positions
            angles = np.linspace(0, 2*np.pi, self.numAgent, endpoint=False) + self.formationRotation
            ideal_x = center_x + self.formationRadius * np.cos(angles)
            ideal_y = center_y + self.formationRadius * np.sin(angles)
            
            # Plot ideal formation polygon
            ideal_positions = np.column_stack((ideal_x, ideal_y))
            # Close the polygon by adding the first point at the end
            ideal_positions_closed = np.vstack([ideal_positions, ideal_positions[0]])
            ax1.plot(ideal_positions_closed[:, 0], ideal_positions_closed[:, 1], 
                    color='purple', linestyle=':', linewidth=2, alpha=0.7, label='Ideal Formation')
            
            # Plot ideal formation vertices
            ax1.scatter(ideal_x, ideal_y, marker='s', color='purple', s=10, alpha=0.7, 
                       label='Ideal Positions')

        # ax1.set_title("Trajectory")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)

        # plot legends
        if legendFlag:
            labels = ["Start", "Goal"]
            marker = ["o", "^"]
            colors = ["magenta", "green"]
            f = lambda m,c: plt.plot([], [], marker=m, color=c, ls="none")[0]
            handles = [f(marker[i], colors[i]) for i in range(len(labels))]
            handles.append(plt.plot([],[], linestyle=None, color="blue", linewidth=2)[0])
            labels.append("Trajectory")
            if self.iota is not None:
                handles.append(plt.plot([],[], linestyle='-.', color="red", linewidth=2)[0])
                labels.append("Shepherding Bondary")
            if self.relativePosition.any():
                handles.append(plt.plot([],[], linestyle=':', color="purple", linewidth=2)[0])
                labels.append("Ideal Formation")
                handles.append(plt.scatter([], [], marker='s', color='purple', s=10))
                labels.append("Ideal Positions")
            plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left", framealpha=1)

        _, (ax21, ax22, ax23) = plt.subplots(3, 1)
        for idx in range(self.numAgent):        
            ax21.plot(resultDictList[idx]["timeTraj"][:-1], resultDictList[idx]["uTraj"][:,0], color="blue", linewidth=2)
            # ax21.legend(["velocity input"])
            ax21.set_xlabel("time [sec]")
            ax21.set_ylabel("velocity [m/s]")

            ax22.plot(resultDictList[idx]["timeTraj"][:-1], resultDictList[idx]["uTraj"][:,1], color="blue", linewidth=2)
            # ax22.legend(["angular velocity input"])s
            ax22.set_xlabel("time [sec]")
            ax22.set_ylabel("angular velocity [rad/s]")

            ax23.plot(resultDictList[idx]["timeTraj"], resultDictList[idx]["xTraj"][:,2], color="blue", linewidth=2)
            ax23.scatter(resultDictList[idx]["timeTraj"][0], initialStateAll[idx, 2], marker="o", color="blue")
            # ax23.scatter(resultDictList[idx]["timeTraj"][-1], thetaAll[idx, 2], marker="*", color="red")
            # ax23.legend(["Optimal Trajectory", "start", "goal"])
            ax23.set_xlabel("time [sec]")
            ax23.set_ylabel("heading [radian]")

        plt.show(block=blockFlag)
