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
                 xlim = [-2.4, 2.4], ylim = [-1.8, 1.6], sigma = 1, alpha = None, legendFlag=False):

        self.listOcSystem = listOcSystem
        self.numAgent = len(listOcSystem)
        self.configDict = listOcSystem[0].configDict
        self.graphPeriodicFlag = graphPeriodicFlag
        self.xlim = xlim
        self.ylim = ylim
        self.zeta = None  # Halfspace matrix (zeta^T y <= iota)
        self.iota = None  # Halfspace vector
        self.sigma = sigma    # Softplus parameter
        self.alpha = alpha    # Leaky parameter for softplus function
        self.legendFlag = legendFlag
        if not graphPeriodicFlag:
            self.adjacencyMat = adjacencyMat
            self.generateMetropolisWeight(adjacencyMat)
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

        thetaNowAll = initialThetaAll
        lossTraj = list()
        thetaAllTraj  = list()
        thetaErrorTraj = list()
        idxIterMargin = 20
        for idxIter in range(int(paraDict["maxIter"])):
            # for dynamic periodic graph
            if self.graphPeriodicFlag:
                idxGraph = int(idxIter % len(self.adjacencyMatList))
                self.generateMetropolisWeight(self.adjacencyMatList[idxGraph])

            # error among theta
            thetaErrorTraj.append(self.computeThetaError(thetaNowAll))
            # compute the gradients
            lossNow, lossVecNow, gradientMatNow = self.computeGradient(initialStateAll, thetaNowAll)
            # exchange information and update theta
            if idxIter < idxIterMargin:
                thetaNextAll = np.matmul(self.weightMat, thetaNowAll) - paraDict["stepSize"] * gradientMatNow
                # thetaNextAll = thetaNowAll - paraDict["stepSize"] * gradientMatNow
            else:
                thetaNextAll = np.matmul(self.weightMat, thetaNowAll)
                # thetaNextAll = thetaNowAll

            lossTraj.append(lossNow)
            thetaAllTraj.append(thetaNowAll)
            gradientNorm = np.linalg.norm(gradientMatNow, axis=1).sum()
            thetaNowAll = thetaNextAll
            if idxIter >= idxIterMargin:
                gradientNorm = 0.0

            printStr = 'Iter:' + str(idxIter) + ', loss:' + str(lossNow) + ', grad norm:' + str(gradientNorm) + ', theta error:' + str(thetaErrorTraj[idxIter])
            print(printStr)

            if (gradientNorm <= 0.01) and (thetaErrorTraj[idxIter] <= 0.001):
                break

        # last one
        resultDictList = list()
        lossVec = np.zeros((self.numAgent))
        for idx in range(self.numAgent):
            resultDictList.append(self.listOcSystem[idx].solve(initialStateAll[idx], thetaNowAll[idx]))
            lossVec[idx] = self.listPDP[idx].lossFun(resultDictList[idx]["xi"], thetaNowAll[idx]).full()[0, 0]

        print('Iter:', idxIter + 1, ' loss:', lossVec.sum())

        for idx in range(self.numAgent):
            print(f'Final Trajectory of Agent {idx}: \n', resultDictList[idx]["xTraj"])

        # plot the loss
        self.plotLossTraj(lossTraj, thetaErrorTraj, blockFlag=False)

        # visualize
        self.visualize(resultDictList, initialStateAll, thetaNowAll, legendFlag=self.legendFlag)

        plt.show()

    def computeGradient(self, initialStateAll, thetaNowAll):
        lossVec = np.zeros((self.numAgent))
        # i-th row is the full gradient for agent-i
        gradientMat = np.zeros((self.numAgent, self.listOcSystem[0].DynSystem.dimParameters))
        for idx in range(self.numAgent):
            resultDict = self.listOcSystem[idx].solve(initialStateAll[idx], thetaNowAll[idx])
            lqrSystem = self.listPDP[idx].getLqrSystem(resultDict, thetaNowAll[idx])
            resultLqr = self.listPDP[idx].solveLqr(lqrSystem)
            lossVec[idx] = self.listPDP[idx].lossFun(resultDict["xi"], thetaNowAll[idx]).full()[0, 0]
            dLdXi = self.listPDP[idx].dLdXiFun(resultDict["xi"], thetaNowAll[idx])
            dXidTheta = np.vstack((np.concatenate(resultLqr["XTrajList"], axis=0),
                np.concatenate(resultLqr["UTrajList"], axis=0)))
            # this is partial derivative
            dLdTheta = self.listPDP[idx].dLdThetaFun(resultDict["xi"], thetaNowAll[idx])

            # this is full derivative
            gradientMat[idx, :] = np.array(np.dot(dLdXi, dXidTheta) + dLdTheta).flatten()

        return lossVec.sum()/self.numAgent, lossVec, gradientMat

    def computeThetaError(self, thetaNowAll):
        error = 0.0
        for i in range(self.numAgent):
            for j in range(self.numAgent):
                error += np.linalg.norm(thetaNowAll[i, :] - thetaNowAll[j, :]) ** 2
        return error

    def plotLossTraj(self, lossTraj, thetaErrorTraj, blockFlag=True):
        _, (ax1, ax2) = plt.subplots(2, 1)
        lossTraj = lossTraj / lossTraj[0]
        ax1.plot(np.arange(len(lossTraj), dtype=int), lossTraj, color="blue", linewidth=2)
        # ax1.set_title("Loss")
        # ax1.legend(["Loss"])
        # ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss (Relative)")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

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
        
        for pdp in self.listPDP:
            pdp.setConstraints(self.zeta, self.iota, self.sigma, self.alpha)   # See the function in PDP.py for details
        

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
