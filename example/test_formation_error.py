#!/usr/bin/env python3
"""
Test script for formation error computation in MultiPDP.
This script demonstrates how to use the formation error functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from MultiPDP import MultiPDP
from Unicycle import Unicycle
from OcSystem import OcSystem

def test_formation_error():
    """
    Test the formation error computation with a simple multi-agent scenario.
    """
    # Create a simple adjacency matrix for 3 agents in a line formation
    adjacencyMat = np.array([
        [0, 1, 0],  # Agent 0 connected to agent 1
        [1, 0, 1],  # Agent 1 connected to agents 0 and 2
        [0, 1, 0]   # Agent 2 connected to agent 1
    ])
    
    # Create unicycle systems for each agent
    timeStep = 0.1
    horizonSteps = 20
    configDict = {"timeStep": timeStep, "horizonSteps": horizonSteps}
    
    # Create dynamical systems
    listOcSystem = []
    for i in range(3):
        dynSystem = Unicycle(timeStep, horizonSteps)
        ocSystem = OcSystem(dynSystem, configDict)
        listOcSystem.append(ocSystem)
    
    # Create MultiPDP instance with formation error enabled
    multiPDP = MultiPDP(
        listOcSystem=listOcSystem,
        adjacencyMat=adjacencyMat,
        graphPeriodicFlag=False,
        rho=0.5,  # 50% weight on formation error, 50% on individual loss
        legendFlag=True
    )
    
    # Generate initial conditions
    initialThetaAll = multiPDP.generateRandomInitialTheta(radius=1.0, center=[0.0, 0.0])
    initialStateAll = multiPDP.generateRandomInitialState(initialThetaAll, radius=0.2)
    
    print("Initial Theta (goals):")
    print(initialThetaAll)
    print("\nInitial States:")
    print(initialStateAll)
    
    # Test formation error computation
    resultDictList = []
    for idx in range(3):
        resultDictList.append(listOcSystem[idx].solve(initialStateAll[idx], initialThetaAll[idx]))
    
    formationError = multiPDP.computeFormationError(resultDictList, initialThetaAll)
    print(f"\nInitial Formation Error: {formationError}")
    
    # Test formation error gradient
    formationGradient = multiPDP.computeFormationErrorGradient(resultDictList, initialThetaAll)
    print(f"\nFormation Error Gradient Shape: {formationGradient.shape}")
    print("Formation Error Gradient:")
    print(formationGradient)
    
    # Run optimization with formation error
    paraDict = {
        "stepSize": 0.01,
        "maxIter": 50,
        "method": "Vanilla"
    }
    
    print("\nRunning optimization with formation error...")
    multiPDP.solve(initialStateAll, initialThetaAll, paraDict)
    
    print("\nFormation error computation and gradient migration completed successfully!")

if __name__ == "__main__":
    test_formation_error()
