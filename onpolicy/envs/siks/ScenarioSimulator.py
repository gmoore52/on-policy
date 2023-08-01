from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ScenarioDataset import ScenarioDatasetAPI
from .MultiAgentEnv import MultiAgentEnv
import numpy as np
import skimage.draw as skd

from gym.spaces import Discrete


class ScenarioSimulator(MultiAgentEnv):
    def __init__(self,
                 datasetName="filename.json",
                 filepath = "./",
                 episode_limit = 50,
                 randomly_deploy=True):
        
        # Create the API and dataset
        self.API = ScenarioDatasetAPI(filepath=filepath,
                                      filename=datasetName)
        self.Dataset = self.API.Dataset
        
        self.episode_limit = episode_limit
        self.start_power = self.Dataset.Sensors[0].RemainingPower
        
        self.ScenarioDim = self.Dataset.Size
        # Initialize class members/variables but do not populate them properly
        self.Sensors = {}
        self.SenseData = self.Dataset.SensorData
        # Make the sensors a map storing their respective ID and their sensor object
        for id, sensor in enumerate(self.Dataset.Sensors):
            self.Sensors[id] = sensor
            
        self.DeploymentField = np.zeros(self.ScenarioDim)
    # change int type to int32 for 2^32 degrees of coverage, current 255 max coverage
        self.CoverageMatrix = np.zeros(self.ScenarioDim, dtype=np.int8)
        # Create the desired matrix to contain the desired degress of coverage
        self.DesiredMatrix = np.zeros(self.ScenarioDim, dtype=np.int8)

        # Encoded action index matrix to encode and decode actions
        self.EncodeActionMatrix = np.reshape(np.arange(0, self.ScenarioDim[0]*self.ScenarioDim[1]), self.ScenarioDim)
        
        
        # Randomly deploy sensors immediately and store random sensor deployments
        if randomly_deploy:
            self.RandomlyDeploySensors()
            self.API.StoreDataset()

        # Set foi centroid
        foi_points = [[f.Position.yPos, f.Position.xPos] for f in self.Dataset.FOI]
        foi_x = [p[0] for p in foi_points]
        foi_y = [p[1] for p in foi_points]
        self.foi_centroid = (int(np.sum(foi_y) / len(foi_points)),
                             int(np.sum(foi_x) / len(foi_points)))
        # print(self.Sensors)
        assert len(self.Sensors.keys()) > 0, \
                "Invalid number of sensors"

        self.n_agents = len(self.Sensors.keys())

        # self.n_actions = len(self.get_avail_agent_actions(0))
        self.n_actions = self.ScenarioDim[0]*self.ScenarioDim[1]

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append(self.get_obs_size())

        # for i in range(self.n_agents):
        #     # Multiply by two because there are two datapoints for each action
        #     self.action_space.append(Discrete(self.n_actions))

        self.share_observation_space.append(self.get_state_size())
        # print("shared_obs: ", self.share_observation_space)
        # print("act: ", self.action_space)

        
        # Use reset() to populate variables
        self.reset()
        # print("here1")
                                   
                                   
    def step(self, actions):
        '''
        Returns reward, teminated info.
        Steps environment through each set of actions
        '''
        
        assert list(range(len(actions))) == list(self.Sensors.keys()), \
            "Not enough actions loaded to actions list or invalid actions"
        


        # Initialize optimal reward values to be updated
        # Our first experiment for a reward function we will have optimal coverage
        # reward equal to the n_senors*2
        # and have the optimal energy reward equal to 0 with the minimum or worst reward
        # equal to -1*n_sensors
        totalEnergyReward   = 0
        totalCoverageReward = 0
        # totalCoverageReward = 2*len(self.Sensors.keys())
        
        # Make infos list
        infos = [{} for i in range(len(self.Sensors.keys()))]
        dones = np.zeros((len(self.Sensors.keys())), dtype=bool)
        


        for id, act in enumerate(actions):
            # Decode the actions
            action = np.where(self.EncodeActionMatrix == act)
            # action = (action[1], action[0])
            # print(action)


            zeroCostAction = [self.Sensors[id].Position.yPos, 
                              self.Sensors[id].Position.xPos]
            # If the action is invalid because it has been taken by another sensor 
            # then dont move
            # print(self.DeploymentField)
            if self.DeploymentField[action[0], action[1]] != 0 \
                or self.Sensors[id].RemainingPower == 0:
                action = zeroCostAction
            
            # Make the energy cost reward with a constant energy cost 
            # of 1 energy per moving action
            energyReward = 0 if list(action) == list(zeroCostAction) \
                             else 0
            
           
            # Sum the agent energy reward to total energy reward
            totalEnergyReward += energyReward
            # Make current position available
            self.DeploymentField[zeroCostAction[0], zeroCostAction[1]] = 0
            # Make new position taken
            self.DeploymentField[action[0], action[1]] = 1
            # Conduct the movement/action
            self.Sensors[id].Position.yPos = action[0]
            self.Sensors[id].Position.xPos = action[1]
            # Set the dones to true to show that the agent at id is done
            dones[id] = False
            # infos[id]["energy_reward"] = energyReward
            infos[id]["position"] = action
            
            
        self.ComputeDeploymentField()
        self.ComputeCoverage()
        
        # Set the positive values to a zero to show that they don't matter
        RewardMatrix = np.subtract(self.CoverageMatrix, self.DesiredMatrix)
        RewardMatrix[RewardMatrix > 0] = 0
        totalCoverageDifference = 0
        # print(RewardMatrix)
        
        for foi in self.Dataset.FOI:
            leftX, rightX = foi.TopLeft.xPos, foi.BotRight.xPos
            leftY, rightY = foi.TopLeft.yPos, foi.BotRight.yPos
            
            # As foi are squares use total distance between each point on a line
            # to get the width and height and then use this to calculate total area
            # foiArea = (abs(leftX-rightX))*(abs(leftY-rightY))
            # coverageDifference = 0
            
            totalCoverageDifference += np.sum(RewardMatrix[leftY:rightY,
                                                           leftX:rightX])
            
        totalCoverageReward += totalCoverageDifference
            
        # Populate the infos values
        for id in range(len(self.Sensors.keys())):
            # infos[id]["total_energy_reward"] = totalEnergyReward
            infos[id]["total_coverage_reward"] = totalCoverageReward
            # infos[id]["coverage_difference"] = totalCoverageDifference
            infos[id]["sensing_range"] = self.SenseData.SensingRange
            infos[id]["movement_range"] = self.SenseData.MoveRange
            
        # Get final rewards
        finalReward = totalCoverageReward + totalEnergyReward
        rewards = [[[finalReward]]*len(self.Sensors.keys())]
        
        
        
        avail_actions = []
        for i in range(len(self.Sensors.keys())):
            avail_actions.append(self.get_avail_agent_actions(i))
        
        
        
        local_obs = self.get_obs()
        global_state = self.get_state()
        
        
        
        return local_obs, global_state, rewards, dones, infos, avail_actions
    
    def get_obs_agent(self, agentID):
        '''
        Function that gets observation data for each respective sensor with id agentID
        '''
        # Get Sensor data for each sensor that contains the proper 
        sensor = self.Sensors[agentID]
        senseRange, moveRange = self.SenseData.SensingRange, self.SenseData.MoveRange
        # Make size and position of the observations in relation to the sensor data
        # and to each respective sensors position
        obsSize = (1+(senseRange+moveRange)*2, 1+(senseRange+moveRange)*2)
        obsPos = [sensor.Position.yPos - (obsSize[0]//2), sensor.Position.xPos - (obsSize[1]//2)]
        # Make the observation matrices to be returned at the end of the function
        covMatrix = np.zeros(obsSize)
        posMatrix = np.zeros(obsSize)
        
        borderSize = ((obsSize[0]*2)+self.ScenarioDim[0], (obsSize[1]*2)+self.ScenarioDim[1])
        # Use -1 for coverageMatrix as negative is nonexistent coverage
        BorderCoverageMatrix   = np.full(borderSize, -1)
        # Use 2 for deploymentMatrix as 2 is invalid index for deployment
        BorderDeploymentMatrix = np.full(borderSize,  2)
        
        # Fill the proper areas in the border matrices
        BorderCoverageMatrix[obsSize[0]:-obsSize[0], obsSize[1]:-obsSize[1]] =  \
                np.subtract(self.CoverageMatrix, self.DesiredMatrix)
        BorderDeploymentMatrix[obsSize[0]:-obsSize[0], obsSize[1]:-obsSize[1]] = self.DeploymentField
        
        # TODO: Debug if the positions are valid or if they need pushed over by a constant
        # Make new positions for the border matrices
        borderPos = (int(obsPos[0]+obsSize[0]), int(obsPos[1]+obsSize[1]))
        # print(borderPos)
        # print(borderPos[0]-1, borderPos[0]+obsSize[0], 
        #                            borderPos[1]-1, borderPos[1]+obsSize[1])
        # Calculate the coverage and positon matrix based on the border matrices
        covMatrix = \
            BorderCoverageMatrix[borderPos[0]-1:borderPos[0]+obsSize[0],
                                 borderPos[1]-1:borderPos[1]+obsSize[1]]
        
        posMatrix = \
            BorderDeploymentMatrix[borderPos[0]-1:borderPos[0]+obsSize[0], 
                                   borderPos[1]-1:borderPos[1]+obsSize[1]]
            
        return covMatrix
    
    def get_obs(self):
        '''
        Returns the observations of each agent in a list
        '''
        # self.ComputeCoverage()
        # self.ComputeDeploymentField()
        return [self.get_obs_agent(id) for id in self.Sensors.keys()]
    
    def get_obs_size(self):
        return [x for x in np.shape(self.get_obs())]
    
    def get_state(self):
        '''
        Returns the global state
        '''
        return self.CoverageMatrix, self.DeploymentField, self.DesiredMatrix
    
    def get_state_size(self):
        return [x for x in np.shape(self.get_state())]

    def get_avail_actions(self):
        '''
        Function returns the list of available actions for each agent id
        '''
        # actions = {}
        # for id in self.Sensors.keys():
        #     actions[id] = self.get_avail_agent_actions(id)
            
        
        return [self.get_avail_agent_actions(id) for id in self.Sensors.keys()]
    
    def get_avail_agent_actions(self, agentID):
        '''
        Function returns an action list that contains valid movement positions
        for the sensor with the id agentID
        '''
        actionList = []
        # We will use the midpoint-circle algorithm to calculate valid action indices
        sensor = self.Sensors[agentID]
        xPos, yPos = sensor.Position.yPos, sensor.Position.xPos
        # Make the zero cost action which will be the "null" action set and causes sensor to remain in place
        zeroCostAction = [xPos, yPos]
        # Append a default action that will not cost any movement
        actionList.append(zeroCostAction)
        if self.Sensors[agentID].RemainingPower == 0:
            return actionList
        # We make the radius the movement range
        radius = self.SenseData.MoveRange
        f = 1 - radius
        ddf_x = 1
        ddf_y = -2 * radius
        x = 0
        y = radius
        actionList.append([xPos, yPos+radius])
        actionList.append([xPos, yPos-radius])
        actionList.append([xPos+radius, yPos])
        actionList.append([xPos-radius, yPos])

        while x < y:
            if f >= 0: 
                y -= 1
                ddf_y += 2
                f += ddf_y
            x += 1
            ddf_x += 2
            f += ddf_x    
            actionList.append([xPos + x, yPos + y])
            actionList.append([xPos - x, yPos + y])
            actionList.append([xPos + x, yPos - y])
            actionList.append([xPos - x, yPos - y])
            actionList.append([xPos + y, yPos + x])
            actionList.append([xPos - y, yPos + x])
            actionList.append([xPos + y, yPos - x])
            actionList.append([xPos - y, yPos - x])
            
        # Trim the actions list to only account for valid movements
        # print("action list: ", actionList)
        # print("trimmed: " trimmedActions)
        trimmedActions = \
            np.clip(actionList, [0, 0], [self.ScenarioDim[1]-1, self.ScenarioDim[0]-1])
        # print(trimmedActions.shape)
        # normalize
        if len(trimmedActions.shape) == 3:
            trimmedActions = [[a[0, 0], a[1, 0]] for a in trimmedActions]

        # print(trimmedActions)
        trimmedActions = [a for a in trimmedActions if self.DeploymentField[a[0], a[1]] == 0]
    
        # Size normalization Difference

        # Variable to trim actions more by their distance and quadrant
        TrimByDistance = False

        if TrimByDistance:
            diff = len(actionList) - len(trimmedActions)
            # Calculate max action length by shortening the list by a predetermined factor
            maxActionLength = (len(trimmedActions)+diff) // 4

            # Make a list of tuples containing the distance of the action from the foi centroid and the action itself
            distanceAndAction = [(np.sqrt((a[0]-self.foi_centroid[0])**2 + 
                                        (a[1]-self.foi_centroid[1])**2),
                                a) for a in trimmedActions]
            # Sort the (distance, action) tuples by the distance
            sortedActions = sorted(distanceAndAction, key=lambda tup: tup[0])
            # Use the max action length to put the maxActionLength shortest distances
            trimmedActions = [sortedActions[i][1] for i in range(maxActionLength)]
        # print(trimmedActions)

        # Encode the actions
        trimmedActions = [self.EncodeActionMatrix[a[1], a[0]] for a in trimmedActions]
        
        trimMod = self.n_actions % len(trimmedActions)

        # print(len(trimmedActions))
        # print(trimMod)

        # print(trimMod)
        finalActions = []

        trimAmnt = len(trimmedActions)
        while trimAmnt+trimMod < self.n_actions:
            finalActions.extend(trimmedActions)
            trimAmnt = len(finalActions)

        for i in range(trimMod):
            finalActions.append(trimmedActions[i])
    
        # print(np.shape(trimmedActions))

        # trimAmnt = len(trimmedActions)
        # for a in trimmedActions:
        #     while trimAmnt < self.n_actions:
        #         trimmedActions.append(a)
        #         trimAmnt = len(trimmedActions)


        # Return only the trimmed valid movements that are available positions
        # according to the deployment field
        return finalActions
    
    def get_total_actions(self):
        return len(self.get_avail_actions())
    
    def reset(self):
        self.Dataset = self.API.LoadDataset()
        self.Sensors = {}
        self.SenseData = self.Dataset.SensorData
        # Make the sensors a map storing their respective ID and their sensor object
        for id, sensor in enumerate(self.Dataset.Sensors):
            self.Sensors[id] = sensor
            
        
        self.ComputeDeploymentField()
        self.ComputeCoverage()
        
        # Create the desired matrix to contain the desired degress of coverage
        self.DesiredMatrix = np.zeros(self.ScenarioDim, dtype=np.int8)
        for foi in self.Dataset.FOI:
            self.DesiredMatrix[foi.TopLeft.yPos:foi.BotRight.yPos, 
                               foi.TopLeft.xPos:foi.BotRight.xPos] = \
                                   foi.RequiredCoverage
                                   
        return self.get_obs(), self.get_state(), self.get_avail_actions()
    
    def render(self):
        '''
        Function to render an output to visualize what we are doing with 
        our experiments
        !!! NOT IMPLEMENTED !!!
        '''
        pass
    
    def save_replay(self):
        '''
        Save a replay of the experiment conducted
        !!! NOT IMPLEMENTED !!!
        '''
        pass
    
    def ComputeCoverage(self):
        '''
        Method that will compute our coverage matrix based on current sensor data
        and on current sensor positions
        '''
        # Reset coverage matrix to be all zeros
        self.CoverageMatrix = np.zeros(self.ScenarioDim, dtype=np.int8)
        # We will use the disk method from skimage.draw that will help draw a disk of
        # appropriate size and position in our coverage matrix relative to each sensor
        # each sensor will increase the coverage of each point inside its disk by 1 
        for id, sensor in self.Sensors.items():
            x, y = \
                int(sensor.Position.yPos), int(sensor.Position.xPos)
            rr, cc = skd.disk((x, y), self.SenseData.SensingRange)
            rr, cc = np.clip(rr, 0, self.ScenarioDim[0]-1), np.clip(cc, 0, self.ScenarioDim[1]-1)
            self.CoverageMatrix[rr, cc] += 1
                                   
    def ComputeDeploymentField(self):
        '''
        Function to compute deployment field based on sensor positions
        '''
        self.DeploymentField = np.zeros(self.ScenarioDim, dtype=np.int8)
        for sensor in self.Sensors.values():
            x, y = sensor.Position.yPos, sensor.Position.xPos
            self.DeploymentField[x, y] = 1
                          
    def RandomlyDeploySensors(self):
        '''
        Function that can be used to reset the simulation via randomly deploying sensors
        will check for valid indices each time it deploys a sensor
        '''
        for id, sensor in self.Sensors.items():
            validIndices = np.argwhere(self.DeploymentField==0)
            validPos = validIndices[np.random.choice(len(validIndices))]
            self.Sensors[id].Position.yPos, self.Sensors[id].Position.xPos = \
                validPos[0], validPos[1]
            self.DeploymentField[validPos[0], validPos[1]] = 1
        
        self.ComputeDeploymentField()
        self.ComputeCoverage()

    def LoadActionToID(self, ID, action):
        '''
        Function loads the action to the action list
        '''
        assert np.any(np.all(action == self.GetValidActionForID(ID), axis=1)), \
            "Invalid action passed as action"
            
        self.Actions[ID] = action
    