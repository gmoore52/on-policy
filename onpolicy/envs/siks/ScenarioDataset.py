##################################
#                                # 
# Author: Gavin Moore            #
# Date: 06/23/2023               #
#                                #
# File that contains the dataset #
# to be used to store scenarios  #
#                                #
##################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import ScenarioData as SData
import os
import json

class ScenarioDataset():
    def __init__(self, areaSize=(64,64)):
    
        self.Size = areaSize
        self.SensorData = SData.SensorData()
        self.Obstacles = []
        self.FOI = []
        self.Sensors = []
        
    
    def SetSize(self, param=(64,64)):
        self.Size = param
    
    def SetSensorData(self, param: SData.SensorData):
        self.SensorData = param
            
    def AddSensor(self, param: SData.Sensor):
        self.Sensors.append(param)
    
    def AddObstacle(self, param: SData.Shape):
        self.Obstacles.append(param)
    
    def AddFieldOfInterest(self, param: SData.FieldOfInterest):
        self.FOI.append(param)
        
class ScenarioDatasetAPI():
    def __init__(self, filename: str ="default.json", filepath: str =os.path.curdir):
        '''
        @ARGS
        filename => Defines the filename that the file will be stored under 
        '''
        self.Filepath = filepath
        self.FileName = os.path.join(self.Filepath, filename)
            
        assert self.FileName[-5:] == ".json", \
            f"Invalid filename, please use .json extension; Set filename is {filename}"
            
        assert self.FileName != "default.json", \
            "Filename is set to default, please pass a filename on initilization of dataset API"
            
            
        self.Dataset = ScenarioDataset()
        self.DatasetDict = {}
        
        if os.path.exists(self.FileName):
            self.Dataset = self.LoadDataset()
        else:
            self.StoreDataset()
            
            
    def LoadDataset(self):
        result = ScenarioDataset()
        with open(self.FileName, 'r') as infile:
            resultDict = json.load(infile)
            infile.close()
            
        result = self.SerializeDataset(resultDict)
        
        return result
    
    def StoreDataset(self):
        self.DatasetDict = self.DatasetToDict()
        with open(self.FileName, 'w') as outfile:
            json.dump(self.DatasetDict, outfile)
            outfile.close()
            
    def SerializeDataset(self, dictionary):
        result = ScenarioDataset()
        # Load Size
        result.Size = dictionary["size"]
        # Load SensorData
        sense_data = dictionary["sensor_data"].split()
        result.SensorData = SData.SensorData(senseRange=int(sense_data[0]),
                                             commRange=int(sense_data[1]),
                                             moveRange=int(sense_data[2]))
        # Load Sensors
        sensors = [a.split() for a in dictionary["sensors"]]
        final_sensors = [SData.Sensor(xPos=int(a[0]),
                                      yPos=int(a[1]),
                                      startPower=int(a[2]))
                         for a in sensors]
        result.Sensors = final_sensors
        # Load FOIs
        foi = [a.split() for a in dictionary["foi"]]
        final_foi = []
        for f in foi:
            temp = SData.FieldOfInterest()
            temp.Position = SData.Point(x=int(f[0]), y = int(f[1]))
            temp.TopLeft  = SData.Point(x=int(f[2]), y = int(f[3]))
            temp.BotRight = SData.Point(x=int(f[4]), y = int(f[5]))
            temp.RequiredCoverage = int(f[6])
            final_foi.append(temp)
        result.FOI = final_foi
    
        return result
              
    def DatasetToDict(self):
        tempDict = {"size": self.Dataset.Size,
                    "sensor_data": repr(self.Dataset.SensorData),
                    "sensors": [],
                    "foi": []}
        
        
        for sensor in self.Dataset.Sensors:
            tempDict["sensors"].append(repr(sensor))
            
        for foi in self.Dataset.FOI:
            tempDict["foi"].append(repr(foi))
        
        return tempDict        
            
    def SetFilepath(self, filepath: str):
        self.Filepath = filepath
            
    def SetFilename(self, filename: str):
        assert self.FileName[-5:] == ".json", \
            f"Invalid filename, please use .json extension; Set filename is {filename}"
        self.FileName = os.path.join(self.Filepath, filename)
            
    def SetDataset(self, dataset: ScenarioDataset):
        self.Dataset = dataset