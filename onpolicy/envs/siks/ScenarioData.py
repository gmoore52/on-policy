##################################
#                                # 
# Author: Gavin Moore            #
# Date: 06/23/2023               #
#                                #
# File that contains definitions #
# of all data used in dataset    #
#                                #
##################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#################################
#                               # 
# Define position/shape classes #
#                               #
#################################
class Point():
    def __init__(self, x=0, y=0) -> None:
        '''
        Barebones class that simply packages points so that we can access x and y
        coordinates of any location in space
        '''
        self.xPos = x
        self.yPos = y
    
    def __repr__(self):
        return f"{self.xPos} {self.yPos}"
        
    def __str__(self):
        '''
        Define str method for printing class data
        '''
        return f"Position:\nx = {self.xPos}  y = {self.yPos}"

class Shape():
    def __init__(self, xPos=0, yPos=0) -> None:
        '''
        Barebones shape object that only contains the position of each shape
        can be modified through sub-classes to contain a more well defined shape
        '''
        self.Position = Point(x=xPos, y=yPos)

    
class Rectangle(Shape):
    def __init__(self, xPos=0, yPos=0, xLen=1, yLen=1) -> None:
        super().__init__(xPos, yPos)
        # WIP outline of how the position/lengths should relate 
        # GOAL: Keep position center of shape while adhering to side lengths
        self.TopLeft  = Point(x=xPos-(xLen//2), y=yPos-(yLen//2))
        self.BotRight = Point(x=xPos+(xLen//2), y=yPos+(yLen//2))
        
    
    def __repr__(self):
        return f"{repr(self.Position)} {repr(self.TopLeft)} {repr(self.BotRight)}"
        
    def __str__(self):
        '''
        Define str method for printing class data
        '''
        return f"Rectangle positions: \nTop Left: {self.TopLeft}  Bottom Right: {self.BotRight}"
    
#################################
#                               # 
# End of position/shape classes #
#                               #
#################################
    
#########################################################################################
    
#################################
#                               # 
#   Define Sensor/ROI classes   #
#                               #
#################################
    
    
class SensorData():
    def __init__(self, senseRange=0, commRange=0, moveRange=2) -> None:
        '''
        We assume that sensor data will always consist of a circular 
        sensing/communication range but their relationship is arbitrary
        '''
        self.SensingRange = senseRange
        self.CommRange = commRange
        self.MoveRange = moveRange
        
    def __repr__(self):
        return f"{self.SensingRange} {self.CommRange} {self.MoveRange}"
        
    def __str__(self):
        '''
        Define str method for printing class data
        '''
        return f"Sensor Data:\nSensing Range: {self.SensingRange}\nCommunication Range: {self.CommRange}\nMovement Range: {self.MoveRange}"
        

class Sensor(Shape):
    def __init__(self, xPos=0, yPos=0, startPower = 40) -> None:
        '''
        Inherits from shape so that it can contain position
        also contains a SenseData member that contains sensor data
        RemainingPower is a value representing how much energy the sensor still has
        '''
        super().__init__(xPos=xPos, yPos=yPos)
        self.RemainingPower = startPower
        
    def __repr__(self):
        return f"{repr(self.Position)} {self.RemainingPower}"
        
    
class FieldOfInterest(Rectangle):
    def __init__(self, xPos=0, yPos=0, width=1, height=1, reqCoverage=4) -> None:
        '''
        Object that will essentially define a shape with the property that it will
        be considered differently than other shapes as it is the "region to be covered"
        '''
        super().__init__(xPos=xPos, yPos=yPos, xLen=width, yLen=height)
        self.RequiredCoverage = reqCoverage
        
    def __repr__(self):
        return super().__repr__()+f" {self.RequiredCoverage}"