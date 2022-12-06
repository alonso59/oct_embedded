
#Import System Util
import sys, os, time, platform
# Import PySide6 classes
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
## Torch
#import torch
#from PIL import Image
#from PIL.ImageQt import ImageQt
#import numpy as np
## import OCT
#from oct_library import OCTProcessing
#from jtop import jtop
#from torch2trt import TRTModule

class EtdrsGridItem(QGraphicsItem):
    def __init__(self, center: QPointF(), centralRad: float, innerRad: float, outerRad: float):
        super().__init__()
        self.penWidth= 1.0
        self.debugLines = False
        self.centralRingCenter =  center
        self.centralRingRadius = centralRad
        self.innerRingRadius = innerRad
        self.outerRingRadius = outerRad
        print(f"NEW ITEM, BOUNDING RECT : {self.boundingRect()}, CIO-RAD: {self.centralRingRadius} {self.innerRingRadius} {self.outerRingRadius} ")
        database = QFontDatabase()
        fontTree = QTreeWidget()
        fontTree.setColumnCount(2)

        for family in database.families():
            print(family)

    def boundingRect(self):
        return QRectF(  self.centralRingCenter.x()-self.outerRingRadius, 
                        self.centralRingCenter.y()-self.outerRingRadius, 
                        self.outerRingRadius * 2, 
                        self.outerRingRadius * 2)

    def paint(self, painter, option, widget):
        rect = self.boundingRect()
        textFont = QFont("Comic Sans MS", 60, QFont.Bold)

        painter.setFont(textFont)
        """
        #Draw Outer Ring
        painter.drawEllipse(self.centralRingCenter, self.outerRingRadius, self.outerRingRadius)
        #Draw Inner Ring
        painter.drawEllipse(self.centralRingCenter, self.innerRingRadius, self.innerRingRadius)
        #Draw Central Ring
        painter.drawEllipse(self.centralRingCenter, self.centralRingRadius, self.centralRingRadius)
        """
        linePen = QPen(Qt.blue, 3)
        painter.setPen(linePen)

        #Draw Most Center Text
        #painter.drawStaticText(self.centralRingCenter,QStaticText("C"))
        painter.drawText(self.boundingRect(), Qt.AlignCenter, "C0")
        
        #Draw Outer Donut
        path = QPainterPath()
        color = QColor(Qt.yellow)
        color.setAlpha(50)
        path.addEllipse(self.centralRingCenter,self.outerRingRadius,self.outerRingRadius)
        path.addEllipse(self.centralRingCenter,self.innerRingRadius,self.innerRingRadius)
        painter.setBrush(color)
        painter.drawPath(path)

        #Draw Inner Donut
        path = QPainterPath()
        color = QColor(Qt.yellow)
        color.setAlpha(50)
        path.addEllipse(self.centralRingCenter,self.innerRingRadius,self.innerRingRadius)
        path.addEllipse(self.centralRingCenter,self.centralRingRadius,self.centralRingRadius)
        painter.setBrush(color)
        painter.drawPath(path)   

        #Draw Central Circle
        color = QColor(Qt.red)
        color.setAlpha(50)
        painter.setBrush(color) 
        painter.drawEllipse(self.centralRingCenter, self.centralRingRadius, self.centralRingRadius)

        #Draw Lines
        CONSTANT_45D = (1.4142/2)
        path = QPainterPath()
        path.moveTo(self.centralRingRadius * CONSTANT_45D + self.centralRingCenter.x(),
                    self.centralRingRadius * CONSTANT_45D + self.centralRingCenter.y())
        path.lineTo(self.outerRingRadius * CONSTANT_45D + self.centralRingCenter.x(), 
                    self.outerRingRadius * CONSTANT_45D + self.centralRingCenter.y())
        painter.drawPath(path)

        path = QPainterPath()
        path.moveTo(self.centralRingRadius * -CONSTANT_45D + self.centralRingCenter.x(),
                    self.centralRingRadius * CONSTANT_45D + self.centralRingCenter.y())
        path.lineTo(self.outerRingRadius * -CONSTANT_45D + self.centralRingCenter.x(), 
                    self.outerRingRadius * CONSTANT_45D + self.centralRingCenter.y())
        painter.drawPath(path)

        path = QPainterPath()
        path.moveTo(self.centralRingRadius * CONSTANT_45D + self.centralRingCenter.x(), 
                    self.centralRingRadius * -CONSTANT_45D + self.centralRingCenter.y())
        path.lineTo(self.outerRingRadius * CONSTANT_45D + self.centralRingCenter.x(), 
                    self.outerRingRadius * -CONSTANT_45D + self.centralRingCenter.y())
        painter.drawPath(path)

        path = QPainterPath()
        path.moveTo(self.centralRingRadius * -CONSTANT_45D + self.centralRingCenter.x(), 
                    self.centralRingRadius * -CONSTANT_45D + self.centralRingCenter.y())
        path.lineTo(self.outerRingRadius * -CONSTANT_45D + self.centralRingCenter.x(), 
                    self.outerRingRadius * -CONSTANT_45D + self.centralRingCenter.y())
        painter.drawPath(path)


        painter.setBrush(Qt.NoBrush)
        #Draw Bounding rects
            #x,y,w,h
        # INNER LEFT-RIGHT
        centralInnerDifference = self.innerRingRadius - self.centralRingRadius
        innerRightRect = QRectF(self.centralRingRadius + self.centralRingCenter.x(), 
                                -centralInnerDifference / 2  + self.centralRingCenter.y(), 
                                centralInnerDifference, 
                                centralInnerDifference)

        innerLeftRect = QRectF(-self.innerRingRadius + self.centralRingCenter.x(), 
                                -centralInnerDifference / 2  + self.centralRingCenter.y(), 
                                centralInnerDifference, 
                                centralInnerDifference)

        # INNER UP-DOWN
        painter.setPen(Qt.magenta)
        centralInnerDifference = self.innerRingRadius - self.centralRingRadius
        innerUpRect = QRectF(-centralInnerDifference / 2 + self.centralRingCenter.x(), 
                                -self.innerRingRadius  + self.centralRingCenter.y(), 
                                centralInnerDifference, 
                                centralInnerDifference)

        innerDownRect = QRectF(-centralInnerDifference / 2 + self.centralRingCenter.x(), 
                                self.centralRingRadius  + self.centralRingCenter.y(), 
                                centralInnerDifference, 
                                centralInnerDifference)

        # OUTER LEFT-RIGHT
        painter.setPen(Qt.red)
        innerOuterDifference = self.outerRingRadius - self.innerRingRadius
        outerRightRect = QRectF(self.innerRingRadius + self.centralRingCenter.x(),
                                -innerOuterDifference / 2 + self.centralRingCenter.y(), 
                                innerOuterDifference, 
                                innerOuterDifference)

        outerLeftRect = QRectF(-self.outerRingRadius + self.centralRingCenter.x(),
                                -innerOuterDifference / 2 + self.centralRingCenter.y(), 
                                innerOuterDifference, 
                                innerOuterDifference)

        # OUTER UP- DOWN
        painter.setPen(Qt.green)
        innerOuterDifference = self.outerRingRadius - self.innerRingRadius

        outerUpRect = QRectF(-innerOuterDifference / 2 + self.centralRingCenter.x(), 
                                -self.outerRingRadius + self.centralRingCenter.y() , 
                                innerOuterDifference, 
                                innerOuterDifference)

        outerDownRect = QRectF(-innerOuterDifference / 2 + self.centralRingCenter.x(),
                                 self.innerRingRadius + self.centralRingCenter.y(), 
                                 innerOuterDifference, 
                                 innerOuterDifference)

        if self.debugLines == True:    
            #Draw Text Bounding Rect helper
            painter.drawRect(innerRightRect)
            painter.drawRect(innerLeftRect)
            painter.drawRect(innerUpRect)
            painter.drawRect(innerDownRect)
            painter.drawRect(outerRightRect)
            painter.drawRect(outerLeftRect)            
            painter.drawRect(outerUpRect)
            painter.drawRect(outerDownRect)
            #Draw Middle Outer-Inner Ring helper
            painter.setPen(Qt.cyan)
            helperRadius = (self.outerRingRadius + self.innerRingRadius) / 2
            painter.drawEllipse(self.centralRingCenter, helperRadius, helperRadius)
            #Draw Middle Inner-Center Ring helper     
            helperRadius = (self.innerRingRadius + self.centralRingRadius) / 2
            painter.drawEllipse(self.centralRingCenter, helperRadius, helperRadius)
            #Draw bounding Rect
            painter.setPen(Qt.red)
            painter.drawRect(rect)
        else:
            ...

        painter.setPen(Qt.blue)
        painter.drawText(innerRightRect, Qt.AlignCenter, "N1")
        painter.drawText(innerLeftRect, Qt.AlignCenter, "T1")
        painter.drawText(innerUpRect, Qt.AlignCenter, "S1")
        painter.drawText(innerDownRect, Qt.AlignCenter, "I1")
        painter.drawText(outerRightRect, Qt.AlignCenter, "N2")
        painter.drawText(outerLeftRect, Qt.AlignCenter, "T2")
        painter.drawText(outerUpRect, Qt.AlignCenter, "S2")
        painter.drawText(outerDownRect, Qt.AlignCenter, "I2")

        self.update()

    def modifyAll(self, center: QPointF(), centralRad: float, innerRad: float, outerRad: float):
        self.centralRingCenter =  center
        self.centralRingRadius = centralRad
        self.innerRingRadius = innerRad
        self.outerRingRadius = outerRad   

    def modifyCentralRadius(self, centralRad: float):
        self.centralRingRadius = centralRad

    def modifyInnerRadius(self, innerRad: float):
        self.innerRingRadius = innerRad      

    def modifyOuterRadius(self, outerRad: float):
        self.outerRingRadius = outerRad   

    def modifyXPos(self, xPos: float):
        self.centralRingCenter.setX(xPos)

    def modifyYPos(self, yPos: float):
        self.centralRingCenter.setY(yPos)
    
    def modifyCenter(self, center: QPointF()):
        self.centralRingCenter = center

def setCentralLabelValue():
    value = centralRSlider.value()
    print(f"Central changed, {value}")
    centralRValue.setText(str(value))
    etdrsGrid.modifyCentralRadius(value)
    graphicsScene.update()

def setInnerLabelValue():
    value = innerRSlider.value()
    print(f"Inner changed, {value}")
    innerRValue.setText(str(value))
    etdrsGrid.modifyInnerRadius(value)
    graphicsScene.update()

def setOuterLabelValue():
    value = outerRSlider.value()
    print(f"Outer changed, {value}")
    outerRValue.setText(str(value))
    etdrsGrid.modifyOuterRadius(value)
    graphicsScene.update()

def setXPosLabelValue():
    value = positionXSlider.value()
    print(f"X changed, {value}")
    positionXValue.setText(str(value))
    etdrsGrid.modifyXPos(value)
    graphicsScene.update()

def setYPosLabelValue():
    value = positionYSlider.value()
    print(f"Y changed, {value}")
    positionYValue.setText(str(value))
    etdrsGrid.modifyYPos(value)
    graphicsScene.update()

if __name__ == "__main__":
    RAD_SLIDER_MAX = 500
    CENTRAL_RADIUS = 100
    INNER_RADIUS = 200
    OUTER_RADIUS = 400
    POS_SLIDER_MIN = 0
    POS_SLIDER_MAX = 2000
    XPOS = 1000
    YPOS = 1000

    app = QApplication()
    mainWindow = QMainWindow()
    mainWindow.resize(1600,900)
    graphicsScene = QGraphicsScene()
    graphicsView = QGraphicsView()
    graphicsView.setScene(graphicsScene)

    centralWidget = QWidget(mainWindow)

    mainWindow.setCentralWidget(centralWidget)
    layout = QGridLayout(centralWidget)
    
    layout.addWidget(graphicsView)

    etdrsGrid = EtdrsGridItem(QPoint(XPOS,YPOS), CENTRAL_RADIUS, INNER_RADIUS, OUTER_RADIUS)
    rectangleBackground = QGraphicsRectItem(0,0,2000,2000)
    graphicsScene.addItem(rectangleBackground)
    graphicsScene.addItem(etdrsGrid)
    graphicsView.fitInView(rectangleBackground, Qt.KeepAspectRatio)


    sliderWidget = QWidget()
    sliderLayout = QGridLayout()
    sliderWidget.setLayout(sliderLayout)

    centralRSlider = QSlider()
    centralRSlider.setOrientation(Qt.Horizontal)
    centralRSlider.setMinimum(0)
    centralRSlider.setMaximum(RAD_SLIDER_MAX)
    centralRSlider.setTickPosition(QSlider.TicksBelow)
    centralRSlider.setTickInterval(10)
    centralRSlider.setValue(CENTRAL_RADIUS)

    innerRSlider = QSlider()
    innerRSlider.setOrientation(Qt.Horizontal)
    innerRSlider.setMinimum(0)
    innerRSlider.setMaximum(RAD_SLIDER_MAX)
    innerRSlider.setTickPosition(QSlider.TicksBelow)
    innerRSlider.setTickInterval(10)
    innerRSlider.setSingleStep(10)
    innerRSlider.setValue(INNER_RADIUS)

    outerRSlider = QSlider()
    outerRSlider.setOrientation(Qt.Horizontal)
    outerRSlider.setMinimum(0)
    outerRSlider.setMaximum(RAD_SLIDER_MAX)
    outerRSlider.setTickPosition(QSlider.TicksBelow)
    outerRSlider.setTickInterval(10)
    outerRSlider.setValue(OUTER_RADIUS)

    positionXSlider = QSlider()
    positionXSlider.setOrientation(Qt.Horizontal)
    positionXSlider.setMinimum(POS_SLIDER_MIN)
    positionXSlider.setMaximum(POS_SLIDER_MAX)
    positionXSlider.setTickPosition(QSlider.TicksBelow)
    positionXSlider.setTickInterval(10)
    positionXSlider.setSingleStep(10)
    positionXSlider.setValue(XPOS)

    positionYSlider = QSlider()
    positionYSlider.setOrientation(Qt.Horizontal)
    positionYSlider.setMinimum(POS_SLIDER_MIN)
    positionYSlider.setMaximum(POS_SLIDER_MAX)
    positionYSlider.setTickPosition(QSlider.TicksBelow)
    positionYSlider.setTickInterval(10)
    positionYSlider.setSingleStep(10)
    positionYSlider.setValue(YPOS)

    centralRLabel = QLabel()
    centralRLabel.setText("Center Radius")
    innerRLabel = QLabel()
    innerRLabel.setText("Inner Radius")
    outerRLabel = QLabel()
    outerRLabel.setText("Outer Radius")
    positionXLabel = QLabel()
    positionXLabel.setText("X Pos")
    positionYLabel = QLabel()
    positionYLabel.setText("Y Pos")

    centralRValue = QLabel()
    centralRValue.setText(str(CENTRAL_RADIUS))
    centralRValue.setFixedSize(100,10)
    innerRValue = QLabel()
    innerRValue.setText(str(INNER_RADIUS))   
    innerRValue.setFixedSize(100,10)
    outerRValue = QLabel()
    outerRValue.setText(str(OUTER_RADIUS))
    outerRValue.setFixedSize(100,10)
    positionXValue = QLabel()
    positionXValue.setText(str(XPOS))
    positionXValue.setFixedSize(100,10)
    positionYValue = QLabel()
    positionYValue.setText(str(YPOS))
    positionYValue.setFixedSize(100,10)

    # First Column
    sliderLayout.addWidget(centralRLabel, 0, 0, 1, 1)
    sliderLayout.addWidget(innerRLabel, 1, 0, 1, 1)
    sliderLayout.addWidget(outerRLabel, 2, 0, 1, 1)
    sliderLayout.addWidget(positionXLabel, 3, 0, 1, 1)
    sliderLayout.addWidget(positionYLabel, 4, 0, 1, 1)

    #Second Column
    sliderLayout.addWidget(centralRSlider, 0, 1, 1, 1)
    sliderLayout.addWidget(innerRSlider, 1, 1, 1, 1)
    sliderLayout.addWidget(outerRSlider, 2, 1, 1, 1)
    sliderLayout.addWidget(positionXSlider, 3, 1, 1, 1)
    sliderLayout.addWidget(positionYSlider, 4, 1, 1, 1)

    #Third Column
    sliderLayout.addWidget(centralRValue, 0, 2, 1, 1)
    sliderLayout.addWidget(innerRValue, 1, 2, 1, 1)
    sliderLayout.addWidget(outerRValue, 2, 2, 1, 1)
    sliderLayout.addWidget(positionXValue, 3, 2, 1, 1)
    sliderLayout.addWidget(positionYValue, 4, 2, 1, 1)

    layout.addWidget(sliderWidget)

    centralRSlider.valueChanged.connect(setCentralLabelValue)
    innerRSlider.valueChanged.connect(setInnerLabelValue)
    outerRSlider.valueChanged.connect(setOuterLabelValue)
    positionXSlider.valueChanged.connect(setXPosLabelValue)
    positionYSlider.valueChanged.connect(setYPosLabelValue)


    mainWindow.show()
    app.exec_()

"""
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

#PySide6 port of the Nested Donuts example from Qt v5.x

import sys

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication, QGridLayout, QWidget
from PySide6.QtCharts import QChart, QChartView, QPieSeries, QPieSlice


from random import randrange
from functools import partial


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.donuts = []
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart = self.chart_view.chart()
        self.chart.legend().setVisible(False)
        self.chart.setTitle("Nested donuts demo")
        self.chart.setAnimationOptions(QChart.AllAnimations)

        self.min_size = 0.1
        self.max_size = 0.9
        self.donut_count = 5

        self.setup_donuts()

        # create main layout
        self.main_layout = QGridLayout(self)
        self.main_layout.addWidget(self.chart_view, 1, 1)
        self.setLayout(self.main_layout)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_rotation)
        self.update_timer.start(1250)

    def setup_donuts(self):
        for i in range(self.donut_count):
            donut = QPieSeries()
            slccount = randrange(3, 6)
            for j in range(slccount):
                value = randrange(100, 200)

                slc = QPieSlice(str(value), value)
                slc.setLabelVisible(True)
                slc.setLabelColor(Qt.white)
                slc.setLabelPosition(QPieSlice.LabelInsideTangential)

                # Connection using an extra parameter for the slot
                slc.hovered[bool].connect(partial(self.explode_slice, slc=slc))

                donut.append(slc)
                size = (self.max_size - self.min_size) / self.donut_count
                donut.setHoleSize(self.min_size + i * size)
                donut.setPieSize(self.min_size + (i + 1) * size)

            self.donuts.append(donut)
            self.chart_view.chart().addSeries(donut)

    @Slot()
    def update_rotation(self):
        for donut in self.donuts:
            phase_shift = randrange(-50, 100)
            donut.setPieStartAngle(donut.pieStartAngle() + phase_shift)
            donut.setPieEndAngle(donut.pieEndAngle() + phase_shift)

    def explode_slice(self, exploded, slc):
        if exploded:
            self.update_timer.stop()
            slice_startangle = slc.startAngle()
            slice_endangle = slc.startAngle() + slc.angleSpan()

            donut = slc.series()
            idx = self.donuts.index(donut)
            for i in range(idx + 1, len(self.donuts)):
                self.donuts[i].setPieStartAngle(slice_endangle)
                self.donuts[i].setPieEndAngle(360 + slice_startangle)
        else:
            for donut in self.donuts:
                donut.setPieStartAngle(0)
                donut.setPieEndAngle(360)

            self.update_timer.start()

        slc.setExploded(exploded)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec())

"""