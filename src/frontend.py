#Import System Util
import sys, os, time, platform
# Import PySide6 classes
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
# Torch
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np
# import OCT
from oct_library import OCTProcessing
from jtop import jtop

class ImageViewer(QGraphicsView):
	def __init__(self):
		super(ImageViewer, self).__init__()
		self.zoom = 0

	def keyPressEvent(self, event):
		print("pressed key in ImageViewer")
		if event.key() == Qt.Key_F:
			print("F was pressed")
			self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
		elif event.key() == Qt.Key_W:
			print("W was pressed")
			self.fitInView(self.scene().sceneRect())
		else:
			...

	def wheelEvent(self, event):
		if event.delta() >0:
			factor = 1.25
			self.zoom +=1
		else:
			factor = 0.8
			self.zoom -=1
		if self.zoom > 0:
			self.scale(factor, factor)
		elif self.zoom == 0:
			self.fitInView()
		else:
			self.zoom = 0

class PropertiesHolder(QWidget):
	def __init__(self):
		super().__init__()
		#GENERAL METADATA
		#-> AI Model
		#-> File
		#-> Patient ID
		#-> Visit Date
		#-> Scan Pos
		#-> Scale X
		#-> Scale Z
		#-> Size XSlo
		#-> Size YSlo
		#-> Scale XSlo
		#-> Scale YSlo
		#-> No. Bscans
		# FOVEA BSCAN METADATA
		#-> # Bscan
		#-> Start X
		#-> End X
		self.setFixedWidth(400)
		self.thisLayout = QVBoxLayout()
		self.setLayout(self.thisLayout)
		self.titleLabel = QLabel()
		self.titleLabel.setText("SLO")
		self.titleLabel.setFixedWidth(400)
		self.titleLabel.setFixedHeight(20)
		self.titleLabel.setAlignment(Qt.AlignCenter)
		self.titleLabel.setStyleSheet("background-color:rgb(70,70,70); color:rgb(255,255,255)")

		self.graphicsScene = QGraphicsScene()
		self.graphicsView = ImageViewer()
		self.graphicsView.setStyleSheet("background-color:rgb(0,0,0)")
		#self.graphicsView.setFixedHeight(400)
		#self.graphicsView.setFixedWidth(400)
		self.graphicsView.setFixedSize(400,400)
		self.graphicsPixmapItem = QGraphicsPixmapItem()
		self.graphicsScene.addItem(self.graphicsPixmapItem)
		self.graphicsView.setScene(self.graphicsScene)

		#GENERAL METADATA
		#-> AI Model
		#-> File
		#-> Patient ID
		#-> Visit Date
		#-> Scan Pos
		#-> Scale X
		#-> Scale Z
		#-> Size XSlo
		#-> Size YSlo
		#-> Scale XSlo
		#-> Scale YSlo
		#-> No. Bscans
		self.generalMetadataTitle = QLabel()
		self.generalMetadataTitle.setText("General Metadata")
		self.generalMetadataTitle.setFixedWidth(400)
		self.generalMetadataTitle.setFixedHeight(20)
		self.generalMetadataTitle.setAlignment(Qt.AlignCenter)
		self.generalMetadataTitle.setStyleSheet("background-color:rgb(70,70,70); color:rgb(255,255,255)")
		self.generalMetadataStandardModel = QStandardItemModel()
		self.generalMetadataStandardModel.appendRow([QStandardItem("AI Model"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("File"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Patient ID"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Visit Date"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Scan Pos"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Scale X"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Scale Z"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Size XSlo"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Size YSlo"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Scale XSlo"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("Scale YSlo"), QStandardItem("")])
		self.generalMetadataStandardModel.appendRow([QStandardItem("No. Bscans"), QStandardItem("")])
		self.generalMetadataTableView = QTableView()
		self.generalMetadataTableView.setModel(self.generalMetadataStandardModel)
		self.generalMetadataTableView.setFixedWidth(400)
		self.generalMetadataTableView.horizontalHeader().setVisible(False)
		self.generalMetadataTableView.verticalHeader().setVisible(False)
		#self.generalMetadataTableView.setColumnWidth(0,80)
		#self.generalMetadataTableView.setColumnWidth(1,10)
		self.generalMetadataTableView.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
		self.generalMetadataTableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
		self.generalMetadataTableView.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)


		# FOVEA BSCAN METADATA
		#-> # Bscan
		#-> Start X
		#-> End X
		self.foveaMetadataTitle = QLabel()
		self.foveaMetadataTitle.setText("Fovea Bscan Metadata")
		self.foveaMetadataTitle.setFixedWidth(400)
		self.foveaMetadataTitle.setFixedHeight(20)
		self.foveaMetadataTitle.setAlignment(Qt.AlignCenter)
		self.foveaMetadataTitle.setStyleSheet("background-color:rgb(70,70,70); color:rgb(255,255,255)")
		self.foveaMetadataStandardModel = QStandardItemModel()
		self.foveaMetadataStandardModel.appendRow([QStandardItem("Bscan"), QStandardItem("")])
		self.foveaMetadataStandardModel.appendRow([QStandardItem("Start X"), QStandardItem("")])
		self.foveaMetadataStandardModel.appendRow([QStandardItem("End X"), QStandardItem("")])
		self.foveaMetadataTableView = QTableView()
		self.foveaMetadataTableView.setModel(self.foveaMetadataStandardModel)
		self.foveaMetadataTableView.horizontalHeader().setVisible(False)
		self.foveaMetadataTableView.verticalHeader().setVisible(False)
		self.foveaMetadataTableView.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
		self.foveaMetadataTableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
		self.foveaMetadataTableView.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

		self.foveaMetadataTableView.setFixedWidth(400)
		self.foveaMetadataTableView.setFixedHeight(100)


		self.thisLayout.addWidget(self.titleLabel)	
		self.thisLayout.addWidget(self.graphicsView)
		self.thisLayout.addWidget(self.generalMetadataTitle)
		self.thisLayout.addWidget(self.generalMetadataTableView)
		self.thisLayout.addWidget(self.foveaMetadataTitle)
		self.thisLayout.addWidget(self.foveaMetadataTableView)

	def storeImage(self, pixmap):
		self.graphicsPixmapItem.setPixmap(pixmap)
		...

	def setMetaData(self):
		...


class ImageHolder(QWidget):
	def __init__(self, title):
		super().__init__()
		self.thisLayout = QVBoxLayout()
		self.setLayout(self.thisLayout)

		self.savedPixmap = QPixmap()

		self.titleLabel = QLabel()
		self.titleLabel.setText(title)
		self.titleLabel.setFixedHeight(20)
		self.titleLabel.setAlignment(Qt.AlignCenter)
		self.titleLabel.setStyleSheet("background-color:rgb(70,70,70); color:rgb(255,255,255)")
		self.imgLabel = QLabel()
		self.imgLabel.setStyleSheet("background-color:rgb(0,0,0)")

		self.graphicsScene = QGraphicsScene()

		self.graphicsView = ImageViewer()
		#self.graphicsView.setFixedHeight(90)
		#self.graphicsView.setFixedWidth(240)
		self.graphicsPixmapItem = QGraphicsPixmapItem()
		self.graphicsScene.addItem(self.graphicsPixmapItem)
		self.graphicsView.setScene(self.graphicsScene)

		self.thisLayout.addWidget(self.titleLabel)
		self.thisLayout.addWidget(self.graphicsView)

	def setImage(self, pixmap):
		w = self.imgLabel.width()
		h = self.imgLabel.height()
		self.savedPixmap = pixmap
		self.imgLabel.setPixmap(self.savedPixmap.scaled(w,h, Qt.KeepAspectRatio))
		...

	def storeImage(self, pixmap):
		self.graphicsPixmapItem.setPixmap(pixmap)
		...

	def displayImage():
		...


class Ui_MainWindow(object):
	def setupUi(self, MainWindow: QMainWindow):
		if MainWindow.objectName():
			MainWindow.setObjectName(u"MainWindow")
		MainWindow.resize(1366,768)
		MainWindow.setMinimumSize(1366,768)


		self.centralWidget = QWidget(MainWindow)
		self.centralWidget.setObjectName(u"centralWidget")
		#centralWidgetStyleSheet = open(r"src\frontend_gui\frontend_stylesheet.css","r")
		#self.centralWidget.setStyleSheet(centralWidgetStyleSheet.read())

		self.horizontalLayout = QHBoxLayout(self.centralWidget)
		self.horizontalLayout.setObjectName(u"horizontalLayout")

		self.verticalLayout = QVBoxLayout()
		self.verticalLayout.setObjectName(u"verticalLayout")

		# self.topLabelImage = QLabel()
		# self.topLabelImage.setObjectName(u"topLabelImage")
		# self.topLabelImage.setStyleSheet("background-color:rgb(255,255,150)")
		# self.bottomLabelImage = QLabel()
		# self.bottomLabelImage.setObjectName(u"bottomLabelImage")
		# self.bottomLabelImage.setStyleSheet("background-color:rgb(255,255,150)")

		self.propertiesP = PropertiesHolder()
		self.topLabelImg = ImageHolder("B-Scan Centra Fovea")		
		self.middleLabelImg = ImageHolder("B-Scan Output Model")
		self.bottomLabelImg = ImageHolder("B-Scan Layer Segmentation")

		self.verticalLayout.addWidget(self.topLabelImg)
		self.verticalLayout.addWidget(self.middleLabelImg)
		self.verticalLayout.addWidget(self.bottomLabelImg)

		self.mainMenuBar = QMenuBar(MainWindow)
		self.mainMenuBar.setObjectName(u"mainMenuBar")
		self.mainMenuBar.setGeometry(QRect(0,0,800,21))
		MainWindow.setMenuBar(self.mainMenuBar)
		self.mainMenuBar.setNativeMenuBar(False)

		self.menuFile = QMenu(self.mainMenuBar)
		self.menuFile.setObjectName(u"menuFile")

		self.menuEdit = QMenu(self.mainMenuBar)
		self.menuEdit.setObjectName(u"menuEdit")

		self.menuOptions = QMenu(self.mainMenuBar)
		self.menuOptions.setObjectName(u"menuOptions")

		self.menuHelp = QMenu(self.mainMenuBar)
		self.menuHelp.setObjectName(u"menuHelp")

		self.toolBar = QToolBar(MainWindow)
		self.toolBar.setObjectName(u"toolBar")
		self.toolBar.setMovable(False)
		MainWindow.addToolBar(Qt.RightToolBarArea, self.toolBar)
								#A    #RNFL #GCLIPL  #INL   #OPL   #ONL   #IS    #OS    #RPE
		self.buttonBooleans = [False, False, False, False, False, False, False, False, False]
		DefaultPushButtonStyleSheet = "QPushButton {background-color:red;} QPushButton:checked{background-color:green;}"
		self.rnflAction = QPushButton()
		self.rnflAction.setObjectName(u"rnflAction")
		self.rnflAction.setCheckable(True)
		self.rnflAction.setChecked(False)
		self.rnflAction.setStyleSheet(DefaultPushButtonStyleSheet)

		self.gcliplAction = QPushButton()
		self.gcliplAction.setObjectName(u"gcliplAction")
		self.gcliplAction.setCheckable(True)
		self.gcliplAction.setChecked(False)
		self.gcliplAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.inlAction = QPushButton()
		self.inlAction.setObjectName(u"inlAction")
		self.inlAction.setCheckable(True)
		self.inlAction.setChecked(False)
		self.inlAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.oplAction = QPushButton()
		self.oplAction.setObjectName(u"oplAction")
		self.oplAction.setCheckable(True)
		self.oplAction.setChecked(False)
		self.oplAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.onlAction = QPushButton()
		self.onlAction.setObjectName(u"onlAction")
		self.onlAction.setCheckable(True)
		self.onlAction.setChecked(False)
		self.onlAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.isAction = QPushButton()
		self.isAction.setObjectName(u"isAction")
		self.isAction.setCheckable(True)
		self.isAction.setChecked(False)
		self.isAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.osAction = QPushButton()
		self.osAction.setObjectName(u"ezAction")
		self.osAction.setCheckable(True)
		self.osAction.setChecked(False)
		self.osAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.rpeAction = QPushButton()
		self.rpeAction.setObjectName(u"rpeAction")
		self.rpeAction.setCheckable(True)
		self.rpeAction.setChecked(False)
		self.rpeAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.allAction = QPushButton()
		self.allAction.setObjectName(u"allAction")
		self.allAction.setCheckable(True)
		self.allAction.setChecked(False)
		self.allAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.loadAction = QAction()
		self.loadAction.setObjectName(u"loadAction")

		self.saveAction = QAction()
		self.saveAction.setObjectName(u"saveAction")

		self.finishAction = QAction()
		self.finishAction.setObjectName(u"finishAction")

		self.actionOpenFile = QAction(MainWindow)
		self.actionOpenFile.setObjectName(u"actionOpenFile")
		self.menuFile.addAction(self.actionOpenFile)

		self.actionClose = QAction(MainWindow)
		self.actionClose.setObjectName(u"actionClose")
		self.menuFile.addAction(self.actionClose)

		self.actionSwitchTheme = QAction(MainWindow)
		self.actionSwitchTheme.setObjectName(u"actionSwitchTheme")
		self.menuOptions.addAction(self.actionSwitchTheme)

		self.allOpButtons = []
		self.allOpButtons.append(self.rnflAction)
		self.allOpButtons.append(self.gcliplAction)
		self.allOpButtons.append(self.inlAction)
		self.allOpButtons.append(self.oplAction)
		self.allOpButtons.append(self.onlAction)
		self.allOpButtons.append(self.isAction)
		self.allOpButtons.append(self.osAction)
		self.allOpButtons.append(self.rpeAction)

		self.toolBar.addWidget(self.rnflAction)
		self.toolBar.addWidget(self.gcliplAction)
		self.toolBar.addWidget(self.inlAction)
		self.toolBar.addWidget(self.oplAction)
		self.toolBar.addWidget(self.onlAction)
		self.toolBar.addWidget(self.isAction)
		self.toolBar.addWidget(self.osAction)
		self.toolBar.addWidget(self.rpeAction)
		self.toolBar.addWidget(self.allAction)
		self.toolBar.addSeparator()
		self.toolBar.addAction(self.loadAction)
		self.toolBar.addAction(self.saveAction)
		self.toolBar.addAction(self.finishAction)

		self.mainMenuBar.addAction(self.menuFile.menuAction())
		self.mainMenuBar.addAction(self.menuEdit.menuAction())
		self.mainMenuBar.addAction(self.menuOptions.menuAction())
		self.mainMenuBar.addAction(self.menuHelp.menuAction())

		self.horizontalLayout.addWidget(self.propertiesP)
		self.horizontalLayout.addLayout(self.verticalLayout)


		MainWindow.setCentralWidget(self.centralWidget)

		self.retranslateUi(MainWindow)

	def retranslateUi(self,MainWindow):
		MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow",u"OCT Layer Segmentation", None))
		self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
		self.actionOpenFile.setText(QCoreApplication.translate("MainWindow",u"Open file...",None))
		self.actionClose.setText(QCoreApplication.translate("MainWindow",u"Close",None))
		self.actionSwitchTheme.setText(QCoreApplication.translate("MainWindow",u"Switch color theme",None))
		self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
		self.menuOptions.setTitle(QCoreApplication.translate("MainWindow", u"Options", None))
		self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
		self.rnflAction.setText(QCoreApplication.translate("MainWindow", u"RNFL", None))
		self.gcliplAction.setText(QCoreApplication.translate("MainWindow", u"GCL+IPL", None))
		self.inlAction.setText(QCoreApplication.translate("MainWindow", u"INL", None))
		self.oplAction.setText(QCoreApplication.translate("MainWindow", u"OPL", None))
		self.onlAction.setText(QCoreApplication.translate("MainWindow", u"ONL", None))
		self.isAction.setText(QCoreApplication.translate("MainWindow", u"IS", None))
		self.osAction.setText(QCoreApplication.translate("MainWindow", u"OS", None))
		self.rpeAction.setText(QCoreApplication.translate("MainWindow", u"RPE", None))
		self.allAction.setText(QCoreApplication.translate("MainWindow", u"ALL", None))
		self.loadAction.setText(QCoreApplication.translate("MainWindow", u"Load", None))
		self.saveAction.setText(QCoreApplication.translate("MainWindow", u"Save", None))
		self.finishAction.setText(QCoreApplication.translate("MainWindow", u"Finish", None))

class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		QMainWindow.__init__(self,parent)
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.workingFile = ""
		self.currentOctProcess = None
		#Main menu actions
			#File
		self.ui.actionOpenFile.triggered.connect(self.openFile)
		self.ui.actionClose.triggered.connect(self.closeApp)
			#Edit
			#Options
		self.ui.actionSwitchTheme.triggered.connect(self.switchTheme)
			#Help
		#Menubar buttons
		self.ui.rnflAction.clicked.connect(self.operationRnfl)
		self.ui.gcliplAction.clicked.connect(self.operationGclipl)
		self.ui.inlAction.clicked.connect(self.operationInl)
		self.ui.oplAction.clicked.connect(self.operationOpl)
		self.ui.onlAction.clicked.connect(self.operationOnl)
		self.ui.isAction.clicked.connect(self.operationIs)
		self.ui.osAction.clicked.connect(self.operationOs)
		self.ui.rpeAction.clicked.connect(self.operationRpe)
		self.ui.allAction.clicked.connect(self.operationAll)
		#Menu bar actions
		self.ui.loadAction.triggered.connect(self.operationLoadAction)
		#self.ui.saveAction
		#self.ui.finishAction


	def openFile(self):
		print("[INFO] Selecting file...")
		fname = QFileDialog.getOpenFileName(self, "Select .vol file...")
		fext = fname[0].split(".")[-1]
		print("[INFO] Selected file: ", fname[0] ," Extension:", fext)
		if fname != "":
			if fname[0].split(".")[-1] != 'vol':
				print("[ERROR] Invalid type of file")
			else:
				print("[INFO] Correct file...Reading...")	
				self.workingFile = fname[0]
		else:
			print("[ERROR] Invalid file")	

	def closeApp(self):
		print("[INFO] Closing app...")
		self.close()
		...

	def switchTheme(self):
		if app.palette() == darkPalette:
			#print("dark palette => white palette")
			app.setPalette(whitePalette)
		else:
			#print("white palette => dark palette")
			app.setPalette(darkPalette)

	def operationRnfl(self):
		print("Triggered: RNFL")
		self.ui.buttonBooleans[1] = self.ui.rnflAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationGclipl(self):
		print("Triggered: GCL+IPL")
		self.ui.buttonBooleans[2] = self.ui.gcliplAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationInl(self):
		print("Triggered: INL")
		self.ui.buttonBooleans[3] = self.ui.inlAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationOpl(self):
		print("Triggered: OPL")
		self.ui.buttonBooleans[4] = self.ui.oplAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationOnl(self):
		print("Triggered: ONL")
		self.ui.buttonBooleans[5] = self.ui.onlAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationIs(self):
		print("Triggered: IS")
		self.ui.buttonBooleans[6] = self.ui.isAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationOs(self):
		print("Triggered: OS")
		self.ui.buttonBooleans[7] = self.ui.osAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationRpe(self):
		print("Triggered: RPE")
		self.ui.buttonBooleans[8] = self.ui.rpeAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationAll(self):
		print("Triggered: ALL")
		state = self.ui.allAction.isChecked()
		for button in self.ui.allOpButtons:
			button.setChecked(state)
		if self.ui.allAction.isChecked():
			for index in range(len(self.ui.buttonBooleans)-1):
				self.ui.buttonBooleans[index+1] = True
		else:
			for index in range(len(self.ui.buttonBooleans)-1):
				self.ui.buttonBooleans[index+1] = False
		print(f"ALL BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...	

	def operationAllChecked(self):
		print(f"SUM BOOL: {sum(self.ui.buttonBooleans[1:9])}")
		if sum(self.ui.buttonBooleans[1:9]) < 8:
			self.ui.allAction.setChecked(False)		
			allTrue = False
		elif sum(self.ui.buttonBooleans[1:9]) == 8:
			self.ui.allAction.setChecked(True)
			allTrue = True
		else:
			...
		print(f"CHECK ALL BUTTONS: {self.ui.buttonBooleans}, BOOL: {allTrue}")
		self.sendOperation()
		...

	def sendOperation(self):
		bottomimg = self.currentOctProcess.get_individual_layers_segmentation(self.ui.buttonBooleans)
		bottomimg = bottomimg.astype('uint8')
		
		print(bottomimg.max())
		print(f"Current shape: {bottomimg.shape}")
		qImg = QImage(bottomimg, bottomimg.shape[1], bottomimg.shape[0], QImage.Format_Indexed8)

		print(f"Image width: {qImg.width()} Image height: {qImg.height()}")
		qPix = QPixmap(qImg)	
		print("Loading image...")
		self.ui.bottomLabelImg.storeImage(qPix)
		self.ui.bottomLabelImg.graphicsView.fitInView(self.ui.middleLabelImg.graphicsScene.sceneRect(),Qt.KeepAspectRatio)
		print("Loaded image!")


		# qPix = QPixmap(qImg)	
		# rPix = qPix.scaled(QSize(self.ui.topLabelImg.width(),self.ui.topLabelImg.height()))	
		# print("Loading image...")
		# self.ui.topLabelImg.setImage(rPix)

		...

	def operationLoadAction(self):
		print("[INFO] Selecting VOL FILE...")
		fname = QFileDialog.getOpenFileName(self, "Select .vol file...")
		fext = fname[0].split(".")[-1]
		print("[INFO] Selected file: ", fname[0] ," Extension:", fext)
		if fname != "":
			if fname[0].split(".")[-1] != 'vol':
				print("[ERROR] Invalid type of file")
			else:
				print("[INFO] Correct file...Reading...")	
				oc_file = fname[0]
		else:
			print("[ERROR] Invalid file")

		print("[INFO] Selecting MODEL FILE...")
		fname = QFileDialog.getOpenFileName(self, "Select .pth file...")
		fext = fname[0].split(".")[-1]
		print("[INFO] Selected file: ", fname[0] ," Extension:", fext)
		if fname != "":
			if fname[0].split(".")[-1] != 'pth':
				print("[ERROR] Invalid type of file")
			else:
				print("[INFO] Correct file...Reading...")	
				model_path = fname[0]
		else:
			print("[ERROR] Invalid file")
		print("MEASURING BEFORE")
		jetsonStats() #TAKING BEFORE
		print (f"MODEL: {model_path} OCT(VOL): {oc_file}")

		print("Creating model...")
		model = torch.load(model_path, map_location='cuda')
		print("Creating process...")
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.currentOctProcess = OCTProcessing(oct_file=oc_file, torchmodel=model, half=False, device=device)
		self.currentOctProcess.fovea_forward(imgw=512, imgh=512)
		print("Reporting process gpu, temp, pow")
		print("MEASURING AFTER")
		jetsonStats() #TAKING AFTER
		print("Creating image...")
		topimg = self.currentOctProcess.bscan_fovea
		print(f"Current shape: {topimg.shape}")
		qImg = QImage(topimg,topimg.shape[1],topimg.shape[0],QImage.Format_Indexed8)
		print(f"Image width: {qImg.width()} Image height: {qImg.height()}")
		qPix = QPixmap(qImg)	
		rPix = qPix.scaled(QSize(self.ui.topLabelImg.width(),self.ui.topLabelImg.height()))	
		print("Loading image...")
		#self.ui.topLabelImg.setImage(rPix)
		self.ui.topLabelImg.storeImage(qPix)
		self.ui.topLabelImg.graphicsView.fitInView(self.ui.topLabelImg.graphicsScene.sceneRect(),Qt.KeepAspectRatio)
		print("Loaded image!")

		middleimg = self.currentOctProcess.overlay
		print(f"Current shape: {middleimg.shape}")
		qImg = QImage(middleimg,middleimg.shape[1],middleimg.shape[0],QImage.Format_RGBA8888)
		print(f"Image width: {qImg.width()} Image height: {qImg.height()}")
		qPix = QPixmap(qImg)	
		print("Loading image...")
		#self.ui.middleLabelImg.imgLabel.setPixmap(qPix.scaled(self.ui.middleLabelImg.imgLabel.size(),Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
		self.ui.middleLabelImg.storeImage(qPix)
		self.ui.middleLabelImg.graphicsView.fitInView(self.ui.middleLabelImg.graphicsScene.sceneRect(),Qt.KeepAspectRatio)
		print("Loaded image!")


		eyeimg = self.currentOctProcess.oct.irslo
		print(f"Current shape: {eyeimg.shape}")
		qImg = QImage(eyeimg,eyeimg.shape[1],eyeimg.shape[0],QImage.Format_Indexed8)
		print(f"Image width: {qImg.width()} Image height: {qImg.height()}")
		qPix = QPixmap(qImg)	
		print("Loading image...")
		#self.ui.topLabelImg.setImage(rPix)
		self.ui.propertiesP.storeImage(qPix)
		self.ui.propertiesP.graphicsView.fitInView(self.ui.propertiesP.graphicsScene.sceneRect(),Qt.KeepAspectRatio)
		print("Loaded image!")


		print("Loading metadata...")
		model = self.ui.propertiesP.generalMetadataStandardModel

		genMetadata = self.currentOctProcess.oct.fileHeader
		genMetasearch = ['PatientID','VisitDate', 'scanPos', 'scaleX', 'scaleZ', 'sizeXSlo', 'sizeYSlo', 'scaleXSlo', 'scaleYSlo', 'numBscan']
		
		print("GENERAL METADATA")
		print(model_path)
		AIModel = model_path.split('/').pop()
		model.setData(model.index(0,1), AIModel)

		file = oc_file.split('/').pop()
		model.setData(model.index(1,1), file )

		index = 2
		for item in genMetasearch:
			print(f"{item}: {genMetadata[item]}")
			if index == 2 or index ==4:
				model.setData(model.index(index,1), genMetadata[item].decode() )
			else:
				model.setData(model.index(index,1), str(genMetadata[item]) )
			index = index + 1
		
		print("FOVEA BSCAN METADATA")
		model = self.ui.propertiesP.foveaMetadataStandardModel
		Bscan = genMetadata['numBscan']//2
		StartX = self.currentOctProcess.oct.bScanHeader(genMetadata['numBscan']//2)['startX']
		StartY = self.currentOctProcess.oct.bScanHeader(genMetadata['numBscan']//2)['startY']
		print(f"numBscan: {genMetadata['numBscan']} Bscan: {Bscan} StartX: {StartX} StartY: {StartY}")
		model.setData(model.index(0,1), str(Bscan) )
		model.setData(model.index(1,1), str(StartX) )
		model.setData(model.index(2,1), str(StartY) )
		...

class DarkPalette(QPalette):
	def __init__(self):
		super().__init__()
		self.setColor(QPalette.Window, QColor(53, 53, 53))
		self.setColor(QPalette.WindowText, Qt.white)
		self.setColor(QPalette.Base, QColor(25, 25, 25))
		self.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
		self.setColor(QPalette.ToolTipBase, Qt.black)
		self.setColor(QPalette.ToolTipText, Qt.gray)
		self.setColor(QPalette.Text, Qt.white)
		self.setColor(QPalette.Button, QColor(53, 53, 53))
		self.setColor(QPalette.ButtonText, Qt.white)
		self.setColor(QPalette.BrightText, Qt.red)
		self.setColor(QPalette.Link, QColor(42, 130, 218))
		self.setColor(QPalette.Highlight, QColor(42, 130, 218))
		self.setColor(QPalette.HighlightedText, Qt.black)
		self.setColor(QPalette.PlaceholderText, QColor(245, 245, 245, 100))

def jetsonStats():
	with jtop() as jetson:
		print(f"BOARD: {jetson.board['info']['machine']}")
		print(f"SOC HARDWARE: {jetson.board['hardware']['SOC']}")
		print(f"CPU MODEL: {jetson.cpu['CPU1']['model']}")
		print(f"TEMP GPU: {jetson.temperature['GPU']} °C")
		print(f"TEMP BCPU: {jetson.temperature['BCPU']} °C")
		print(f"TEMP THERMAL: {jetson.temperature['thermal']} °C")
		print(f"TEMP TDIODE: {jetson.temperature['Tdiode']} °C")
		print(f"POWER GENERAL CURRENT: {jetson.power[0]['cur']} mW")
		print(f"POWER GENERAL AVG: {jetson.power[0]['avg']} mW")
		print(f"POWER SYS SOC CURRENT: {jetson.power[1]['SYS SOC']['cur']} mW")
		print(f"POWER SYS SOC AVG: {jetson.power[1]['SYS SOC']['avg']} mW")
		print(f"POWER SYS GPU CURRENT: {jetson.power[1]['SYS GPU']['cur']} mW")
		print(f"POWER SYS GPU AVG: {jetson.power[1]['SYS GPU']['avg']} mW")
		print(f"POWER SYS CPU CURRENT: {jetson.power[1]['SYS CPU']['cur']} mW")
		print(f"POWER SYS CPU AVG: {jetson.power[1]['SYS CPU']['avg']} mW")
		print(f"POWER SYS DDR CURRENT: {jetson.power[1]['SYS DDR']['cur']} mW")
		print(f"POWER SYS DDR AVG: {jetson.power[1]['SYS DDR']['avg']} mW")


if __name__ == "__main__":
	print(os.name)
	print(platform.system(), platform.release())

	app = QApplication()
	darkPalette = DarkPalette()
	whitePalette = app.palette()
	#app.setStyle('Fusion')
	app.setPalette(darkPalette)  
	mainWindow = MainWindow()
	mainWindow.show()
	print(mainWindow.ui.topLabelImg.imgLabel.width(), mainWindow.ui.topLabelImg.imgLabel.height(), )
	sys.exit(app.exec_())