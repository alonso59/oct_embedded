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

class PropertiesHolder(QWidget):
	def __init__(self):
		super().__init__()
		self.thisLayout = QVBoxLayout()
		self.setLayout(self.thisLayout)
		self.titleLabel = QLabel()
		self.titleLabel.setText("Patient Info")
		self.titleLabel.setFixedWidth(60)
		self.titleLabel.setAlignment(Qt.AlignCenter)
		self.titleLabel.setStyleSheet("background-color:rgb(70,70,70); color:rgb(255,255,255)")
		self.thisLayout.addWidget(self.titleLabel)	



class ImageHolder(QWidget):
	def __init__(self, title):
		super().__init__()
		self.thisLayout = QVBoxLayout()
		self.setLayout(self.thisLayout)

		self.img = QPixmap()

		self.titleLabel = QLabel()
		self.titleLabel.setText(title)
		self.titleLabel.setFixedHeight(20)
		self.titleLabel.setAlignment(Qt.AlignCenter)
		self.titleLabel.setStyleSheet("background-color:rgb(70,70,70); color:rgb(255,255,255)")
		self.imgLabel = QLabel()
		self.imgLabel.setStyleSheet("background-color:rgb(0,0,0)")

		self.thisLayout.addWidget(self.titleLabel)
		self.thisLayout.addWidget(self.imgLabel)

	def setImage(self, pixmap):
		self.img = pixmap
		self.imgLabel.setPixmap(self.img)
		...

class Ui_MainWindow(object):
	def setupUi(self, MainWindow: QMainWindow):
		if MainWindow.objectName():
			MainWindow.setObjectName(u"MainWindow")
		MainWindow.resize(800,600)

		self.centralWidget = QWidget(MainWindow)
		self.centralWidget.setObjectName(u"centralWidget")
		#centralWidgetStyleSheet = open(r"src\frontend_gui\frontend_stylesheet.css","r")
		#self.centralWidget.setStyleSheet(centralWidgetStyleSheet.read())

		self.verticalLayout = QVBoxLayout(self.centralWidget)
		self.verticalLayout.setObjectName(u"verticalLayout")

		# self.topLabelImage = QLabel()
		# self.topLabelImage.setObjectName(u"topLabelImage")
		# self.topLabelImage.setStyleSheet("background-color:rgb(255,255,150)")
		# self.bottomLabelImage = QLabel()
		# self.bottomLabelImage.setObjectName(u"bottomLabelImage")
		# self.bottomLabelImage.setStyleSheet("background-color:rgb(255,255,150)")

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
								#A
		self.buttonBooleans = [False, False, False, False, False, False, False, False, False, False]
		DefaultPushButtonStyleSheet = "QPushButton {background-color:red;} QPushButton:checked{background-color:green;}"
		self.ilmrnflAction = QPushButton()
		self.ilmrnflAction.setObjectName(u"ilmrnflAction")
		self.ilmrnflAction.setCheckable(True)
		self.ilmrnflAction.setChecked(False)
		self.ilmrnflAction.setStyleSheet(DefaultPushButtonStyleSheet)

		self.gclAction = QPushButton()
		self.gclAction.setObjectName(u"gclAction")
		self.gclAction.setCheckable(True)
		self.gclAction.setChecked(False)
		self.gclAction.setStyleSheet(DefaultPushButtonStyleSheet)

		self.iplAction = QPushButton()
		self.iplAction.setObjectName(u"iplAction")
		self.iplAction.setCheckable(True)
		self.iplAction.setChecked(False)
		self.iplAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
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
		
		self.elmAction = QPushButton()
		self.elmAction.setObjectName(u"elmAction")
		self.elmAction.setCheckable(True)
		self.elmAction.setChecked(False)
		self.elmAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.ezAction = QPushButton()
		self.ezAction.setObjectName(u"ezAction")
		self.ezAction.setCheckable(True)
		self.ezAction.setChecked(False)
		self.ezAction.setStyleSheet(DefaultPushButtonStyleSheet)
		
		self.izrpeAction = QPushButton()
		self.izrpeAction.setObjectName(u"izrpeAction")
		self.izrpeAction.setCheckable(True)
		self.izrpeAction.setChecked(False)
		self.izrpeAction.setStyleSheet(DefaultPushButtonStyleSheet)


		
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
		self.allOpButtons.append(self.ilmrnflAction)
		self.allOpButtons.append(self.gclAction)
		self.allOpButtons.append(self.iplAction)
		self.allOpButtons.append(self.inlAction)
		self.allOpButtons.append(self.oplAction)
		self.allOpButtons.append(self.onlAction)
		self.allOpButtons.append(self.elmAction)
		self.allOpButtons.append(self.ezAction)
		self.allOpButtons.append(self.izrpeAction)

		self.toolBar.addWidget(self.ilmrnflAction)
		self.toolBar.addWidget(self.gclAction)
		self.toolBar.addWidget(self.iplAction)
		self.toolBar.addWidget(self.inlAction)
		self.toolBar.addWidget(self.oplAction)
		self.toolBar.addWidget(self.onlAction)
		self.toolBar.addWidget(self.elmAction)
		self.toolBar.addWidget(self.ezAction)
		self.toolBar.addWidget(self.izrpeAction)
		self.toolBar.addWidget(self.allAction)
		self.toolBar.addSeparator()
		self.toolBar.addAction(self.loadAction)
		self.toolBar.addAction(self.saveAction)
		self.toolBar.addAction(self.finishAction)

		self.mainMenuBar.addAction(self.menuFile.menuAction())
		self.mainMenuBar.addAction(self.menuEdit.menuAction())
		self.mainMenuBar.addAction(self.menuOptions.menuAction())
		self.mainMenuBar.addAction(self.menuHelp.menuAction())


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
		self.ilmrnflAction.setText(QCoreApplication.translate("MainWindow", u"ILM-RNFL", None))
		self.gclAction.setText(QCoreApplication.translate("MainWindow", u"GCL", None))
		self.iplAction.setText(QCoreApplication.translate("MainWindow", u"IPL", None))
		self.inlAction.setText(QCoreApplication.translate("MainWindow", u"INL", None))
		self.oplAction.setText(QCoreApplication.translate("MainWindow", u"OPL", None))
		self.onlAction.setText(QCoreApplication.translate("MainWindow", u"ONL", None))
		self.elmAction.setText(QCoreApplication.translate("MainWindow", u"ELM", None))
		self.ezAction.setText(QCoreApplication.translate("MainWindow", u"EZ", None))
		self.izrpeAction.setText(QCoreApplication.translate("MainWindow", u"IZ-RPE", None))
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
		self.ui.ilmrnflAction.clicked.connect(self.operationIlmrnf)
		self.ui.gclAction.clicked.connect(self.operationGcl)
		self.ui.iplAction.clicked.connect(self.operationIpl)
		self.ui.inlAction.clicked.connect(self.operationInl)
		self.ui.oplAction.clicked.connect(self.operationOpl)
		self.ui.onlAction.clicked.connect(self.operationOnl)
		self.ui.elmAction.clicked.connect(self.operationElm)
		self.ui.ezAction.clicked.connect(self.operationEz)
		self.ui.izrpeAction.clicked.connect(self.operationIzrpe)
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

	def operationIlmrnf(self):
		print("Triggered: ILMRNF")
		self.ui.buttonBooleans[1] = self.ui.ilmrnflAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationGcl(self):
		print("Triggered: GCL")
		self.ui.buttonBooleans[2] = self.ui.gclAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationIpl(self):
		print("Triggered: IPL")
		self.ui.buttonBooleans[3] = self.ui.iplAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationInl(self):
		print("Triggered: INL")
		self.ui.buttonBooleans[4] = self.ui.inlAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationOpl(self):
		print("Triggered: OPL")
		self.ui.buttonBooleans[5] = self.ui.oplAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationOnl(self):
		print("Triggered: ONL")
		self.ui.buttonBooleans[6] = self.ui.onlAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationElm(self):
		print("Triggered: ELM")
		self.ui.buttonBooleans[7] = self.ui.elmAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationEz(self):
		print("Triggered: EZ")
		self.ui.buttonBooleans[8] = self.ui.ezAction.isChecked()
		print(f"BUTTONS: {self.ui.buttonBooleans}")
		self.operationAllChecked()
		...

	def operationIzrpe(self):
		print("Triggered: IZRPE")
		self.ui.buttonBooleans[9] = self.ui.izrpeAction.isChecked()
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
		print(f"SUM BOOL: {sum(self.ui.buttonBooleans[1:10])}")
		if sum(self.ui.buttonBooleans[1:10]) < 9:
			self.ui.allAction.setChecked(False)		
			allTrue = False
		elif sum(self.ui.buttonBooleans[1:10]) == 9:
			self.ui.allAction.setChecked(True)
			allTrue = True
		else:
			...
		print(f"CHECK ALL BUTTONS: {self.ui.buttonBooleans}, BOOL: {allTrue}")
		self.sendOperation()
		...

	def sendOperation(self):
		bottomimg = self.currentOctProcess.get_individual_layers_segmentation(self.ui.buttonBooleans)


		print(f"Current shape: {bottomimg.shape}")
		qImg = QImage(bottomimg, bottomimg.shape[1], bottomimg.shape[0],QImage.Format_Indexed8)
		print(f"Image width: {qImg.width()} Image height: {qImg.height()}")
		qPix = QPixmap(qImg)	
		print("Loading image...")
		self.ui.bottomLabelImg.imgLabel.setPixmap(qPix.scaled(self.ui.bottomLabelImg.imgLabel.size(),Qt.KeepAspectRatio, Qt.SmoothTransformation))
		print("Loaded image!")

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
		
		print (f"MODEL: {model_path} OCT(VOL): {oc_file}")

		print("Creating model...")
		model = torch.load(model_path, map_location='cuda')
		print("Creating process...")
		self.currentOctProcess = OCTProcessing(oct_file=oc_file, torchmodel=model)
		print("Creating image...")

		topimg = self.currentOctProcess.bscan_fovea
		print(f"Current shape: {topimg.shape}")
		qImg = QImage(topimg,topimg.shape[1],topimg.shape[0],QImage.Format_Indexed8)
		print(f"Image width: {qImg.width()} Image height: {qImg.height()}")
		qPix = QPixmap(qImg)	
		rPix = qPix.scaled(QSize(self.ui.topLabelImg.width(),self.ui.topLabelImg.height()))	
		print("Loading image...")
		self.ui.topLabelImg.setImage(rPix)
		print("Loaded image!")

		middleimg = self.currentOctProcess.overlay
		print(f"Current shape: {middleimg.shape}")
		qImg = QImage(middleimg,middleimg.shape[1],middleimg.shape[0],QImage.Format_RGBA8888)
		print(f"Image width: {qImg.width()} Image height: {qImg.height()}")
		qPix = QPixmap(qImg)	
		print("Loading image...")
		self.ui.middleLabelImg.imgLabel.setPixmap(qPix.scaled(self.ui.middleLabelImg.imgLabel.size(),Qt.KeepAspectRatio, Qt.SmoothTransformation))
		print("Loaded image!")

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