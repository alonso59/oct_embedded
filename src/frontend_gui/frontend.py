#Import System Util
import sys, os, time, platform
# Import PySide6 classes
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

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
		self.imgLabel.setStyleSheet("background-color:rgb(255,255,150)")

		self.thisLayout.addWidget(self.titleLabel)
		self.thisLayout.addWidget(self.imgLabel)

	def setImage(self, pixmap):
		self.img.load(pixmap)
		self.imgLabel.setPixmap(self.img)
		...

class Ui_MainWindow(object):
	def setupUi(self, MainWindow: QMainWindow):
		if MainWindow.objectName():
			MainWindow.setObjectName(u"MainWindow")
		MainWindow.resize(800,600)

		self.centralWidget = QWidget(MainWindow)
		self.centralWidget.setObjectName(u"centralWidget")

		self.verticalLayout = QVBoxLayout(self.centralWidget)
		self.verticalLayout.setObjectName(u"verticalLayout")

		self.topLabelImage = QLabel()
		self.topLabelImage.setObjectName(u"topLabelImage")
		self.topLabelImage.setStyleSheet("background-color:rgb(255,255,150)")
		self.bottomLabelImage = QLabel()
		self.bottomLabelImage.setObjectName(u"bottomLabelImage")
		self.bottomLabelImage.setStyleSheet("background-color:rgb(255,255,150)")

		self.topLabelImg = ImageHolder("OCT FOVEA")		
		self.bottomLabelImg = ImageHolder("OCT Segmentation")

		self.verticalLayout.addWidget(self.topLabelImg)
		self.verticalLayout.addWidget(self.bottomLabelImg)

		self.mainMenuBar = QMenuBar(MainWindow)
		self.mainMenuBar.setObjectName(u"mainMenuBar")
		self.mainMenuBar.setGeometry(QRect(0,0,800,21))
		MainWindow.setMenuBar(self.mainMenuBar)

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

		self.ilmrnflAction = QAction()
		self.ilmrnflAction.setObjectName(u"ilmrnflAction")

		self.gclAction = QAction()
		self.gclAction.setObjectName(u"gclAction")

		self.iplAction = QAction()
		self.iplAction.setObjectName(u"iplAction")
		
		self.inlAction = QAction()
		self.inlAction.setObjectName(u"inlAction")
		
		self.oplAction = QAction()
		self.oplAction.setObjectName(u"oplAction")
		
		self.onlAction = QAction()
		self.onlAction.setObjectName(u"onlAction")
		
		self.elmAction = QAction()
		self.elmAction.setObjectName(u"elmAction")
		
		self.ezAction = QAction()
		self.ezAction.setObjectName(u"ezAction")
		
		self.izrpeAction = QAction()
		self.izrpeAction.setObjectName(u"izrpeAction")
		
		self.allAction = QAction()
		self.allAction.setObjectName(u"allAction")

		self.backAction = QAction()
		self.backAction.setObjectName(u"backAction")

		self.saveAction = QAction()
		self.saveAction.setObjectName(u"saveAction")

		self.finishAction = QAction()
		self.finishAction.setObjectName(u"finishAction")

		self.toolBar.addAction(self.ilmrnflAction)
		self.toolBar.addAction(self.gclAction)
		self.toolBar.addAction(self.iplAction)
		self.toolBar.addAction(self.inlAction)
		self.toolBar.addAction(self.oplAction)
		self.toolBar.addAction(self.onlAction)
		self.toolBar.addAction(self.elmAction)
		self.toolBar.addAction(self.izrpeAction)
		self.toolBar.addAction(self.allAction)
		self.toolBar.addSeparator()
		self.toolBar.addAction(self.backAction)
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
		self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
		self.menuOptions.setTitle(QCoreApplication.translate("MainWindow", u"Options", None))
		self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
		self.ilmrnflAction.setText(QCoreApplication.translate("MainWindow", u"ILM-RNFL", None))
		self.gclAction.setText(QCoreApplication.translate("MainWindow", u"GCL", None))
		self.iplAction.setText(QCoreApplication.translate("MainWindow", u"IPL", None))
		self.inlAction.setText(QCoreApplication.translate("MainWindow", u"INL", None))
		self.oplAction.setText(QCoreApplication.translate("MainWindow", u"OPL", None))
		self.onlAction.setText(QCoreApplication.translate("MainWindow", u"ELM", None))
		self.elmAction.setText(QCoreApplication.translate("MainWindow", u"EZ", None))
		self.izrpeAction.setText(QCoreApplication.translate("MainWindow", u"IZ-RPE", None))
		self.allAction.setText(QCoreApplication.translate("MainWindow", u"ALL", None))
		self.backAction.setText(QCoreApplication.translate("MainWindow", u"Back", None))
		self.saveAction.setText(QCoreApplication.translate("MainWindow", u"Save", None))
		self.finishAction.setText(QCoreApplication.translate("MainWindow", u"Finish", None))

class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		QMainWindow.__init__(self,parent)
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

if __name__ == "__main__":
	print(os.name)
	print(platform.system(), platform.release())

	app = QApplication()
	mainWindow = MainWindow()
	mainWindow.show()
	sys.exit(app.exec_())