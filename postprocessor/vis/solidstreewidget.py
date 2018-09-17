import sys
#from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


class SolidsTreeWidget(QtGui.QWidget):

    def __init__(self, solids_dict):
        super( SolidsTreeWidget, self ).__init__()
        self.treeWidget = QtGui.QTreeWidget()
        self.treeWidget.setHeaderHidden(True)
        self.solids_state = {}
        self.solid_checkboxes = {}
        self.addItems(self.treeWidget.invisibleRootItem(), solids_dict)
        self.treeWidget.itemChanged.connect (self.handleChanged)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.treeWidget)
        self.setLayout(layout)
        

    def addItems(self, parent, solids_dict):
        column = 0
        solids_item = self.addParent(parent, column, 'Solids', 'Name')

        for k in solids_dict:
            if k not in self.solids_state:
                self.solids_state[str(k)] = True
            self.addChild(solids_item, column, k, 'data Solid')

    def addParent(self, parent, column, title, data):
        item = QtGui.QTreeWidgetItem(parent, [title])
        item.setData(column, QtCore.Qt.UserRole, data)
        item.setChildIndicatorPolicy(QtGui.QTreeWidgetItem.ShowIndicator)
        item.setExpanded (True)
        return item

    def addChild(self, parent, column, title, data):
        #item = QtGui.QTreeWidgetItem(parent, [title])
        item = pg.TreeWidgetItem(parent, [title])
        self.solid_checkboxes[title] = item        
        item.setData(column, QtCore.Qt.UserRole, data)
        item.setCheckState (column, QtCore.Qt.Checked )
        return item

    def handleChanged(self, item, column):
        if item.checkState(column) == QtCore.Qt.Checked:
            self.solids_state[ str(item.text(column)) ] = True
            print "checked", item, item.text(column), self.solids_state[ str(item.text(column)) ]
        if item.checkState(column) == QtCore.Qt.Unchecked:
            self.solids_state[ str(item.text(column)) ] = False
            print "unchecked", item, item.text(column), self.solids_state[ str(item.text(column)) ]

if __name__ == "__main__":

    solids = {"TPC":[], "PMT":[] }
    app = QtGui.QApplication(sys.argv)
    window = SolidsTreeWidget(solids)
    window.show()
    sys.exit(app.exec_())
