from PyQt5.QtWidgets import QApplication, QLabel 


class MainWindow(QApplication):
    def __init__(self):
        super().__init__([])
        self.window = QLabel("Hello World")
        self.window.show()
        self.exec_()

if __name__ == "main":
    MainWindow()