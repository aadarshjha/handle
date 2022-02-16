from PyQt5.QtWidgets import QApplication, QLabel 


class MainWindow(QApplication):
    def __init__(self):
        super().__init__([])
        # bigger window size:
        self.window_size = (1000, 800)
        
    # function to create the camera window: 
    def create_camera_window(self):
        self.camera_window = QLabel("Camera Window")
        self.camera_window.show()
        self.camera_window.setFixedSize(self.window_size[0], self.window_size[1])
        self.exec_()


if __name__ == "__main__":
    MainWindow().create_camera_window()
    MainWindow().execute_window()

# # create a new window
# app = QApplication([])
# window = QLabel("Hello World")
# window.show()
# app.exec_()
