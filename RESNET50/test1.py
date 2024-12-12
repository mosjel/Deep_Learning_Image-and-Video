import os

from PyQt5.QtGui import QPixmap,QImage
from PIL.ImageQt import ImageQt

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton

from PIL import Image



class ImageGallery(QWidget):

    def __init__(self, image_folder):

        super().__init__()

        self.image_folder = image_folder

        self.current_page = 0

        self.images_per_page = 4

        self.image_labels = []

        self.initUI()

    

    def initUI(self):

        self.setWindowTitle('Image Gallery')

        self.layout = QVBoxLayout()

        self.setLayout(self.layout)

        

        # Add buttons for navigation

        button_layout = QHBoxLayout()

        self.prev_button = QPushButton('Previous')

        self.prev_button.clicked.connect(self.show_previous_page)

        self.next_button = QPushButton('Next')

        self.next_button.clicked.connect(self.show_next_page)

        button_layout.addWidget(self.prev_button)

        button_layout.addStretch(1)

        button_layout.addWidget(self.next_button)

        self.layout.addLayout(button_layout)

        

        # Add image labels

        for i in range(self.images_per_page):

            image_label = QLabel()

            self.image_labels.append(image_label)

            self.layout.addWidget(image_label)

        

        self.show_current_page()

        self.show()

    

    def show_current_page(self):

        start_index = self.current_page * self.images_per_page

        end_index = (self.current_page + 1) * self.images_per_page

        image_files = os.listdir(self.image_folder)[start_index:end_index]

        for i, image_file in enumerate(image_files):

            image_path = os.path.join(self.image_folder, image_file)

            image = Image.open(image_path)
            qimage=QImage(ImageQt(image))
            pixmap = QPixmap.fromImage(qimage)

            self.image_labels[i].setPixmap(pixmap)

    

    def show_previous_page(self):

        if self.current_page > 0:

            self.current_page -= 1

            self.show_current_page()

    

    def show_next_page(self):

        if self.current_page < (len(os.listdir(self.image_folder)) // self.images_per_page):

            self.current_page += 1

            self.show_current_page()


if __name__ == '__main__':
    
    app = QApplication([])

    gallery = ImageGallery(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\test1')


    app.exec_()