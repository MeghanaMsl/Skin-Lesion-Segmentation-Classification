# The styling for this code has been referenced from link below:
# https://github.com/HassanSajjad229/BrainTumorSegmentation-PYTHON/blob/master/tumorsegmentation.py
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
import glob, os.path
import os, sys
import pickle
import numpy as np
from keras.layers import concatenate, Activation, BatchNormalization, Dropout, Input
from keras.models import Model
import tensorflow as tf
from skimage.transform import resize
from skimage import io, img_as_ubyte, feature, color, measure
import cv2
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

tf.config.run_functions_eagerly(True)
global a

folder_path = Path() / "data"

path = folder_path / "gui_outputs"
data_path = folder_path / "test"


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        """ Function to define the styling for the GUI"""
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1710, 600)
        MainWindow.setStyleSheet("background-color: #005bea;")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(110, 55, 1710, 900))
        self.frame.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.630318, y1:0.392, x2:0.068, y2:0, stop:0.551136 #4facfe, stop:1 #00f2fe);\n"
                                 "   border-radius: 25px;\n"
                                 "    border: 2px solid #00c6fb;")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.getimg = QtWidgets.QPushButton(self.frame)
        self.getimg.setGeometry(QtCore.QRect(70, 530, 171, 61))
        self.getimg.setStyleSheet("color:white;\n"
                                   "font: 14pt \"Gadugi\";\n"
                                   "   border-radius: 20px;\n"
                                   "    border: 2px solid #00c6fb;\n"
                                   "background-color:#005bea;\n"
                                   "width:171;\n"
                                   "height:61")
        self.getimg.setObjectName("getimg")
        self.rgbtgray = QtWidgets.QPushButton(self.frame)
        self.rgbtgray.setGeometry(QtCore.QRect(349, 530, 177, 61))
        self.rgbtgray.setStyleSheet("color:white;\n"
                                    "font: 14pt \"Gadugi\";\n"
                                    "   border-radius: 20px;\n"
                                    "    border: 2px solid #00c6fb;\n"
                                    "background-color:#005bea;\n"
                                    "width:171px;\n"
                                    "height:61px;")
        self.rgbtgray.setObjectName("rgbtgray")
        self.morph = QtWidgets.QPushButton(self.frame)
        self.morph.setGeometry(QtCore.QRect(628, 530, 171, 61))
        self.morph.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#005bea;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.morph.setObjectName("morph")
        self.inpaint = QtWidgets.QPushButton(self.frame)
        self.inpaint.setGeometry(QtCore.QRect(907, 530, 171, 61))
        self.inpaint.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#005bea;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.inpaint.setObjectName("inpaint")
        self.gauss = QtWidgets.QPushButton(self.frame)
        self.gauss.setGeometry(QtCore.QRect(1186, 530, 171, 61))
        self.gauss.setStyleSheet("color:white;\n"
                                 "font: 14pt \"Gadugi\";\n"
                                 "   border-radius: 20px;\n"
                                 "    border: 2px solid #00c6fb;\n"
                                 "background-color:#005bea;\n"
                                 "width:171px;\n"
                                 "height:61px;")
        self.gauss.setObjectName("gauss")
        self.unet = QtWidgets.QPushButton(self.frame)
        self.unet.setGeometry(QtCore.QRect(1425, 530, 210, 61))
        self.unet.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#005bea;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.unet.setObjectName("unet")

        self.classify = QtWidgets.QPushButton(self.frame)
        self.classify.setGeometry(QtCore.QRect(70, 670, 200, 61))
        self.classify.setStyleSheet("color:white;\n"
                                  "font: 14pt \"Gadugi\";\n"
                                  "   border-radius: 20px;\n"
                                  "    border: 2px solid #00c6fb;\n"
                                  "background-color:#005bea;\n"
                                  "width:171px;\n"
                                  "height:61px;")
        self.classify.setObjectName("classify")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(20, 90, 261, 421))
        self.label.setStyleSheet("background-color: rgb(218, 218, 218);\n"
                                 "   border-radius: 20px;\n"
                                 "    border: 2px solid #00c6fb;")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(299, 90, 261, 421))
        self.label_2.setStyleSheet("background-color: rgb(218, 218, 218);\n"
                                   "   border-radius: 20px;\n"
                                   "    border: 2px solid #00c6fb;")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(578, 90, 261, 421))
        self.label_3.setStyleSheet("background-color: rgb(218, 218, 218);\n"
                                   "   border-radius: 20px;\n"
                                   "    border: 2px solid #00c6fb;")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(857, 90, 261, 421))
        self.label_4.setStyleSheet("background-color: rgb(218, 218, 218);\n"
                                   "   border-radius: 20px;\n"
                                   "    border: 2px solid #00c6fb;")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setGeometry(QtCore.QRect(1136, 90, 261, 421))
        self.label_5.setStyleSheet("background-color: rgb(218, 218, 218);\n"
                                   "   border-radius: 20px;\n"
                                   "    border: 2px solid #00c6fb;")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setGeometry(QtCore.QRect(1415, 90, 261, 421))
        self.label_6.setStyleSheet("background-color: rgb(218, 218, 218);\n"
                                   "   border-radius: 20px;\n"
                                   "    border: 2px solid #00c6fb;")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")

        self.titlelbl = QtWidgets.QLabel(self.centralwidget)
        self.titlelbl.setGeometry(QtCore.QRect(650, 10, 701, 41))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(26)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.titlelbl.setFont(font)
        self.titlelbl.setStyleSheet("opacity:0.6;\n"
                                    "font: 26pt \"MS Shell Dlg 2\";\n"
                                    "color: rgb(216, 216, 216);")
        self.titlelbl.setObjectName("titlelbl")

        self.lbl1 = QtWidgets.QLabel(self.centralwidget)
        self.lbl1.setGeometry(QtCore.QRect(410, 723, 160, 61))
        self.lbl1.setFont(font)
        self.lbl1.setStyleSheet("opacity:0.6;\n"
                                    "font: 14pt \"MS Shell Dlg 2\";\n"
                                    "color: rgb(216, 216, 216);")
        self.lbl1.setObjectName("lbl1")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        error_dialog = QtWidgets.QErrorMessage()
        error_dialog.setWindowTitle('Image processing tool')
        error_dialog.showMessage('Please click the buttons in Sequential order to proceed!')
        error_dialog.exec()


        filelist = glob.glob(os.path.join(path, "*.jpg"))
        for f in filelist:
            os.remove(f)

        self.getimg.clicked.connect(self.get_image)
        self.rgbtgray.clicked.connect(self.convert_grayscale)
        self.morph.clicked.connect(self.apply_blackhat)
        self.inpaint.clicked.connect(self.apply_inpaint)
        self.gauss.clicked.connect(self.apply_gaussianblur)
        self.unet.clicked.connect(self.get_unetmask)
        self.classify.clicked.connect(self.classify_image)

    def retranslateUi(self, MainWindow):
        """ Function to invoke all the components defined in the GUI"""
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Skin Lesion Segmentation & Classification"))
        self.getimg.setText(_translate("MainWindow", "Select Image"))
        self.rgbtgray.setText(_translate("MainWindow", "Grayscale Image"))
        self.morph.setText(_translate("MainWindow", "Morphology"))
        self.inpaint.setText(_translate("MainWindow", "Inpaint Image"))
        self.gauss.setText(_translate("MainWindow", "GaussianBlur"))
        self.unet.setText(_translate("MainWindow", "Segmentation mask"))
        self.classify.setText(_translate("MainWindow", "View Classification"))
        self.titlelbl.setText(_translate("MainWindow", "Skin Lesion Segmentation Panel"))

    def get_image(self):
        """ Function to let the user select an image for image segmentation"""
        global a
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", r"C:\Users\mslme\Documents\MSL\UH\SEM3\FMI\Project\data\test",
                                                            "Image Files (*.png *.jpg *jpeg *.bmp)")  # Ask for file

        a = fileName

        if fileName:  # If the user gives a file

            pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.label.width(), self.label.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center

    def convert_grayscale(self):
        """ Function to convert the selected input image to grayscale image"""
        image = io.imread(a)
        greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        io.imsave(os.path.join(path, 'grayscale.jpg'), greyscale)

        pixmap = QtGui.QPixmap(os.path.join(path,'grayscale.jpg'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_2.setPixmap(pixmap)  # Set the pixmap onto the label

    def apply_blackhat(self):
        """ Function to apply blackhat operation on the grayscale image"""
        image = io.imread(os.path.join(path, "grayscale.jpg"))
        kernel = cv2.getStructuringElement(1, (17, 17))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        io.imsave(os.path.join(path, 'blackhat.jpg'), blackhat)

        pixmap = QtGui.QPixmap(os.path.join(path, 'blackhat.jpg'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_3.width(), self.label_3.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_3.setPixmap(pixmap)  # Set the pixmap onto the label

    def apply_inpaint(self):
        """ Function to perform inpainting on the thrsholded blackhat image"""
        selected_img = io.imread(a)
        image = io.imread(os.path.join(path, "blackhat.jpg"))
        ret, thresh2 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(selected_img, thresh2, 1, cv2.INPAINT_TELEA)
        io.imsave(os.path.join(path, 'inpaint.jpg'), dst)

        pixmap = QtGui.QPixmap(os.path.join(path, 'inpaint.jpg'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_4.width(), self.label_4.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_4.setPixmap(pixmap)  # Set the pixmap onto the label

    def apply_gaussianblur(self):
        """ Function to perform gaussian blur on the inpainted image"""
        image = io.imread(os.path.join(path, "inpaint.jpg"))
        img_filtered = cv2.GaussianBlur(image, (7, 7), 0)
        io.imsave(os.path.join(path, 'final.jpg'), img_filtered)

        pixmap = QtGui.QPixmap(os.path.join(path, 'final.jpg'))  # Setup pixmap with the provided image
        pixmap = pixmap.scaled(self.label_5.width(), self.label_5.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_5.setPixmap(pixmap)  # Set the pixmap onto the label


    def get_unetmask(self):
        """ Function to invoke the UNET model and predict the segmentation mask for the pre-processed image"""
        IMG_WIDTH = 384
        IMG_HEIGHT = 256
        IMG_CHANNELS = 3

        X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

        input_img = Input((IMG_HEIGHT, IMG_WIDTH, 3), name='img')
        model = self.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

        model.load_weights('model-skin-lesion-segmentation-org2000.h5')

        img = io.imread(os.path.join(path, "final.jpg"))[:, :, :IMG_CHANNELS]
        img1 = resize(img, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
        X_test[0] = img1

        self.predicted = model.predict(X_test, verbose=1)
        self.predicted = (self.predicted > 0.5).astype(bool)

        cv2.imwrite(os.path.join(path, 'segmented.jpg'), img_as_ubyte(self.predicted.squeeze()))
        pixmap = QtGui.QPixmap(os.path.join(path, 'segmented.jpg'))  # Setup pixmap with the provided image

        pixmap = pixmap.scaled(self.label_6.width(), self.label_6.height(),
                               QtCore.Qt.KeepAspectRatio)  # Scale pixmap
        self.label_6.setPixmap(pixmap)  # Set the pixmap onto the label


    def classify_image(self):
        """ Function to invoke the trained SVM model to perform classification on the image"""
        _translate = QtCore.QCoreApplication.translate
        test_dict = {}
        for i in range(self.predicted.shape[0]):
            img_label = measure.label(np.squeeze(self.predicted[i]))
            region_props = measure.regionprops(img_label)
            num_labels = len(region_props)
            areas = [region.area for region in region_props]
            if num_labels > 0 and areas[np.argmax(areas)] >= 1200:
                target_label = region_props[np.argmax(areas)].label
                test_dict[a[-16:]] = region_props[target_label - 1]
        test_features = self.extract_features(test_dict, data_path)
        X_test_svm = list(test_features.values())
        clf = pickle.load(open('svm_model_poly.pkl', 'rb'))
        y_pred = clf.predict(X_test_svm)
        self.lbl1.setText(_translate("MainWindow", y_pred[0]))


    # function code referenced from https://github.com/biagiom/skin-lesions-classifier/blob/master/skin_lesions_classifier.ipynb
    def extract_features(self, input_dict, base_path):
        """Function to perform feature extraction for the segmented masks by calculating their Asymmetry, Border irregularity,
        Color variation, Diameter and Texture"""
        features = {}
        for idx, image_name in enumerate(input_dict):
            path = base_path / image_name
            image = io.imread(path)
            gray_img = color.rgb2gray(image)
            lesion_region = input_dict[image_name]

            # Asymmetry
            area_total = lesion_region.area
            img_mask = lesion_region.image
            horizontal_flip = np.fliplr(img_mask)
            diff_horizontal = img_mask * ~horizontal_flip
            vertical_flip = np.flipud(img_mask)
            diff_vertical = img_mask * ~vertical_flip
            diff_horizontal_area = np.count_nonzero(diff_horizontal)
            diff_vertical_area = np.count_nonzero(diff_vertical)
            asymm_idx = 0.5 * ((diff_horizontal_area / area_total) + (diff_vertical_area / area_total))
            ecc = lesion_region.eccentricity

            # Border irregularity
            compact_index = (lesion_region.perimeter ** 2) / (4 * np.pi * area_total)

            # Color variegation:
            sliced = image[lesion_region.slice]
            lesion_r = sliced[:, :, 0]
            lesion_g = sliced[:, :, 1]
            lesion_b = sliced[:, :, 2]
            C_r = np.std(lesion_r) / np.max(lesion_r)
            C_g = np.std(lesion_g) / np.max(lesion_g)
            C_b = np.std(lesion_b) / np.max(lesion_b)

            # Diameter:
            eq_diameter = lesion_region.equivalent_diameter

            # Texture:
            glcm = feature.graycomatrix(image=img_as_ubyte(gray_img), distances=[1],
                                        angles=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 2],
                                        symmetric=True, normed=True)
            correlation = np.mean(feature.graycoprops(glcm, prop='correlation'))
            homogeneity = np.mean(feature.graycoprops(glcm, prop='homogeneity'))
            energy = np.mean(feature.graycoprops(glcm, prop='energy'))
            contrast = np.mean(feature.graycoprops(glcm, prop='contrast'))

            features[image_name] = [asymm_idx, ecc, compact_index, C_r, C_g, C_b,
                                    eq_diameter, correlation, homogeneity, energy, contrast]
        return features

    def conv2d_block(self,input_tensor, n_filters, kernel_size=3, batchnorm=True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                   kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                   kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def get_unet(self,input_img, n_filters=16, dropout=0.1, batchnorm=True):
        """Function to define the UNET Model"""
        # Contracting Path
        c1 = self.conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = self.conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self.conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.showMaximized()
    MainWindow.show()
    sys.exit(app.exec_())

