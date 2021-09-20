import pickle

from PyQt5 import QtGui, QtCore
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton, QGridLayout, QWidget, QLabel


class EvalSetFilter(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.eval_sets = pickle.load(open("data/noun-1000.p", "rb"))
        self.eval_sets_filtered = {key: [] for key in self.eval_sets.keys()}

        self.eval_set_index = 0
        self.eval_set_key = list(self.eval_sets.keys())[self.eval_set_index]

        while len(self.eval_sets[self.eval_set_key]) < 1:
            self.eval_set_index += 1
            self.eval_set_key = list(self.eval_sets.keys())[self.eval_set_index]

        self.eval_set = self.eval_sets[self.eval_set_key]
        print(f"{self.eval_set_key} ({len(self.eval_set)} examples)")

        self.sample_index = 0
        self.sample = self.eval_set[self.sample_index]

        grid = QGridLayout()
        self.setLayout(grid)

        self.pic_example = QLabel(self)
        grid.addWidget(self.pic_example, *(0, 0))
        self.pic_counter_example = QLabel(self)
        grid.addWidget(self.pic_counter_example, *(0, 1))

        self.text_example = QLabel(self)
        self.text_example.setFixedHeight(15)
        grid.addWidget(self.text_example, *(1, 0))
        self.text_counterexample = QLabel(self)
        self.text_counterexample.setFixedHeight(15)
        grid.addWidget(self.text_counterexample, *(1, 1))

        self.plot_sample(self.sample)
        self.pic_example.show()
        self.pic_counter_example.show()
        self.text_example.show()
        self.text_counterexample.show()


        button_accept = QPushButton("accept")
        button_accept.clicked.connect(self.accept)
        grid.addWidget(button_accept, 2, 0)

        button_reject = QPushButton("reject")
        button_reject.clicked.connect(self.reject)
        grid.addWidget(button_reject, 2, 1)

        button_save = QPushButton("save filtered results")
        button_save.clicked.connect(self.save)
        grid.addWidget(button_save, 3, 0, 1, 2)

        self.resize(1500, 1000)

        self.setWindowTitle('Filter eval set')
        self.show()

    def plot_sample(self, sample):
        # set rectangle colors and thicknesses
        penRectangleGreen = QtGui.QPen(QtCore.Qt.green)
        penRectangleGreen.setWidth(3)
        penRectangleRed = QtGui.QPen(QtCore.Qt.red)
        penRectangleRed.setWidth(3)
        penWhite = QtGui.QPen(QtCore.Qt.white)
        penWhite.setWidth(3)

        for img in [sample["img_example"], sample["img_counterexample"]]:
            pixmap = QtGui.QPixmap(img)
            pixmap = pixmap.scaledToWidth(700)

            # create painter instance with pixmap
            self.painterInstance = QtGui.QPainter(pixmap)

            relationships = [self.sample["relationship_target"], self.sample["relationship_visual_distractor"]]
            if img == sample["img_counterexample"]:
                relationships = [self.sample["counterexample_relationship_target"],
                                 self.sample["counterexample_relationship_visual_distractor"]]

            text = ""
            for relationship, pen in zip(relationships, [penRectangleGreen, penRectangleRed]):
                bb = relationship.bounding_box

                # draw rectangle on painter
                self.painterInstance.setPen(pen)
                self.painterInstance.drawRect(round(bb[0] * pixmap.width()), round(bb[1] * pixmap.height()),
                                              round(bb[2] * pixmap.width()),
                                              round(bb[3] * pixmap.height()))
                label = relationship.Label1 + f" ({relationship.Label2})"
                self.painterInstance.setPen(penWhite)
                self.painterInstance.drawText(round(bb[0] * pixmap.width()), round(bb[1] * pixmap.height()), label)

                text += label

            self.painterInstance.end()

            if img == sample["img_counterexample"]:
                self.pic_example.setPixmap(pixmap)
                self.text_example.setText(text)
            else:
                self.pic_counter_example.setPixmap(pixmap)
                self.text_counterexample.setText(text)

    def get_next_sample(self):
        self.sample_index += 1
        while self.sample_index >= len(self.eval_set):
            self.eval_set_index += 1
            if self.eval_set_index >= len(self.eval_sets.keys()):
                self.save()
                QMessageBox.information(self, "QMessageBox.information()",
                                        "End of dataset, filtered everything!")
                self.close()
                sys.exit()
            self.eval_set_key = list(self.eval_sets.keys())[self.eval_set_index]
            self.eval_set = self.eval_sets[self.eval_set_key]
            self.sample_index = 0
            print(f"finished eval set {self.eval_set_key}")

        sample = self.eval_set[self.sample_index]

        return sample

    def accept(self):
        self.eval_sets_filtered[self.eval_set_key].append(self.sample)
        self.sample = self.get_next_sample()
        self.plot_sample(self.sample)

    def reject(self):
        self.sample = self.get_next_sample()
        self.plot_sample(self.sample)

    def save(self):
        # TODO: save progress (indices!)
        pickle.dump(self.eval_sets_filtered, open(f"data/filtered.p", "wb"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = EvalSetFilter()
    sys.exit(app.exec_())
