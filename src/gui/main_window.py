import tkinter as tk
from src.gui.pdf_converter_ui import PDFConverterUI

TITLE = "PDF to Image Converter"
WIDTH = 600
HEIGHT = 600


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITLE)
        self.geometry(f"{WIDTH}x{HEIGHT}")

        self.initialize_ui()

    def initialize_ui(self):
        # Add the PDFConverterUI component
        pdf_converter_ui = PDFConverterUI(self)
        pdf_converter_ui.pack()
