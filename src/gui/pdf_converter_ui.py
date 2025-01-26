import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from src.processors.pdf_converter import PDFConverter


class PDFConverterUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        self.converter = PDFConverter(overwrite=True)
        self.create_widgets()

    def create_widgets(self):
        # Input selection
        tk.Label(self, text="Select PDF or Directory:", font=("Arial", 12)).pack(pady=10)
        self.file_path_entry = tk.Entry(self, width=50)
        self.file_path_entry.pack(pady=5)
        tk.Button(self, text="Browse...", command=self.browse_files).pack(pady=5)
        tk.Button(self, text="Clear", command=lambda: self.file_path_entry.delete(0, tk.END)).pack(pady=5)

        # Output selection (Optional)
        tk.Label(self, text="Output Directory (Optional):", font=("Arial", 12)).pack(pady=10)
        self.output_dir_entry = tk.Entry(self, width=50)
        self.output_dir_entry.pack(pady=5)
        self.output_dir_entry.insert(0, str(self.converter.output_dir))
        tk.Button(self, text="Browse...", command=self.browse_output_directory).pack(pady=5)
        tk.Button(self, text="Clear", command=lambda: self.output_dir_entry.delete(0, tk.END)).pack(pady=5)

        # Options
        self.single_image = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Single image per PDF", variable=self.single_image).pack(pady=10)

        # Convert Button
        tk.Button(self, text="Convert", command=self.convert_pdf).pack(pady=10)

    def browse_files(self):
        """Open a file dialog to select a PDF or directory and set a default output path."""
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if not file_path:  # if user cancels, returns an empty string
            file_path = filedialog.askdirectory()
        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, file_path)

    def browse_output_directory(self):
        """Open a file dialog to select an output directory."""
        output_dir = filedialog.askdirectory()
        if output_dir:  # Ensure a directory was selected
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, output_dir)
            self.converter.output_dir = Path(output_dir)

    def convert_pdf(self):
        """Convert selected PDF or all PDFs in a directory to images."""
        input_path_entry = self.file_path_entry.get()
        if not input_path_entry:
            messagebox.showerror("Error", "Please select a PDF or directory.")
            return
        input_path = Path(input_path_entry)
        if not input_path.exists():
            messagebox.showerror("Error", f"File or directory not found: {input_path}")
            return

        output_path_entry = self.output_dir_entry.get()
        output_path = Path(output_path_entry) if output_path_entry else None

        single_image = self.single_image.get()
        try:
            self.converter.convert(input_path, output_path, single_image)
            messagebox.showinfo("Success", "PDF conversion completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during conversion: {e}")
