import tkinter as tk
from tkinter import filedialog

# Create a Tkinter root window (this is required for the file dialog)
root = tk.Tk()
root.withdraw()

# Open a file dialog window and allow the user to select a file
file_path = filedialog.askopenfilename()

# Print the selected file path (or an error message if no file was selected)
if file_path:
    print(f"Selected file: {file_path}")
else:
    print("No file selected")
