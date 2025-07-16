#2025 Makkiblog.com MIT License
#

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from markitdown import MarkItDown

class OfficeConverter:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Office to Markdown Converter")
        
        # Input frame
        input_frame = tk.Frame(self.window)
        input_frame.pack(padx=10, pady=5)
        
        tk.Label(input_frame, text="Select Office File:").pack(side=tk.LEFT)
        self.file_entry = tk.Entry(input_frame, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        
        browse_button = tk.Button(input_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.LEFT)
        
        # Convert button
        convert_button = tk.Button(self.window, text="Convert to Markdown", 
                                 command=self.convert_file)
        convert_button.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(self.window, text="")
        self.status_label.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Office Files", ".docx .xlsx .pptx")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def convert_file(self):
        try:
            input_file = self.file_entry.get().strip()
            
            if not input_file:
                raise ValueError("Please select a file first")
                
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"File not found: {input_file}")
                
            ext = os.path.splitext(input_file)[1].lower()
            if ext not in ['.docx', '.xlsx', '.pptx']:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Get input file directory and base name
            input_dir = os.path.dirname(os.path.abspath(input_file))
            base_name = os.path.basename(input_file)
            stem = os.path.splitext(base_name)[0]
            
            # Create output filename with underscores instead of spaces
            output_filename = f"{stem.replace(' ', '_')}.md"
            output_path = os.path.join(input_dir, output_filename)
            
            # Convert file
            md = MarkItDown(enable_plugins=False)
            result = md.convert(input_file)
            
            # Add Marp front matter and format content for slides
            marp_content = self.prepare_marp_content(result.text_content)
            
            # Ensure the directory exists
            os.makedirs(input_dir, exist_ok=True)
            
            # Save converted content with Marp front matter
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(marp_content)
                self.status_label.config(
                    text=f"Converted successfully to {output_filename} with Marp formatting!", 
                    fg="green"
                )
            except Exception as write_error:
                raise IOError(f"Failed to write output file: {str(write_error)}")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            print(f"Error details: {type(e).__name__}: {str(e)}")

    def prepare_marp_content(self, content):
        """Add Marp front matter and format content for slides"""
        # Marp front matter
        marp_front_matter = """---
marp: true
theme: default
paginate: true
---

"""
        # Split content by headers to create slides
        lines = content.split('\n')
        slide_content = []
        current_slide = []
        
        for line in lines:
            # If it's a header, start a new slide
            if line.startswith('# ') or line.startswith('## '):
                if current_slide:
                    slide_content.append('\n'.join(current_slide))
                    current_slide = []
                current_slide.append(line)
            else:
                current_slide.append(line)
        
        # Add the last slide
        if current_slide:
            slide_content.append('\n'.join(current_slide))
        
        # Join slides with slide separators
        formatted_content = '\n\n---\n\n'.join(slide_content)
        
        return marp_front_matter + formatted_content

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = OfficeConverter()
    app.run()