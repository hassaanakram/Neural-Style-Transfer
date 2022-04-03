# THIS FILE CONTAINS ALL THE OBJECTS THAT ARE REQUIRED IN MULTIPLE FILES
# TO ANYONE WHO IS READING THIS CODE IN __future__, PARDON FOR USING THIS
# METHOD.

# GUI widgets
import tkinter as tk
root = tk.Tk()
textVar = tk.StringVar()
status_lbl = tk.Label(root, textvariable=textVar, font=("",10))
status_lbl.grid(row=12, column=0)
tv_weight_sel = tk.Entry(root)
content_weight_sel = tk.Entry(root)
style_weight_sel = tk.Entry(root)
epoch_sel = tk.Entry(root)
steps_epoch_sel = tk.Entry(root)
width_sel = tk.Entry(root)
height_sel = tk.Entry(root)
# Variables
content_image = 0
style_image = 0
output_dir = ''
epochs = 10
steps_per_epoch = 300
tv_weight = 1e8
content_weight = 10000
style_weight = 0.01
learning_rate = 0.02
beta_1 = 0.99
beta_2 = 0.999
epsilon = 0.10
width = 0
height = 0
max_dim = 512 # Scaling factor control
