from variables import *
import utils
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Creating the gui using tk
root.geometry("300x500")
root.title('Neural Style Transfer')
root.resizable(False, False)

# Adding gui widgets
content_image_but = tk.Button(root, text='Choose Content Image', padx=35,
                          pady=6, fg="White", bg="grey", command=utils.get_content_img, font=("",9),
                              width=10)
content_image_but.grid(row=8, column=0)

style_image_but = tk.Button(root, text='Choose Style Image', padx=35,
                          pady=6, fg="White", bg="grey", command=utils.get_style_img, width=10)
style_image_but.grid(row=9, column=0)

tv_lbl = tk.Label(root, text="Total variation weight",font=("",10)).grid(row=1,column=0)
tv_weight_sel.insert(0, "100000000")
tv_weight_sel.grid(row=1, column=1)
tv_weight = float(tv_weight_sel.get())

style_lbl = tk.Label(root, text="Style weight",font=("",10)).grid(row=2,column=0)
style_weight_sel.insert(0, "0.10")
style_weight_sel.grid(row=2, column=1)
style_weight = float(style_weight_sel.get())

content_lbl = tk.Label(root, text="Content weight",font=("",10)).grid(row=3,column=0)
content_weight_sel.insert(0, "10000")
content_weight_sel.grid(row=3, column=1)
content_weight = float(content_weight_sel.get())

epoch_lbl = tk.Label(root, text="Iterations to run", font=("",10)).grid(row=4,column=0)
epoch_sel.insert(0, "10")
epoch_sel.grid(row=4, column=1)
epochs = int(epoch_sel.get())

steps_epoch_lbl = tk.Label(root, text="Steps per iteration", font=("",10)).grid(row=5,column=0)
steps_epoch_sel.insert(0, "100")
steps_epoch_sel.grid(row=5, column=1)
steps_per_epoch = int(steps_epoch_sel.get())

width_lbl = tk.Label(root, text="Image width",font=("",10)).grid(row=6,column=0)
width_sel.insert(0, "512")
width_sel.grid(row=6, column=1)
width = int(width_sel.get())

height_lbl = tk.Label(root, text="Image height",font=("",10)).grid(row=7,column=0)
height_sel.insert(0, "512")
height_sel.grid(row=7, column=1)
height = int(height_sel.get())

sel_output_but = tk.Button(root, text='Select output file', padx=35,
                          pady=6, fg="White", bg="grey", width=10, command=utils.get_output_dir)
sel_output_but.grid(row=10, column=0)
lets_go_but = tk.Button(root, text='Lets go!', padx=35,
                          pady=6, fg="White", bg="grey", width=10, command=utils.commence)
lets_go_but.grid(row=11, column=0)

textVar.set("Status")
root.mainloop()