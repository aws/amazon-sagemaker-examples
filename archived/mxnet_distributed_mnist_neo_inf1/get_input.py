import math
from tkinter import *
from tkinter import messagebox

import numpy as np

# track welcome status so we only display the welcome message once
welcome = False

# saved array size
width = 28.0
height = 28.0

# scale from array size to GUI size (for readability, ease of drawing/use)
scale = 10.0

# internal array data, initialize to zeros
array = np.zeros((int(width), int(height)))


def update(x, y):
    """
    Update the internal array with given x/y coordinate
    """
    global array
    # compute location in array using scaling factor
    real_x = math.floor(x / scale)
    real_y = math.floor(y / scale)
    # update array with value '1'
    array[real_y][real_x] = 1


def paint(event):
    """
    Event handler for mouse motion
    Update the GUI with dots for given x/y coordinate
    """
    global canvas
    # compute size of dot based on scaling factor
    x1, y1 = (event.x - scale), (event.y - scale)
    x2, y2 = (event.x + scale), (event.y + scale)
    # draw dot
    canvas.create_oval(x1, y1, x2, y2, fill="black")
    # update internal array
    update(event.x, event.y)


def save(event):
    """
    Event handler for mouse button release
    Save the internal array to file
    """
    global array
    # save
    np.save("input.npy", array)
    # print array data to console (for understanding)
    for y in range(int(height)):
        s = ""
        for x in range(int(width)):
            s += str(int(array[y][x])) + " "
        print(s)
    # remind user of file name
    print("saved to input.npy")


def clear(event):
    """
    Event handler for mouse click
    Clear internal array and drawing canvas to prepare for new digit
    """
    global canvas
    global array
    # clear internal array
    array = np.zeros((int(width), int(height)))
    # clear drawing canvas
    canvas.delete("all")


def focus(event):
    """
    Event handler for gaining focus on window
    Display welcome message the first time
    """
    global welcome
    # only display message once
    if not welcome:
        # open info pop up with instructions
        messagebox.showinfo(
            "Instructions",
            "Click and drag to draw a digit as a single continuous stroke. Release the mouse to save 28x28 numpy array as 'input.npy' file on disk. Clicking and dragging again will reset and start over. Close the window to exit the python process",
            parent=master,
        )
        # set flag to not repeat
        welcome = True


#######################
# setup GUI
#######################
# setup window
try:
    master = Tk()
except TclError as ex:
    msg = ex.args[0]
    if "display" in msg or "DISPLAY" in msg:
        print(
            "This script must be run in a terminal shell and on a machine with a GUI display like your local computer."
        )
        exit(1)
master.title("Draw a digit")
# register focus handler on window
master.bind("<FocusIn>", focus)

# setup drawing canvas
canvas = Canvas(master, width=width * scale, height=height * scale, background="gray75")
# if user resizes window, dont scale canvas
canvas.pack(expand=NO, fill=NONE)

# register handlers for canvas
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease>", save)
canvas.bind("<Button-1>", clear)

# message at bottom of window
message = Label(master, text="Press and Drag the mouse to draw")
message.pack(side=BOTTOM)

# main GUI loop
mainloop()
