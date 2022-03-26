from tkinter import *
from tkinter.ttk import *
import PIL
from PIL import Image, ImageTk

import demo
window = Tk()
window.geometry('560x500')
window.title("功能选择")
window.resizable(width=False, height=False)
btn1 = Button(window, text='demo1', command=lambda :demo.run(1))
btn2 = Button(window, text='demo2', command=lambda :demo.run(2))
btn3 = Button(window, text='demo3', command=lambda :demo.run(3))
btn4 = Button(window, text='demo4', command=lambda :demo.run(4))
btn1.place(x=90, y=200)
btn2.place(x=380, y=200)
btn3.place(x=90, y=430)
btn4.place(x=380, y=430)

im1 = Image.open(".\img\p1.jpg")
img1 = ImageTk.PhotoImage(im1.resize((240, 160), Image.ANTIALIAS))
Label(window, image=img1).place(x=20, y=20)
im2 = Image.open(".\img\p2.jpg")
img2 = ImageTk.PhotoImage(im2.resize((240, 160), Image.ANTIALIAS))
Label(window, image=img2).place(x=300, y=20)
im3 = Image.open(".\img\p3.jpg")
img3 = ImageTk.PhotoImage(im3.resize((240, 160), Image.ANTIALIAS))
Label(window, image=img3).place(x=20, y=250)
im4 = Image.open(".\img\p4.jpg")
img4 = ImageTk.PhotoImage(im4.resize((240, 160), Image.ANTIALIAS))
Label(window, image=img4).place(x=300, y=250)

window.mainloop()
