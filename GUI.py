from tkinter import *
from tkinter.ttk import *
import demo
window = Tk()
window.geometry('300x300')
window.title("功能选择")
btn1 = Button(window, text='demo1', command=lambda :demo.run(1))
btn2 = Button(window, text='demo2', command=lambda :demo.run(2))
btn3 = Button(window, text='demo3', command=lambda :demo.run(3))
btn4 = Button(window, text='demo4', command=lambda :demo.run(4))
btn1.place(x=100, y=20)
btn2.place(x=100, y=60)
btn3.place(x=100, y=100)
btn4.place(x=100, y=140)
window.mainloop()
