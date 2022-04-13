from keras.models import load_model
from tkinter import *
import win32gui
from PIL import ImageGrab, ImageTk,Image
import numpy as np


class App:
	def __init__(self):
		self.root = Tk()
		self.root.geometry('500x500')
		self.label_pred = Label(self.root,text='predict')
		self.label_pred.pack(anchor=CENTER)

		self.button_get = Button(self.root,text='write',command=self.new_window)
		self.button_get.pack()

		self.label_img = Label(self.root)
		self.label_img.pack()

		
		self.root.mainloop()
	
	def new_window(self):
		self.new = Toplevel(self.root)
		self.new.attributes("-topmost", True)
		self.new.resizable(0,0)
		self.canvas = Canvas(self.new, width=280, height=280, bg='black', cursor='circle')
		self.canvas.pack()
		self.canvas.old_coords = None
		self.button_get.config(state=DISABLED)
	
		self.new.bind('<Button-1>',self.first_point)
		self.new.bind('<B1-Motion>',self.draw)

		def close():
			self.button_get.config(state=NORMAL)
			self.CaptureScreen()
			self.new.destroy()
			
		self.new.protocol("WM_DELETE_WINDOW", func= close)

	
	def ModelPredict(self):
		self.model = load_model('cnn_model.h5')
		im = Image.open('photo.png').resize((28,28))
		im = im.convert('L')
		im_data = np.array(im).reshape(1,28,28,1)
		output = self.model.predict(im_data)
		pred = np.argmax(output)
		self.label_pred.config(text=pred)


	def first_point(self,event):
		self.canvas.old_coords = event.x, event.y

	def draw(self,event):
		x, y = event.x, event.y
		if self.canvas.old_coords:
			x1, y1 = self.canvas.old_coords
			self.canvas.create_line(x, y, x1, y1, fill='white', width=10, capstyle='round')
		self.canvas.old_coords = x, y

	def CaptureScreen(self):
		HWND = win32gui.GetFocus() #windows
		rect=win32gui.GetWindowRect(HWND) #coordinate
		self.im=ImageGrab.grab(rect) #screenshot
		self.im.save("photo.png",'png') 
		self.draw_img = ImageTk.PhotoImage(Image.open('photo.png'))
		self.label_img.config(image=self.draw_img)
		self.ModelPredict()


if __name__ =='__main__':
	
	a = App()