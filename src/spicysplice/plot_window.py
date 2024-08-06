import toga
from toga.style import Pack
from toga.style.pack import COLUMN
import os
from matplotlib import pyplot as plt
import math

from Plotter3 import plot_logs
import pandas as pd
import numpy as np

class plotandshow(toga.Window):
    def startup(self,data,styles,points=None,pointstyles=None,scale=500,aspect=10,tlen=1000,dpi=100):
        #self.main_window = toga.MainWindow(title=self.name)

        # Create a tall image
        #
        
        #dpi = 100
        #aspect = 5
        #scale=500
        #tlen = 1000  #total length of log in m
        tlenin= round(tlen*39.3701)/500 #so 39370/500
        
        height = round(tlenin*dpi) #height of final image in pixels
        width = math.ceil(height/aspect) #width of final image in pixels
        width = round(width + 0.5*width)
        print(width,height)
        self.create_tall_image(width,height,dpi)
        # Create an ImageView
        fig,axes = plot_logs(data, styles, points, pointstyles, y_min=0, y_max=data.index[-1], plot_labels=False, figsize=(width/dpi, height/dpi), label_height=20, dpi=dpi)
        plt.savefig('log_image.png')
        image = toga.Image('log_image.png')
        self.image_view = toga.ImageView(image)

        # Create a ScrollContainer
        self.scroll_container = toga.ScrollContainer(content=self.image_view, style=Pack(direction='row', padding=0, flex=1),horizontal=False)

        # Main box
        main_box = toga.Box(
            children=[self.scroll_container],
            style=Pack(direction=COLUMN, padding=0, width=width)
        )

        self.window.content = main_box
        self.window.show()