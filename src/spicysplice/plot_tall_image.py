import toga
from toga.style import Pack
from toga.style.pack import COLUMN
import os
from matplotlib import pyplot as plt
import math

class TallImageApp(toga.App):
    def startup(self):
        self.main_window = toga.MainWindow(title=self.name)

        # Create a tall image
        #
        
        dpi = 100
        aspect = 5
        scale=500
        tlen = 1000  #total length of log in m
        tlenin= round(tlen*39.3701)/500 #so 39370/500
        
        height = round(tlenin*dpi) #height of final image in pixels
        width = math.ceil(height/aspect) #width of final image in pixels
        width = round(width + 0.5*width)
        print(width,height)
        self.create_tall_image(width,height,dpi)
        # Create an ImageView
        image = toga.Image('tall_image.png')
        self.image_view = toga.ImageView(image)

        # Create a ScrollContainer
        self.scroll_container = toga.ScrollContainer(content=self.image_view, style=Pack(direction='row', padding=0, flex=1),horizontal=False)

        # Main box
        main_box = toga.Box(
            children=[self.scroll_container],
            style=Pack(direction=COLUMN, padding=0, width=width)
        )

        self.main_window.content = main_box
        self.main_window.show()

        
    def create_tall_image(self,width,height,dpi):
        from Plotter3 import plot_logs
        import pandas as pd
        import numpy as np
        data = pd.DataFrame({
            'log1': np.random.random(100) * 150,
            'log2': np.random.random(100) * 200,
        }, index=np.linspace(0, 1000, 100))

        styles = {
            'log1': {"color": "green", "linewidth": 1.5, "style": '-', "track": 0, "left": 0, "right": 150, "type": 'linear', "unit": "m/s"},
            'log2': {"color": "blue", "linewidth": 1.5, "style": '-', "track": 1, "left": 0, "right": 200, "type": 'linear', "unit": "m/s"},
        }

        points = pd.DataFrame({
            'point1': np.random.random(10) * 100,
            'point2': np.random.random(10) * 50,
        }, index=np.linspace(0, 1000, 10))

        pointstyles = {
            'point1': {'color': 'red', 'pointsize': 50, 'symbol': 'o', 'track': 0, 'left': 0, 'right': 100, 'type': 'linear', 'unit': "Mpa", 'uptosurface': True},
            'point2': {'color': 'purple', 'pointsize': 50, 'symbol': 'o', 'track': 1, 'left': 0, 'right': 50, 'type': 'linear', 'unit': "Mpa"},
        }

        fig,axes = plot_logs(data, styles, points, pointstyles, y_min=0, y_max=data.index[-1], plot_labels=False, figsize=(width/dpi, height/dpi), label_height=20, dpi=dpi)
        # Create a tall image (e.g., 100x1000 pixels)
        # Save the image
        plt.savefig('tall_image.png')



def main():
    return TallImageApp('Tall Image Demo', 'org.example.tallimage')

if __name__ == '__main__':
    app = main()
    app.main_loop()
    