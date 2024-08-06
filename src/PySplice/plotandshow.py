import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import os
import shutil
from matplotlib import pyplot as plt
import math
from Plotter3 import plot_logs, choptop
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator, Locator

class PlotAndShow:
    def __init__(self, app):
        self.app = app

    def create_window(self, data, styles, points=None, pointstyles=None, scale=500, aspect=10, tlen=1000, dpi=100, title='log'):
        # Calculate dimensions
        tlenin = round(tlen * 39.3701) / 500
        height = round(tlenin * dpi)
        width = math.ceil(height / aspect)
        width = round(width + 0.5 * width)

        # Create the plot
        fig, axes = plot_logs(data, styles, points, pointstyles, y_min=data.index[0], y_max=data.index[-1], 
                              plot_labels=False, width = width/dpi, height=height/dpi, label_height=20, dpi=dpi)
        
        for ax in axes:
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(which='minor', axis='y', color='gray', linestyle='-', linewidth=0.25)
            ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5)
        
        # Save the plot
        plot_filename = f'{title}_image.png'
        plt.savefig(plot_filename, dpi=dpi)
        plt.close(fig)

        padT = 9
        padB = 8.5
        choptop(padT*dpi, padB*dpi, plot_filename)

        # Rename TopLabel.png
        top_label_filename = f'{title}_TopLabel.png'
        if os.path.exists('TopLabel.png'):
            if os.path.exists(top_label_filename):
                # If the destination file exists, remove it first
                os.remove(top_label_filename)
            shutil.move('TopLabel.png', top_label_filename)

        # Create a new window
        window = toga.Window(title=title)

        # Create ImageViews
        top_label_image = toga.Image(top_label_filename)
        top_label_view = toga.ImageView(top_label_image)

        main_image = toga.Image(plot_filename)
        main_image_view = toga.ImageView(main_image)

        # Create a ScrollContainer for the main image
        scroll_container = toga.ScrollContainer(content=main_image_view, style=Pack(direction=ROW, padding=0, flex=1), horizontal=False)

        # Main box
        main_box = toga.Box(
            children=[
                top_label_view,
                scroll_container
            ],
            style=Pack(direction=COLUMN, padding=0, width=width)
        )

        window.content = main_box
        window.show()

        return window

def create_plot_window(app, data, styles, points=None, pointstyles=None, scale=500, aspect=10, tlen=1000, dpi=100, title='log'):
    plotter = PlotAndShow(app)
    return plotter.create_window(data, styles, points, pointstyles, scale, aspect, tlen, dpi, title)