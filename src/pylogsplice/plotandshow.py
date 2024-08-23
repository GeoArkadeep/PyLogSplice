import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import os
import shutil
from matplotlib import pyplot as plt
import math
from Plotter3 import plot_logs_labels, choptop
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator, Locator

import mpld3
from mpld3._server import serve

class PlotAndShow:
    def __init__(self, app):
        self.app = app
        self.top_label_view = None
        self.main_image_view = None
        self.titleLog = None
        self.titleLabel = None

    def create_window(self, data, styles, points=None, pointstyles=None, scale=500, aspect=10, tlen=1000, dpi=100, title='log'):
        # Calculate dimensions
        tlenin = round(tlen * 39.3701) / 500
        height = round(tlenin * dpi)
        width = math.ceil(height / aspect)
        width = round(width + 0.5 * width)

        # Create the plot
        fig, axes = plot_logs_labels(data, styles, points, pointstyles, y_min=data.index[0], y_max=data.index[-1], width = width/dpi, height=height/dpi, label_height=20, dpi=dpi)

        """
        for ax in axes:
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(which='minor', axis='y', color='gray', linestyle='-', linewidth=0.25)
            ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5)
        """
        
        # Save the plot
        plot_filename = f'{title}_image.png'
        self.titleLog = f'{title}_image.png'
        plt.savefig(plot_filename, dpi=dpi)
                
        plt.close(fig)

        padT = 9
        padB = 8.5
        choptop(padT*dpi, padB*dpi, plot_filename)

        # Rename TopLabel.png
        top_label_filename = f'{title}_TopLabel.png'
        self.titleLabel = f'{title}_TopLabel.png'
        if os.path.exists('TopLabel.png'):
            if os.path.exists(top_label_filename):
                # If the destination file exists, remove it first
                os.remove(top_label_filename)
            shutil.move('TopLabel.png', top_label_filename)

        # Create a new window
        window = toga.Window(title=title, on_close=self.on_log_window_close)

        # Create ImageViews
        top_label_image = toga.Image(top_label_filename)
        self.top_label_view = toga.ImageView(top_label_image)

        main_image = toga.Image(plot_filename)
        self.main_image_view = toga.ImageView(main_image)

        # Create a ScrollContainer for the main image
        scroll_container = toga.ScrollContainer(content=self.main_image_view, style=Pack(direction=ROW, padding=0, flex=1), horizontal=False)

        # Main box
        main_box = toga.Box(
            children=[
                self.top_label_view,
                scroll_container
            ],
            style=Pack(direction=COLUMN, padding=0, width=width)
        )

        window.content = main_box
        window.show()

        return window

    def on_log_window_close(self, window):
            # Clear references to ImageViews to release file handles
            self.top_label_view.image = None
            self.main_image_view.image = None
            if os.path.exists(self.titleLog):
                os.remove(self.titleLog)
            if os.path.exists(self.titleLabel):
                os.remove(self.titleLabel)
            window.close()

def create_plot_window(app, data, styles, points=None, pointstyles=None, scale=500, aspect=10, tlen=1000, dpi=100, title='log'):
    plotter = PlotAndShow(app)
    return plotter.create_window(data, styles, points, pointstyles, scale, aspect, tlen, dpi, title)