"""
A python GUI and api to combine well logs
"""

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import welly
from welly import Well
import os
import http.server
import socketserver
import threading
import json
import traceback
from matplotlib.ticker import MultipleLocator, Locator
from dlishandler import get_dlis_data, datasets_to_las
from scipy import interpolate

from plotandshow import create_plot_window

# Default aliases and styles for parsing LAS files
default_aliases = {
    "sonic": ["none", "DTC", "DT24", "DTCO", "DT", "AC", "AAC", "DTHM"],
    "ssonic": ["none", "DTSM"],
    "gr": ["none", "GR", "GRD", "CGR", "GRR", "GRCFM"],
    "sp": ["none", "SP", "SPR"],
    "resdeep": ["none", "HDRS", "LLD", "M2RX", "MLR4C", "RD", "RT90", "RLA1", "RDEP", "RLLD", "RILD", "ILD", "RT_HRLT", "RACELM"],
    "resdmed": ["none", "RILM", "ILM"],
    "resmeds": ["none"],
    "resshal": ["none", "LLS", "HMRS", "M2R1", "RS", "RFOC", "ILS", "RSFL", "RMED", "RACEHM", "RXO_HRLT"],
    "density": ["none", "ZDEN", "RHOB", "RHOZ", "RHO", "DEN", "RHO8", "BDCFM"],
    "neutron": ["none", "CNCF", "NPHI", "NEU", "TNPH", "NPHI_LIM"],
    "pe": ["none", "PEFLA", "PEF8", "PE"]
}
aliases = {}
default_styles = {
    "gr": {"color": "green", "linewidth": 1.5, "style": '-', "track": 0, "left": 0, "right": 150, "type": 'linear', "unit": 'gAPI', "fill": "left"},
    "sp": {"color": "blue", "linewidth": 1.5, "style": '--', "track": 0, "left": -1000, "right": 1000, "type": 'linear', "unit": 'mV', "fill": "none"},
    "resshal": {"color": "red", "linewidth": 1.5, "style": '-', "track": 1, "left": 0.2, "right": 200, "type": 'log', "unit": 'ohm/m', "fill": "none"},
    "resdeep": {"color": "black", "linewidth": 1.5, "style": '-.', "track": 1, "left": 0.2, "right": 200, "type": 'log', "unit": 'ohm/m', "fill": "none"},
    "resdmed": {"color": "black", "linewidth": 1.5, "style": '--', "track": 1, "left": 0.2, "right": 200, "type": 'log', "unit": 'ohm/m', "fill": "none"},
    "resmeds": {"color": "black", "linewidth": 1.5, "style": '--', "track": 1, "left": 0.2, "right": 200, "type": 'log', "unit": 'ohm/m', "fill": "none"},
    "pe": {"color": "green", "linewidth": 1.5, "style": '--', "track": 2, "left": 0, "right": 20, "type": 'linear', "unit": 'barns/electron', "fill": "none"},
    "neutron": {"color": "blue", "linewidth": 1.5, "style": '--', "track": 2, "left": 0.54, "right": -0.06, "type": 'linear', "unit": 'p.u.', "fill": "none"},
    "sonic": {"color": "black", "linewidth": 1.5, "style": '--', "track": 2, "left": 140, "right": 40, "type": 'linear', "unit": 'uspf', "fill": "none"},
    "density": {"color": "brown", "linewidth": 1.5, "style": '-', "track": 2, "left": 1.8, "right": 2.8, "type": 'linear', "unit": 'g/cc', "fill": "none"}
}


class LogPlotterApp(toga.App):
    def startup(self):
        # Create the main window
        self.main_window = toga.MainWindow(title=self.formal_name)
        
        # Create file selection buttons for Log One
        self.log_one_label = toga.Label('Log One:', style=Pack(padding=(5, 0)))
        self.file_button_one = toga.Button('Select LAS file', on_press=self.select_las_file_one, style=Pack(padding=1, flex=1))
        self.dlis_button_one = toga.Button('Select DLIS file', on_press=self.select_dlis_file_one, style=Pack(padding=1, flex=1))
        self.button_box_one = toga.Box(children=[self.file_button_one, self.dlis_button_one], style=Pack(direction=ROW))
        
        # Create file selection buttons for Log Two
        self.log_two_label = toga.Label('Log Two:', style=Pack(padding=(5, 0)))
        self.file_button_two = toga.Button('Select LAS file', on_press=self.select_las_file_two, style=Pack(padding=1, flex=1))
        self.dlis_button_two = toga.Button('Select DLIS file', on_press=self.select_dlis_file_two, style=Pack(padding=1, flex=1))
        self.button_box_two = toga.Box(children=[self.file_button_two, self.dlis_button_two], style=Pack(direction=ROW))
        
        # Create merge and plot button
        self.merge_plot_button = toga.Button('Merge and Plot', on_press=self.merge_and_plot, style=Pack(padding=5))
        # Create merge and save button
        self.save_merged_las_button = toga.Button('Save Merged LAS', on_press=self.save_merged_las, style=Pack(padding=5))
        
        # Create text inputs for displaying and modifying aliases and styles
        self.aliases_input = toga.MultilineTextInput(
            value=json.dumps(default_aliases, indent=4), style=Pack(flex=1, padding=5)
        )
        self.styles_input = toga.MultilineTextInput(
            value=json.dumps(default_styles, indent=4), style=Pack(flex=1, padding=5)
        )
        
        # Create input fields for interpolation
        self.interpolate_top_label = toga.Label('Interpolate Top:', style=Pack(padding=(5, 0)))
        self.interpolate_top_input = toga.TextInput(style=Pack(flex=1))
        self.interpolate_bottom_label = toga.Label('Interpolate Bottom:', style=Pack(padding=(5, 0)))
        self.interpolate_bottom_input = toga.TextInput(style=Pack(flex=1))
        
        # Create decimate and interpolate button
        self.decimate_interpolate_button = toga.Button('Decimate and Interpolate', on_press=self.decimate_and_interpolate, style=Pack(padding=5))

        # Create merge and plot button
        self.merge_plot_button = toga.Button('Merge and Plot', on_press=self.merge_and_plot, style=Pack(padding=5))
        
        # Labels
        self.aliases_label = toga.Label('Aliases:', style=Pack(padding=(5, 0)))
        self.styles_label = toga.Label('Styles:', style=Pack(padding=(5, 0)))
        
        self.aliasbox = toga.Box(children=[self.aliases_label, self.aliases_input], style=Pack(direction=COLUMN,padding=10,flex=1))
        self.stylesbox = toga.Box(children=[self.styles_label, self.styles_input], style=Pack(direction=COLUMN,padding=10,flex=1))
        self.combobox = toga.Box(children=[self.aliasbox, self.stylesbox], style=Pack(direction=ROW,padding=10,flex=1))
        
        # Create the main box for the first page
        self.main_box = toga.Box(
            children=[
                self.log_one_label, self.button_box_one,
                self.log_two_label, self.button_box_two,
                self.interpolate_top_label, self.interpolate_top_input,
                self.interpolate_bottom_label, self.interpolate_bottom_input,
                self.decimate_interpolate_button,
                self.merge_plot_button,
                self.save_merged_las_button,  # Add this line
                self.combobox
            ],
            style=Pack(direction=COLUMN, padding=10)
        )
        
        # Initialize dataframes
        self.dataframe1 = None
        self.dataframe2 = None
        self.interpolate_flag = False
        self.interpolate_top = None
        self.interpolate_bottom = None
        
        # Create the second page with a back button and webview
        self.back_button = toga.Button('Back', on_press=self.show_main_page, style=Pack(padding=5))
        self.webview = toga.WebView(style=Pack(flex=1))
        
        self.plot_box = toga.Box(children=[self.back_button, self.webview], style=Pack(direction=COLUMN))
        
        # Show the main page
        self.main_window.content = self.main_box
        self.main_window.show()

    def show_main_page(self, widget=None):
        self.stop_server()
        self.main_window.content = self.main_box
        

    def show_plot_page(self, widget=None):
        self.main_window.content = self.plot_box

    async def select_las_file_one(self, widget):
        await self.select_file(file_type='las', log_number=1)

    async def select_dlis_file_one(self, widget):
        await self.select_file(file_type='dlis', log_number=1)

    async def select_las_file_two(self, widget):
        await self.select_file(file_type='las', log_number=2)

    async def select_dlis_file_two(self, widget):
        await self.select_file(file_type='dlis', log_number=2)

    async def select_file(self, file_type, log_number):
        try:
            selected_file = await self.main_window.open_file_dialog(
                title=f'Select {file_type.upper()} file for Log {log_number}',
                multiple_select=False,
                file_types=[file_type]
            )

            if selected_file:
                self.process_file(selected_file, file_type, log_number)
                
        except Exception as e:
            self.main_window.error_dialog('Error', str(e))

    def process_file(self, file_path, file_type, log_number):
        try:
            global aliases
            aliases = json.loads(self.aliases_input.value)

            if file_type == 'las':
                well = Well.from_las(file_path)
                df = well.df()
                header = well.header
                units = {}  # Empty dictionary for LAS files
            elif file_type == 'dlis':
                df, units, header, pdict = get_dlis_data(file_path, aliases)
                # Ensure header is a DataFrame for DLIS files
                if not isinstance(header, pd.DataFrame):
                    header = pd.DataFrame(header)
            else:
                raise ValueError("Unsupported file type")
            
            if log_number == 1:
                self.dataframe1 = df
                self.log1_header = header
                self.log1_units = units
            else:
                self.dataframe2 = df
                self.log2_header = header  # Store header for log2
                self.log2_units = units
            
            styles = json.loads(self.styles_input.value)
            mnemonic_map = {}
            for key, values in aliases.items():
                for value in values:
                    if value in df.columns:
                        mnemonic_map[value] = key

            updated_styles = {mnemonic: styles[standard_mnemonic] for mnemonic, standard_mnemonic in mnemonic_map.items() if standard_mnemonic in styles}
            df = df[[col for col in df.columns if col in updated_styles]]

            # Check if the "neutron" mnemonic column exists and if its nanmean exceeds the threshold
            neutron_mnemonics = [mnemonic for mnemonic, standard_mnemonic in mnemonic_map.items() if standard_mnemonic == "neutron"]
            for neutron_mnemonic in neutron_mnemonics:
                if neutron_mnemonic in df.columns and np.nanmean(df[neutron_mnemonic]) > 1:
                    df[neutron_mnemonic] /= 100
            
            create_plot_window(self, df, updated_styles, title=f'Log{log_number}')
            #self.main_window.info_dialog('Success', f'Log {log_number} loaded successfully')
        
        except Exception as e:
            traceback.print_exc()
            self.main_window.error_dialog('Error', str(e))
        
    def decimate_and_interpolate(self, widget):
        try:
            self.interpolate_top = float(self.interpolate_top_input.value)
            self.interpolate_bottom = float(self.interpolate_bottom_input.value)
            self.interpolate_flag = True
            self.main_window.info_dialog('Success', 'Decimation and interpolation settings applied')
        except ValueError:
            self.main_window.error_dialog('Error', 'Please enter valid numbers for interpolation range')
    
    

    def merge_and_plot(self, widget):
        if self.dataframe1 is None or self.dataframe2 is None:
            self.main_window.error_dialog('Error', 'Please load both logs before merging')
            return

        try:
            # Align columns by renaming based on aliases or actual names
            common_columns = self.dataframe1.columns.intersection(self.dataframe2.columns)
            all_columns = self.dataframe1.columns.union(self.dataframe2.columns)
            
            # Merge the dataframes with interpolation
            merged_df = pd.concat([self.dataframe1, self.dataframe2], axis=0)

            # Sort by index (depth) and remove duplicate indices
            merged_df = merged_df.sort_index().loc[~merged_df.index.duplicated(keep='first')]

            # Interpolate data onto uniform depth array
            merged_df = merged_df.interpolate(method='linear', limit_direction='both', axis=0)
            
            # Ensure strict monotonicity
            merged_df = merged_df[merged_df.index.to_series().diff().fillna(1) > 0]

            # Calculate average sample spacing
            avg_spacing = 0.15

            # Create new uniform depth array
            new_depth = np.arange(merged_df.index.min(), merged_df.index.max(), avg_spacing)

            # Interpolate data onto new depth array
            resampled_df = pd.DataFrame(index=new_depth, columns=merged_df.columns)
            for column in merged_df.columns:
                f = interpolate.interp1d(merged_df.index, merged_df[column], kind='linear', bounds_error=False, fill_value='extrapolate')
                resampled_df[column] = f(new_depth)

            # Store the result
            self.merged_df = resampled_df
            
            # Merge units
            self.merged_units = self.log1_units.copy() if hasattr(self, 'log1_units') else {}
            if hasattr(self, 'log2_units'):
                for curve, unit in self.log2_units.items():
                    if curve not in self.merged_units:
                        self.merged_units[curve] = unit

            if self.interpolate_flag:
                self.merged_df = self.apply_decimation_and_interpolation(self.merged_df)

            # Use the new PlotAndShow module
            styles = json.loads(self.styles_input.value)
            mnemonic_map = {}
            for key, values in aliases.items():
                for value in values:
                    if value in self.merged_df.columns:
                        mnemonic_map[value] = key

            updated_styles = {mnemonic: styles[standard_mnemonic] for mnemonic, standard_mnemonic in mnemonic_map.items() if standard_mnemonic in styles}
            self.merged_df = self.merged_df[[col for col in self.merged_df.columns if col in updated_styles]]

            # Check if the "neutron" mnemonic column exists and if its nanmean exceeds the threshold
            neutron_mnemonics = [mnemonic for mnemonic, standard_mnemonic in mnemonic_map.items() if standard_mnemonic == "neutron"]
            for neutron_mnemonic in neutron_mnemonics:
                if neutron_mnemonic in self.merged_df.columns and np.nanmean(self.merged_df[neutron_mnemonic]) > 1:
                    self.merged_df[neutron_mnemonic] /= 100
            
            #self.merged_df.to_csv('merged.csv', sep=',', encoding='utf-8', index=False, header=True)

            # Create the plot window
            create_plot_window(self, self.merged_df, updated_styles, title='Spliced')

            # Inform the user about interpolation
            #self.main_window.info_dialog('Data Interpolated', 
            #    'Data has been interpolated to fill missing values.')

        except Exception as e:
            traceback.print_exc()
            self.main_window.error_dialog('Error', str(e))

        
    def apply_decimation_and_interpolation(self, df):
        # Create a mask for the interpolation range
        mask = (df.index >= self.interpolate_top) & (df.index <= self.interpolate_bottom)
        
        # Replace values in the interpolation range with NaN
        df.loc[mask] = np.nan
        
        # Interpolate the NaN values
        df = df.interpolate(method='linear')
        
        return df
            

    async def save_merged_las(self, widget):
        if not hasattr(self, 'merged_df') or self.merged_df is None:
            self.main_window.error_dialog('Error', 'Please merge the logs first')
            return
        try:
            save_path = await self.main_window.save_file_dialog(
                title="Save Merged LAS File",
                suggested_filename="merged_log.las",
                file_types=['las']
            )
            if save_path:
                # Use header from log2
                header = self.log2_header if hasattr(self, 'log2_header') else pd.DataFrame()
                c_units = self.merged_units if hasattr(self, 'merged_units') else {}

                # Ensure WELL section exists in header
                if 'WELL' not in header.index:
                    header.loc['WELL', 'UWI'] = 'Unknown'

                # Update the WELL section with the new TD (total depth)
                self.merged_df['DEPT'] = self.merged_df.index
                if 'WELL' in header.index:
                    header.loc['WELL', 'STRT'] = self.merged_df['DEPT'].iloc[0]
                    header.loc['WELL', 'STOP'] = self.merged_df['DEPT'].iloc[-1]
                
                self.merged_df = self.merged_df[['DEPT'] + [col for col in self.merged_df.columns if col != 'DEPT']]
                datasets_to_las(save_path, {'Header': header, 'Curves': self.merged_df}, c_units)
                self.main_window.info_dialog('Success', f'Merged LAS file saved to {save_path}')
        except Exception as e:
            traceback.print_exc()
            self.main_window.error_dialog('Error', str(e))

    def start_server(self):
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)

        self.server = socketserver.TCPServer(('localhost', 8000), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def stop_server(self):
        if hasattr(self, 'server'):
            #self.server.shutdown()
            print("shutdown")
            self.server.server_close()
            print("server close")
            #self.server_thread.join()
            print("thread join")
            del self.server_thread
            print("del server")
            del self.server
            print("server stopped and thread joined")

def main():
    app = LogPlotterApp('LogPlotter', 'in.rocklab.logplotter')
    app.main_loop()

if __name__ == '__main__':
    main()
