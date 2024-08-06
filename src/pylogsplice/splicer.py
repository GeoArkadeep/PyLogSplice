import pandas as pd
import numpy as np
from scipy import interpolate
from welly import Well
from dlishandler import get_dlis_data, datasets_to_las

class LogSplicer:
    def __init__(self):
        self.dataframe1 = None
        self.dataframe2 = None
        self.log1_header = None
        self.log2_header = None
        self.log1_units = None
        self.log2_units = None
        self.merged_df = None
        self.merged_units = None
        self.interpolate_top = None
        self.interpolate_bottom = None

    def load_file(self, file_path, aliases, log_number):
        if file_path.lower().endswith('.las'):
            well = Well.from_las(file_path)
            df = well.df()
            header = well.header
            units = {}  # Empty dictionary for LAS files
        elif file_path.lower().endswith('.dlis'):
            df, units, header, _ = get_dlis_data(file_path, aliases)
            if not isinstance(header, pd.DataFrame):
                header = pd.DataFrame(header)
        else:
            raise ValueError("Unsupported file type. Please use .las or .dlis files.")
        
        if log_number == 1:
            self.dataframe1 = df
            self.log1_header = header
            self.log1_units = units
        else:
            self.dataframe2 = df
            self.log2_header = header
            self.log2_units = units
        
        return df

    def set_interpolation_range(self, top, bottom):
        self.interpolate_top = top
        self.interpolate_bottom = bottom

    def merge_logs(self):
        if self.dataframe1 is None or self.dataframe2 is None:
            raise ValueError("Please load both logs before merging")

        # Concatenate dataframes
        dataframe2_trimmed = self.dataframe2[~self.dataframe2.index.isin(self.dataframe1.index)]
        merged_df = pd.concat([self.dataframe1, dataframe2_trimmed])

        # Sort the merged dataframe by index (depth) and remove duplicate indices
        merged_df = merged_df.sort_index().loc[~merged_df.index.duplicated(keep='first')]

        # Ensure strict monotonicity
        merged_df = merged_df[merged_df.index.to_series().diff().fillna(1) > 0]

        # Calculate average sample spacing
        avg_spacing = np.diff(merged_df.index).mean()

        # Create new uniform depth array
        new_depth = np.arange(merged_df.index.min(), merged_df.index.max(), avg_spacing)

        # Interpolate data onto new depth array
        resampled_df = pd.DataFrame(index=new_depth, columns=merged_df.columns)
        for column in merged_df.columns:
            f = interpolate.interp1d(merged_df.index, merged_df[column], kind='linear', bounds_error=False, fill_value='extrapolate')
            resampled_df[column] = f(new_depth)

        self.merged_df = resampled_df

        # Merge units
        self.merged_units = self.log1_units.copy() if hasattr(self, 'log1_units') else {}
        if hasattr(self, 'log2_units'):
            for curve, unit in self.log2_units.items():
                if curve not in self.merged_units:
                    self.merged_units[curve] = unit

        if self.interpolate_top is not None and self.interpolate_bottom is not None:
            self.apply_decimation_and_interpolation()

        return self.merged_df

    def apply_decimation_and_interpolation(self):
        if self.merged_df is None:
            raise ValueError("Please merge logs before applying decimation and interpolation")

        # Create a mask for the interpolation range
        mask = (self.merged_df.index >= self.interpolate_top) & (self.merged_df.index <= self.interpolate_bottom)
        
        # Replace values in the interpolation range with NaN
        self.merged_df.loc[mask] = np.nan
        
        # Interpolate the NaN values
        self.merged_df = self.merged_df.interpolate(method='linear')

    def save_merged_las(self, save_path):
        if self.merged_df is None:
            raise ValueError("Please merge the logs before saving")

        # Use header from log2
        header = self.log2_header if hasattr(self, 'log2_header') else pd.DataFrame()

        # Ensure WELL section exists in header
        if 'WELL' not in header.index:
            header.loc['WELL', 'UWI'] = 'Unknown'

        # Update the WELL section with the new TD (total depth)
        if 'WELL' in header.index:
            header.loc['WELL', 'STOP'] = self.merged_df.index[-1]

        datasets_to_las(save_path, {'Header': header, 'Curves': self.merged_df}, self.merged_units)
        return save_path