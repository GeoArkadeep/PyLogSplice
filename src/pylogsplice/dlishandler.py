import pandas as pd
import dlisio
dlisio.common.set_encodings(['utf-8','latin-1'])
from dlisio import dlis
import numpy as np

def get_dlis_data_all(path):
    # Load the DLIS file
    f, *tail = dlis.load(path)
    # Get the list of frames
    frames = f.frames
    
    chnames = []
    description = []
    for ch in f.channels:
        chnames.append(ch.name)
        description.append(ch.long_name)
    return pd.DataFrame(list(zip(chnames,description)),columns =['Mnemonic', 'Description'])
    """
    # Create 2D lists of Channel names and descriptions for each frame
    channel_names, channel_descriptions = zip(*[
        ([channel.name for channel in frame.channels],
         [channel.long_name for channel in frame.channels])
        for frame in frames
    ])

    # Convert the results back to lists
    channel_names = list(channel_names)
    channel_descriptions = list(channel_descriptions)
    """
def get_dlis_data(path, aliases, depthunits='m'):

    # Load the DLIS file
    f, *tail = dlis.load(path)
    origin, *origin_tail = f.origins
    # Initialize a dictionary to store DataFrames for each frame
    dataframes = {}

    # Initialize a dictionary to store units for each extracted curve
    c_units = {}

    # Conversion factors
    conversion_factors = {
        'ft': 0.3048,
        'f': 0.3048,
        'feet': 0.3048,
        '0.1in': 0.00254,
        'inches': 0.0254,
        'in': 0.0254
    }

    # Extract parameters
    parameters = {}
    long_names = {}
    p_units = {}
    p_section = {}
    for param in f.parameters:
        if(len(param.values)==1):
            if str(param.values[0])!="-999.25" and str(param.values[0])!="" and str(param.values[0])!="0" and len(str(param.values[0]))<20:
                parameters[param.name] = param.values[0]
                long_names[param.name] = param.long_name
                #description = param.describe()
                p_units[param.name] = param.attic['VALUES'].units
                p_section[param.name] = "Parameter"

    # Create the DataFrame with the desired column names and order
    header_df = pd.DataFrame([
        {
            'original_mnemonic': param,
            'mnemonic': param,
            'unit': p_units.get(param, ''),
            'value': value,
            'descr': long_names.get(param, ''),
            'section': p_section.get(param, 'Parameter')
        }
        for param, value in parameters.items()
    ])

    # Ensure the correct column order
    header_df = header_df[['original_mnemonic', 'mnemonic', 'unit', 'value', 'descr', 'section']]
    
    # Initialize a set to store all encountered curve names
    all_curves = set()

    # Iterate over all frames in the DLIS file
    for frame in f.frames:
        curves = frame.curves(strict=False)
        
        # Add all curve names to the set
        all_curves.update(curves.dtype.names)
        
        # Initialize a dictionary to store the extracted curves
        extracted_curves = {}
        
        # Set the index channel
        index_channel = frame.index
        
        if index_channel:
            tdep_channel = next((ch for ch in frame.channels if ch.name == index_channel), None)
            if tdep_channel:
                # Extract unit of TDEP
                tdep_unit = tdep_channel.units.lower()
                conversion_factor = conversion_factors.get(tdep_unit, 1)  # Default to 1 if no conversion is needed
                extracted_curves[index_channel] = curves[index_channel] * conversion_factor
                c_units[index_channel] = tdep_unit

        # Extract matching curves based on aliases and store them with original mnemonics
        found_useful_data = False
        for alias, mnemonics in aliases.items():
            for mnemonic in mnemonics:
                if mnemonic in curves.dtype.names:
                    extracted_curves[mnemonic] = curves[mnemonic]
                    found_useful_data = True
                    # Extract the unit for the curve
                    curve_channel = next((ch for ch in frame.channels if ch.name == mnemonic), None)
                    if curve_channel:
                        c_units[mnemonic] = curve_channel.units
                    break

        # Only create DataFrame and store if useful data is found
        if found_useful_data:
            frame_df = pd.DataFrame(extracted_curves)
            
            # Set the index if the index channel is available
            if index_channel and index_channel in frame_df.columns:
                frame_df.set_index(index_channel, inplace=True)
            if not frame_df.empty:
                dataframes[frame.name] = frame_df

    # Print all encountered curves
    print("All encountered curves:")
    print(", ".join(sorted(all_curves)))

    # Combine non-empty DataFrames based on their index values, keeping all values and padding with NaN when necessary
    combined_df = pd.concat(dataframes.values(), axis=1, join='inner')
    combined_df.replace(-999.25, float("nan"), inplace=True)
    if depthunits == 'm':
        combined_df.index = combined_df.index * 0.00245
    elif depthunits == 'f':
        combined_df.index = combined_df.index * 0.00833
    combined_df.sort_index(inplace=True)
    c_units[index_channel] = depthunits
    #Auxiliary Header Items
    servicecom = origin.producer_name
    clientname = origin.company
    welluwi = origin.well_id
    wellname = origin.well_name
    start_depth = combined_df.index.min()
    stop_depth = combined_df.index.max()
    step = np.mean(np.diff(combined_df.index))
    fieldname = origin.field_name
    nully  = -999.25
    
    # Create a dictionary for auxiliary header items
    aux_header = {
        'STRT': start_depth,
        'STOP': stop_depth,
        'STEP': step,
        'NULL': nully,
        'UWI': welluwi,
        'WELL': wellname,
        'SRVC': servicecom,
        'COMP': clientname,
        'FLD': fieldname
    }
    print(start_depth,stop_depth,step)
    # Define units for specific items
    aux_units = {
        'STRT': 'M',
        'STOP': 'M',
        'STEP': 'M',
        'NULL': '',
        'UWI': '',
        'WELL': '',
        'SRVC': '',
        'COMP': '',
        'FLD': ''
    }

    # Create a DataFrame for auxiliary header items
    aux_df = pd.DataFrame([
        {
            'original_mnemonic': k,
            'mnemonic': k,
            'unit': aux_units[k],
            'value': v,
            'descr': '',
            'section': 'Well'
        }
        for k, v in aux_header.items()
    ])

    # Concatenate auxiliary header with the existing header
    header_df = pd.concat([aux_df, header_df], ignore_index=True)
    
    # Copy the index to the first column while preserving the original index
    index_name = combined_df.index.name or 'DEPTH'
    combined_df = combined_df.copy()  # Create a copy to avoid modifying the original
    combined_df[index_name] = combined_df.index  # Add index as a new column

    # Ensure the depth column is the first column
    columns = combined_df.columns.tolist()
    columns.insert(0, columns.pop(columns.index(index_name)))
    combined_df = combined_df[columns]

    # Update the c_units dictionary to include the index column
    c_units[index_name] = depthunits

    # Set the index name explicitly (in case it was None)
    combined_df.index.name = index_name

    return combined_df, c_units, header_df, parameters


def datasets_to_las(path, datasets, custom_units={}, **kwargs):
    """
    Write datasets to a LAS file on disk.

    Args:
        path (Str): Path to write LAS file to
        datasets (Dict['<name>': pd.DataFrame]): Dictionary maps a
            dataset name (e.g. 'Curves') or 'Header' to a pd.DataFrame.
        curve_units (Dict[str, str], optional): Dictionary mapping curve names to their units.
            If a curve's unit is not specified, it defaults to an empty string.
    Returns:
        Nothing, only writes in-memory object to disk as .las
    """
    from functools import reduce
    import warnings
    from datetime import datetime
    from io import StringIO
    from urllib import error, request

    import lasio as laua
    import numpy as np
    import pandas as pd
    from lasio import HeaderItem, CurveItem, SectionItems
    from pandas._config.config import OptionError

    from welly.curve import Curve
    from welly import utils
    from welly.fields import curve_sections, other_sections, header_sections
    from welly.utils import get_columns_decimal_formatter, get_step_from_array
    from welly.fields import las_fields as LAS_FIELDS
    # ensure path is working on every dev set-up
    path = utils.to_filename(path)

    # instantiate new LASFile to parse data & header to
    las = laua.LASFile()

    # set header df as variable to later retrieve curve meta data from
    header = datasets['Header']
    
    extracted_units = {}
    if not header.empty:
        curve_header = header[header['section'] == 'Curves']
        for _, row in curve_header.iterrows():
            if row['unit']:  # Ensure there is a unit specified
                extracted_units[row['original_mnemonic']] = row['unit']

    # Combine extracted units with custom units, custom units take precedence
    all_units = {**extracted_units, **custom_units}
    
    column_fmt = {}
    for curve in las.curves:
        column_fmt[curve.mnemonic] = "%10.5f"
    
    # unpack datasets
    for dataset_name, df in datasets.items():

        # dataset is the header
        if dataset_name == 'Header':
            # parse header pd.DataFrame to LASFile
            for section_name in set(df.section.values):
                # get header section df
                df_section = df[df.section == section_name]

                if section_name == 'Curves':
                    # curves header items are handled in curve data loop
                    pass

                elif section_name == 'Version':
                    if len(df_section[df_section.original_mnemonic == 'VERS']) > 0:
                        las.version.VERS = df_section[df_section.original_mnemonic == 'VERS']['value'].values[0]
                    if len(df_section[df_section.original_mnemonic == 'WRAP']) > 0:
                        las.version.WRAP = df_section[df_section.original_mnemonic == 'WRAP']['value'].values[0]
                    if len(df_section[df_section.original_mnemonic == 'DLM']) > 0:
                        las.version.DLM = df_section[df_section.original_mnemonic == 'DLM']['value'].values[0]

                elif section_name == 'Well':
                    las.sections["Well"] = SectionItems(
                        [HeaderItem(r.original_mnemonic,
                                    r.unit,
                                    r.value,
                                    r.descr) for i, r in df_section.iterrows()])

                elif section_name == 'Parameter':
                    las.sections["Parameter"] = SectionItems(
                        [HeaderItem(r.original_mnemonic,
                                    r.unit,
                                    r.value,
                                    r.descr) for i, r in df_section.iterrows()])

                elif section_name == 'Other':
                    las.sections["Other"] = df_section['descr'].iloc[0]

                else:
                    m = f"LAS Section was not recognized: '{section_name}'"
                    warnings.warn(m, stacklevel=2)

        # dataset contains curve data
        if dataset_name in curve_sections:
            header_curves = header[header.section == dataset_name]
            for column_name in df.columns:
                curve_data = df[column_name]
                curve_unit = all_units.get(column_name, '')  # Use combined units
                las.append_curve(mnemonic=column_name,
                                 data=curve_data,
                                 unit=curve_unit,
                                 descr='',
                                 value='')


    # numeric null value representation from the header (e.g. # -9999)
    try:
        null_value = header[header.original_mnemonic == 'NULL'].value.iloc[0]
    except IndexError:
        null_value = -999.25
    las.null_value = null_value

    # las.write defaults to %.5 decimal points. We want to retain the
    # number of decimals. We first construct a column formatter based
    # on the max number of decimal points found in each curve.
    if 'column_fmt' not in kwargs:
        kwargs['column_fmt'] = column_fmt

    # write file to disk
    with open(path, mode='w') as f:
        las.write(f,**kwargs)

#example usage
"""
# Display the combined DataFrame
alias = {
    "sonic": ["none", "DTC", "DT24", "DTCO", "DT", "AC", "AAC", "DTHM"],
    "ssonic": ["none", "DTSM"],
    "gr": ["none", "GR", "GRD", "CGR", "GRR", "GRCFM"],
    "resdeep": ["none", "HDRS", "LLD", "M2RX", "MLR4C", "RD", "RT90", "RLA1", "RDEP", "RLLD", "RILD", "ILD", "RT_HRLT", "RACELM"],
    "resshal": ["none", "LLS", "HMRS", "M2R1", "RS", "RFOC", "ILM", "RSFL", "RMED", "RACEHM", "RXO_HRLT"],
    "density": ["none", "ZDEN", "RHOB", "RHOZ", "RHO", "DEN", "RHO8", "BDCFM"],
    "neutron": ["none", "CNCF", "NPHI", "NEU", "TNPH", "NPHI_LIM"],
    "pe": ["none", "PEFLA", "PEF8", "PE"]
}


import time
start_time = time.time()

df,units, header, pdict = get_dlis_data('dummy.dlis',alias)
df.index.name = 'DEPT'
    
datasets_to_las("converted.las", {'Header': header,'Curves':df}, units)
print(units)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")
#header['Values'] = header['Values'].apply(remove_brackets)
print(header)
print(units)
# Save the header DataFrame to a CSV file
header.to_csv('header.csv', index=False)
print("Header DataFrame saved to header.csv")

print("Combined DataFrame:")
print(df)
# Plotting DTCO and DTSM over TDEP
df2 = df.copy()
import numpy as np
for i in range(len(df['DTCO'])):
    if df['DTCO'].iloc[i]<75 or df['DTCO'].iloc[i]>100:
        df['DTCO'].iloc[i] = np.nan


from scipy.stats import linregress
mask = ~np.isnan(df['DTCO']) & ~np.isnan(df['DTSM'])
x = df['DTCO'][mask]
y = df['DTSM'][mask]
slope, intercept, r_value, p_value, std_err = linregress(x, y)
print(slope,intercept)
# Calculate linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 60))

plt.plot(df.index, df2['DTCO'], label='DTCO', color='b')
plt.plot(df.index, df2['DTSM'], label='DTSM', color='r')
plt.plot(df.index, df2['DTSM'] - (slope*df2['DTCO'] + intercept), label=f'Linear Fit: DTSM = {slope:.2f}*DTCO + {intercept:.2f}', color='g', linestyle='-')
plt.xlabel('TDEP (scaled)')
plt.ylabel('Value')
plt.title('DTCO and DTSM over TDEP')
plt.legend()
plt.grid(True)

end_time2 = time.time()
elapsed_time2 = end_time2 - start_time
print(f"Time taken: {elapsed_time2:.2f} seconds")

plt.show()
"""