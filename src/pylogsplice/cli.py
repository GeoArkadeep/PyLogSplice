# cli.py

import argparse
import json
from app import LogPlotterApp

def splice_logs_cli(log1_path, log2_path, output_path, aliases_path=None, interpolate_top=None, interpolate_bottom=None):
    # Create an instance of LogPlotterApp
    app = LogPlotterApp('LogPlotter', 'in.rocklab.logplotter')
    
    # Load aliases if provided
    if aliases_path:
        with open(aliases_path, 'r') as f:
            aliases = json.load(f)
        app.aliases_input.value = json.dumps(aliases, indent=4)
    
    # Load the log files
    app.process_file(log1_path, log1_path.split('.')[-1], 1)
    app.process_file(log2_path, log2_path.split('.')[-1], 2)
    
    # Set interpolation values if provided
    if interpolate_top is not None and interpolate_bottom is not None:
        app.interpolate_top_input.value = str(interpolate_top)
        app.interpolate_bottom_input.value = str(interpolate_bottom)
        app.decimate_and_interpolate(None)  # None as we're not using the widget parameter
    
    # Merge the logs
    app.merge_and_plot(None)  # None as we're not using the widget parameter
    
    # Save the merged log
    app.save_merged_las(output_path)

def main():
    parser = argparse.ArgumentParser(description="Splice two well logs.")
    parser.add_argument("log1", help="Path to the first log file (.las or .dlis)")
    parser.add_argument("log2", help="Path to the second log file (.las or .dlis)")
    parser.add_argument("-o", "--output", help="Output path for the spliced log file", default="spliced_log.las")
    parser.add_argument("--interpolate-top", type=float, help="Top depth for interpolation")
    parser.add_argument("--interpolate-bottom", type=float, help="Bottom depth for interpolation")
    parser.add_argument("--aliases", help="Path to JSON file containing aliases")
    
    args = parser.parse_args()
    
    splice_logs_cli(args.log1, args.log2, args.output, args.aliases, args.interpolate_top, args.interpolate_bottom)

if __name__ == "__main__":
    main()