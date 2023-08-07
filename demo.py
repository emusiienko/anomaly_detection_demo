import sys
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt
import argparse
from matplotlib.collections import LineCollection

arg_parser = argparse.ArgumentParser(prog="Anomaly detection demo")
arg_parser.add_argument("-f", "--file", help="csv file with data samples")
args = arg_parser.parse_args()

# Read data from csv file.  Assumes that data is in two columns: datetime and pressure.
# Datetime values are assumed to be in the form: '%m/%d/%Y %H:%M'
# Returns two lists:  first list is datetimes, second list is pressures
def generate_data(data_file_path):
    plunger_data = pd.read_csv(data_file_path)
    times = plunger_data.iloc[:, 0].tolist()
    pressures = plunger_data.iloc[:, 1].tolist()
    return times, pressures


# Calculate the per cent difference between each data value and its corresponding
# value as predicted by the regression line given by reg_slope and reg_intersept
# Returns values as a numpy array
def calculate_percent_diffs(x_data_vals, y_data_vals, reg_slope, reg_intersept):
    y_errors = []
    for point in zip(x_data_vals, y_data_vals):
        reg_y = reg_slope * point[0] + reg_intersept
        basis = point[1] if point[1] != 0 else reg_y
        if basis == 0:
            y_errors.append(0)
        else:
            y_errors.append(100 * (point[1] - reg_y) / basis)
    return np.array(y_errors)


# Calculate the per cent difference between one data value and its corresponding
# value as predicted by the regression line given by reg_slope and reg_intersept
def calculate_percent_diff(x_data_val, y_data_val, reg_slope, reg_intersept):
    reg_y = reg_slope * x_data_val + reg_intersept
    basis = reg_y if reg_y != 0 else y_data_val
    if basis == 0:
        return 0
    return 100 * (y_data_val - reg_y) / reg_y


# create main function in which we set the anomaly standard deviation, plot window size, and a simulated pause to
# slow down the plotting of the anomaly.
def main(csv_data_file_path):
    pl.rcParams['path.simplify'] = True
    pl.rcParams['path.simplify_threshold'] = 1
    # Define anomaly as greater than anomalyDtdDevFactor * StandardDev
    anomaly_std_dev_factor = 4

    # Define simulation pause in seconds
    sim_pause = 0.00001
    # Colors of normal and anomaly deviation
    plot_green_color = (0.0, 0.8, 0.0, 1.0)
    plot_red_color = (0.8, 0.0, 0.0, 1.0)
    # Get data as two numpy arrays: x_vals and y_vals
    x_str_dates, y_vals = generate_data(csv_data_file_path)

    max_y_val = max(y_vals)
    max_y_val += abs(max_y_val * 0.1)
    min_y_val = min(y_vals)
    min_y_val -= abs(max_y_val * 0.1)

    # Convert dates as strings to dates as datetime.datetime values
    dates_list = [dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in x_str_dates]
    # Convert list of datetime.datetime values to ndarray of
    x_vals = mdates.date2num(dates_list)
    # Define datetime window size
    datetime_window_size = x_vals[-1] - x_vals[0]

    # Simulate real time plotting by adding one point at a time to the current
    # data set and plot the data up to that point
    realtime_x = []
    realtime_y = []
    x_old = 0
    y_percent_diff_old = 0
    plot_color = plot_green_color

    # The Graph axis ax1 plots pressure values
    # The Graph axis ax2 plots pressure deviation as a per cent
    fig, ax1 = pl.subplots()
    ax2 = ax1.twinx()
    fig.set_size_inches(15, 5)
    ax1.set_ylim(min_y_val, max_y_val)
    ax1.set_xlabel("Time Units")
    ax1.set_ylabel("Pressure")

    ax2.set_ylabel(" % Deviation From Regression Line", color="g")
    ax2.set_ylim(-120, 120)

    pl.xlim(x_vals[0], x_vals[0] + datetime_window_size)

    ax2.plot(
        [x_vals[0], x_vals[len(x_vals) - 1]],
        [0, 0],
        color="tab:blue",
        linewidth=0.5
    )

    pl.show(block=False)
    pl.pause(sim_pause)
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    # Setup animating data graphs
    realtime_data_path_collection = ax1.scatter([], [], color="black", animated=True)
    percent_diff_line_collection = LineCollection([], linewidth=0.5,
                                                  colors=[], linestyle='solid', animated=True)
    regression_line_collection = LineCollection([], linewidth=1,
                                                colors="blue", linestyle='solid', animated=True)

    ax2.add_collection(percent_diff_line_collection)
    ax1.add_collection(regression_line_collection)

    fig.canvas.blit(fig.bbox)

    # Iterate through all data, adding one point per iteration.  The lists, realtime_x and realtime_y start as 
    # empty and one data point is added during each iteration. 
    # As each point is added, all points that have been added are displayed on graph.
    # This simulates real time data being added to the current graph.
    for x, y in zip(x_vals, y_vals):
        fig.canvas.restore_region(bg, fig.bbox)

        # Plot pressure data
        realtime_data_path_collection.set_offsets(np.c_[realtime_x, realtime_y])
        ax1.draw_artist(realtime_data_path_collection)

        regress_plot_size = 30
        # These two variables hold the learning set data
        realtime_x.append(x)
        realtime_y.append(y)
        x_new = x
        # Wait for 6 points to be collected before calculating stats
        if len(realtime_x) > 10:
            # Do not calculate stats on present point.  It may be an anomaly.
            calc_x = realtime_x[:-1]
            calc_y = realtime_y[:-1]

            fit = np.polyfit(np.array(calc_x), np.array(calc_y), deg=1)

            # Calculate std and mean
            y_percent_diffs = calculate_percent_diffs(calc_x, calc_y, fit[0], fit[1])
            std = np.std(y_percent_diffs)
            mean = np.mean(y_percent_diffs)

            # Calculate percent diff for current point for plotting
            y_percent_diff = calculate_percent_diff(x, y, fit[0], fit[1])

            # If the y_percent_diff is outside anomaly_std_dev_factor*std range, remove it from the realTime
            # data so that the anomaly will not be included in the regression line
            # and not included in the calculation for std and mean.
            if (y_percent_diff < mean - anomaly_std_dev_factor * std) or (
                    y_percent_diff > mean + anomaly_std_dev_factor * std
            ):
                plot_color = plot_red_color
                realtime_x.pop()
                realtime_y.pop()

            y_percent_diff_new = y_percent_diff

            if x_old > 0:
                colors = np.append(percent_diff_line_collection.get_colors(), [plot_color], axis=0)
                percent_diff_line_collection.set_colors(colors)

                segments = percent_diff_line_collection.get_segments()
                segments.append(np.array([(x_old, y_percent_diff_old), (x_new, y_percent_diff_new)]))
                percent_diff_line_collection.set_segments(segments)
                ax2.draw_artist(percent_diff_line_collection)

                X_as_numpy = np.array(calc_x)
                X_len = len(X_as_numpy)
                if X_len > regress_plot_size:
                    regress_start = X_len - regress_plot_size
                    segments = regression_line_collection.get_segments()
                    points = list(zip(calc_x[regress_start: X_len - 1],
                                      (fit[0] * np.array(calc_x[regress_start: X_len - 1]) + fit[1])))
                    segments.append(np.array(points))
                    regression_line_collection.set_segments(segments)
                    ax1.draw_artist(regression_line_collection)

            x_old = x_new
            y_percent_diff_old = y_percent_diff_new
            plot_color = plot_green_color

        fig.canvas.blit(fig.bbox)
        pl.pause(sim_pause)
        bg = fig.canvas.copy_from_bbox(fig.bbox)

    pl.savefig("anomaly.png")


# Run main function which plots the points in black. The linear regression is identified by the blue line.  
# The Deviation from the regression line is colored green.  
# If the percent difference is outside the anomaly standard deviation, we have an anomaly detected.
# Color the line red to denote a detected anomaly.  The plot will also be saved in the notebooks folder as anomaly2.png

if __name__ == "__main__":
    args = arg_parser.parse_args(sys.argv[1:])
    csv_data_file_path = args.file or "src/static/data/casing.csv"
    main(csv_data_file_path=csv_data_file_path)
