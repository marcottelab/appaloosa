#!/usr/bin/env python2


"""
Thin layer chromatography spot segmentation & quantification. 
"""


# Need matplotlib for saving image
import matplotlib
import matplotlib.pyplot as plt

# Import other Python libraries we use
import argparse
from collections import defaultdict
from sys import stdout
from glob import glob
from datetime import datetime
import time
import csv
import multiprocessing
#from imageio import imread  # This causes come problems; using PIL instead
import PIL
import numpy as np
from scipy.spatial.distance import euclidean
from skimage import dtype_limits
from skimage.feature import (peak_local_max,
                             blob_log,
                            )
from skimage.color import (rgb2gray,
                           label2rgb,
                          )
from skimage.measure import label
from skimage.morphology import watershed
from skimage.io import imsave
from skimage.segmentation import find_boundaries
from skimage.util import invert

#We use Tkinter for GUI
import Tkinter as tk
from PIL import Image, ImageTk

# Import image analysis library
import appaloosa


# Define and parse arguments; use custom MyFormatter to do both ArgumentDefault
# and RawDescription Formatters via multiple inheritence, this is a trick to
# preserve docstring formatting in --help output
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter,
                 ):
    pass
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=MyFormatter,
                                )
image_filename_helpstring = "Image of TLC plate"
parser.add_argument('image_filename',
                    help=image_filename_helpstring,
                   )
intermediate_images_helpstring = ("Output intermediate image steps to PNGs. "
                                  "Useful for understanding what's happening."
                                 )
intermediate_images_figsize = 10
parser.add_argument('--intermediate_images',
                    action='store_true',
                    default=False,
                    help=intermediate_images_helpstring,
                   )
zoom_helpstring = ("Image display zoom. "
                   "This determines how large the image and window are."
                  )
parser.add_argument('--zoom',
                    type=float,
                    default=3,
                    help=zoom_helpstring,
                   )
args = parser.parse_args()

# Load plate image
image = np.array(PIL.Image.open(args.image_filename))
plate = appaloosa.Plate(image=image,
                        #image=imread(args.image_filename),
                        tag_in='original_image',
                        source_filename=args.image_filename,
                       )
if args.intermediate_images:
    plate.display(tag_in='original_image',
                  figsize=intermediate_images_figsize,
                  output_filename="original_image.png",
                 )

# Segment the plates from the background
plate.crop_to_plate(tag_in='original_image',
                    tag_out='cropped_image',
                    feature_out='crop_rotation',
                    second_pass=False,
                   )
if args.intermediate_images:
    plate.display(tag_in='cropped_image',
                  figsize=intermediate_images_figsize,
                  output_filename="cropped_image.png",
                 )

# Trim the outermost pixels a bit to make sure no background remains around the
# edges
cropped_image = plate.image_stash['cropped_image']
cropped_image_height, cropped_image_width = cropped_image.shape[:2]
cropped_image_min_dimension = min(cropped_image_height, cropped_image_width)
percent_crop = 0.03
border = int(round(cropped_image_min_dimension * percent_crop))
plate.crop_border(tag_in='cropped_image',
                  tag_out='border_cropped_image',
                  border=border,
                 )
if args.intermediate_images:
    plate.display(tag_in='border_cropped_image',
                  figsize=intermediate_images_figsize,
                  output_filename="border_cropped_image.png",
                 )

# Rescale image to standard size
# This is very important because the image morphology parameters we use for analysis are defined
# in terms of pixels and therefore are specific to a (ballpark) resolution.
cropped_image = plate.image_stash['border_cropped_image']
cropped_height, cropped_width = cropped_image.shape[:2]
largest_dimension = max(cropped_height, cropped_width)
target_scale = 500
scaling_factor = float(target_scale) / largest_dimension
plate.rescale_image(tag_in='border_cropped_image',
                    tag_out='rescaled_image',
                    scaling_factor=scaling_factor,
                   )
if args.intermediate_images:
    plate.display(tag_in='rescaled_image',
                  figsize=intermediate_images_figsize,
                  output_filename="rescaled_image.png",
                 )


# Median correct the image to correct uneven intensity over the plate
uncorrected_image = plate.image_stash['rescaled_image']
corrected_image = appaloosa.Plate.median_correct_image(image=uncorrected_image,
                                                       median_disk_radius=31,
                                                      )
plate.image_stash['corrected_rescaled_image'] = corrected_image
if args.intermediate_images:
    plate.display(tag_in='corrected_rescaled_image',
                  figsize=intermediate_images_figsize,
                  output_filename="corrected_rescaled_image.png",
                 )

# Let's try segmenting the spots using the waterfall algorithm
plate.waterfall_segmentation(tag_in='corrected_rescaled_image',
                             feature_out='waterfall_basins',
                             R_out='R_img',
                             mg_out='mg_img',
                             median_disk_radius=31,
                             smoothing_sigma=2,
                             threshold_opening_size=2,
                             basin_open_close_size=5,
                             skeleton_label=0,
                             debug_output=False,
                            )
if args.intermediate_images:
    plate.display(tag_in='corrected_rescaled_image',
                  basins_feature='waterfall_basins',
                  figsize=intermediate_images_figsize,
                  output_filename="waterfall_basins.png",
                 )

# The largest item found is the background; we need to get rid of it
plate.remove_most_frequent_label(basins_feature='waterfall_basins',
                                 feature_out='filtered_waterfall_basins',
                                 debug_output=False,
                                )

# Overlay finegrained watershed over waterfall segmentation
plate.overlay_watershed(tag_in='corrected_rescaled_image',
                        intensity_image_tag='corrected_rescaled_image',
                        median_radius=None,
                        filter_basins=True,
                        waterfall_basins_feature='filtered_waterfall_basins',
                        feature_out='overlaid_watershed_basins',
                        min_localmax_dist=5,
                        smoothing_sigma=1,
                        min_area=10,
                        min_intensity=0.1,
                        rp_radius_factor=None,
                        debug_output=False,
                        basin_open_close_size=None,
                       )
if args.intermediate_images:
    plate.display(tag_in='corrected_rescaled_image',
                  basins_feature='overlaid_watershed_basins',
                  figsize=intermediate_images_figsize,
                  output_filename='overlaid_watershed_basins.png',
                 )

# Measure basins
plate.measure_basin_intensities(tag_in='corrected_rescaled_image',
                                median_radius=None,
                                filter_basins=True,
                                radius_factor=None,
                                basins_feature='overlaid_watershed_basins',
                                feature_out='basin_intensities',
                                multiplier=10.0,
                               )
plate.find_basin_centroids(tag_in='corrected_rescaled_image',
                           basins_feature='overlaid_watershed_basins',
                           feature_out='basin_centroids',
                          )

# Each spot is given a unique integer identifier
#Its intensity is shown as I= <- this is currently omitted
if args.intermediate_images:
    plate.display(tag_in='rescaled_image',
                  figsize=70,
                  basins_feature='overlaid_watershed_basins',
                  basin_alpha=0.1,
                  baseline_feature=None,
                  solvent_front_feature=None,
                  lanes_feature=None,
                  basin_centroids_feature='basin_centroids',
                  basin_lane_assignments_feature=None,
                  #basin_intensities_feature='basin_intensities',
                  basin_rfs_feature=None,
                  lines_feature=None,
                  draw_boundaries=True,
                  side_by_side=False,
                  display_labels=True,
                  output_filename="initial_output.png",
                 )

# Display basins in GUI and begin interactive segmentation
plate.feature_stash['iterated_basins'] = \
                        plate.feature_stash['overlaid_watershed_basins'].copy()

resize_ratio = args.zoom

def make_pil_image(color_image,
                   basins,
                   resize_ratio=3,
                   background_grid=5,
                   assignment_feature='base_assignments',
                  ):
    if background_grid is not None:
        gridded_image = color_image.copy()
        for (h, w), basin in np.ndenumerate(basins):
            if basin != 0:
                continue
            if h % background_grid != 0:
                continue
            if w % background_grid != 0:
                continue
            gridded_image[h, w] = 100
        color_image = gridded_image
    basin_boundaries = find_boundaries(basins,
                                       mode='inner',
                                      )
    h, w, num_channels = color_image.shape
    if num_channels == 4:
        # If alpha channel is not treated separately, then alpha for boundaries
        # becomes 1, so the boundaries appear white
        boundaries_list = [basin_boundaries] * (num_channels - 1)
        boundaries_list.append(np.zeros_like(basin_boundaries).astype(np.bool))
    else:
        boundaries_list = [basin_boundaries] * num_channels
    stacked_boundaries = np.stack(boundaries_list,
                                  axis=-1,
                                 )
    segmented_image = color_image * ~stacked_boundaries
    global plate, baseline_colors, canvas
    if (assignment_feature is not None
        and assignment_feature in plate.feature_stash
       ):
        assignments = plate.feature_stash[assignment_feature]
        for basin, base_assign_state in assignments.iteritems():
            isolated_basin = np.where(basins == basin,
                                      1,
                                      0,
                                     )
            isolated_boundary = find_boundaries(isolated_basin,
                                                mode='inner',
                                               )
            if num_channels == 4:
                boundaries_list = [isolated_boundary] * (num_channels - 1)
                boundaries_list.append(
                               np.zeros_like(isolated_boundary).astype(np.bool)
                                      )
            else:
                boundaries_list = [isolated_boundary] * num_channels
            stacked_boundaries = np.stack(boundaries_list,
                                          axis=-1,
                                         )
            baseline_color = baseline_colors[base_assign_state - 1]
            baseline_rgb = canvas.winfo_rgb(baseline_color) # 16-bit
            imin, imax = dtype_limits(color_image, clip_negative=False)
            baseline_r = float(baseline_rgb[0]) * imax / 65535
            baseline_g = float(baseline_rgb[1]) * imax / 65535
            baseline_b = float(baseline_rgb[2]) * imax / 65535
            for (h, w), boundary in np.ndenumerate(isolated_boundary):
                if boundary:
                    segmented_image[h, w, 0] = baseline_r
                    segmented_image[h, w, 1] = baseline_g
                    segmented_image[h, w, 2] = baseline_b
    uint8_image = np.rint(segmented_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(uint8_image)
    #display_size = 1000
    image_width, image_height = pil_image.size
    #resize_ratio = min(float(display_size) / image_width,
    #                   float(display_size) / image_height,
    #                  )
    resized_width = int(round(image_width * resize_ratio))
    resized_height = int(round(image_height * resize_ratio))
    pil_image = pil_image.resize((resized_width, resized_height))
    return pil_image

#meshgrid would be faster, but I this is easier to understand
background_grid = np.zeros_like(plate.feature_stash['iterated_basins'])
grid_spacing = 3
for h, w in np.ndindex(background_grid.shape):
    if h % grid_spacing == 0 and w % grid_spacing == 0:
        background_grid[h, w] = 1
background_ovals = []

def grid_background(canvas,
                    basins,
                   ):
    global background_grid
    this_grid = np.where(basins == 0,
                         background_grid,
                         0,
                        )
    global background_ovals
    background_ovals = []
    oval_radius = 3
    global resize_ratio
    for (h, w), g in np.ndenumerate(this_grid):
        if g == 1:
            oval = canvas.create_oval(w * resize_ratio - oval_radius,
                                      h * resize_ratio - oval_radius,
                                      w * resize_ratio + oval_radius,
                                      h * resize_ratio + oval_radius,
                                      width=0,
                                      fill='gray',
                                     )
            background_ovals.append(oval)

root = tk.Tk()
color_image = plate.image_stash['rescaled_image']
pil_image = make_pil_image(color_image=color_image,
                           basins=plate.feature_stash['iterated_basins'],
                           resize_ratio=resize_ratio,
                          )
tk_image = ImageTk.PhotoImage(master=root,
                              image=pil_image,
                             )
image_width, image_height = pil_image.size
canvas = tk.Canvas(root,
                   width=image_width,
                   height=image_height,
                  )
canvas.pack()
canvas_image = canvas.create_image(0, 0,
                                   anchor='nw',
                                   image=tk_image,
                                  )
bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM)
quit_button = tk.Button(bottom_frame,
                        text="Quit",
                        command=quit,
                       )
quit_button.grid(column=0, row=1)

def order_centroids(basin_centroids,
                    line,
                   ):
    (h1, w1), (h2, w2) = line
    projected_centroids = {basin: appaloosa.Plate.project_point_on_segment(
                                                  point=(h, w),
                                                  segment=((h1, w1), (h2, w2)),
                                                                          )
                           for basin, (h, w) in basin_centroids.iteritems()
                          }
    basin_ordering = list(enumerate(sorted(projected_centroids.items(),
                                           key=lambda x:x[1],
                                          ),
                                    start=1,
                                   )
                         )
    basin_map = {basin: index
                 for index, (basin, position) in basin_ordering
                }
    return basin_map


def on_save():
    global plate
    basin_centroids = plate.feature_stash['basin_centroids']
    solvent_front = plate.feature_stash.get('solvent_front', None)
    if solvent_front is not None:
        basin_map = order_centroids(basin_centroids=basin_centroids,
                                    line=solvent_front,
                                   )
    else:
        basin_map = {basin: basin for basin in basin_centroids.iterkeys()}
    basins = plate.feature_stash['iterated_basins']
    sorted_basins = np.zeros_like(basins)
    for basin, sorted_basin in basin_map.iteritems():
        sorted_basins = np.where(basins == basin,
                                 sorted_basin,
                                 sorted_basins,
                                )
    plate.feature_stash['sorted_basins'] = sorted_basins
    sorted_centroids = {basin_map[basin]: centroid
                        for basin, centroid in basin_centroids.iteritems()
                       }
    plate.feature_stash['sorted_centroids'] = sorted_centroids
    epoch_hash = appaloosa.epoch_to_hash(time.time())
    output_basename = epoch_hash
    print("Saving using basename " + str(output_basename))
    image_filename = output_basename + "_segmented.png"
    plate.display(tag_in='rescaled_image',
                  figsize=70,
                  #basins_feature='iterated_basins',
                  basins_feature='sorted_basins',
                  basin_alpha=0.1,
                  baseline_feature=None,
                  solvent_front_feature=None,
                  lanes_feature=None,
                  #basin_centroids_feature='basin_centroids',
                  basin_centroids_feature='sorted_centroids',
                  basin_lane_assignments_feature=None,
                  #basin_intensities_feature='basin_intensities',
                  basin_rfs_feature=None,
                  lines_feature=None,
                  draw_boundaries=True,
                  side_by_side=False,
                  display_labels=True,
                  output_filename=image_filename,
                 )
    csv_filename = output_basename + "_intensities.csv"
    basin_intensities = plate.feature_stash['basin_intensities']
    indexed_basin_rfs = plate.feature_stash.get('indexed_basin_rfs', {})
    collated_basin_rfs = {}
    for base_assign_state, rf_dict in indexed_basin_rfs.iteritems():
        for basin, rf in rf_dict.iteritems():
            assert basin not in collated_basin_rfs
            collated_basin_rfs[basin] = (base_assign_state, rf)
    with open(csv_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_header = ["Spot #", "Intensity", "Baseline #", "Rf"]
        csv_writer.writerow(csv_header)
        for basin, sorted_basin in sorted(basin_map.iteritems(),
                                          key=lambda x:x[1],
                                         ):
        #for basin, intensity in sorted(basin_intensities.iteritems(),
        #                               key=lambda x:x[0],
        #                              ):
            intensity = basin_intensities[basin]
            base_assign_state, rf = collated_basin_rfs.get(basin, (None, None))
            basin_row = [sorted_basin, intensity, base_assign_state, rf]
            csv_writer.writerow(basin_row)
    print("Finished writing CSV.")

save_button = tk.Button(bottom_frame,
                        text="Save & Quit",
                        command=on_save,
                       )
save_button.grid(column=1, row=1)

def alive():
    print("Alive!")

alive_button = tk.Button(bottom_frame,
                         text="Alive?",
                         command=alive,
                        )
alive_button.grid(column=8, row=1)

maxima_distance_label = tk.Label(bottom_frame,
                                 text="Subdivision resolution",
                                )
maxima_distance_label.grid(column=0, row=2)
maxima_distance_entry = tk.Entry(bottom_frame)
maxima_distance_entry.insert(0, 2)
maxima_distance_entry.grid(column=1, row=2)

def remeasure_basins(plate):
    plate.measure_basin_intensities(tag_in='corrected_rescaled_image',
                                    median_radius=None,
                                    filter_basins=True,
                                    radius_factor=None,
                                    basins_feature='iterated_basins',
                                    feature_out='basin_intensities',
                                    multiplier=10.0,
                                   )
    plate.find_basin_centroids(tag_in='corrected_rescaled_image',
                               basins_feature='iterated_basins',
                               feature_out='basin_centroids',
                              )

left_click_buffer = []
left_click_buffer_size = 2
left_click_shapes = []

def right_click(event):
    stdout.write("Deleting spot...")
    stdout.flush()
    w, h = event.x, event.y
    global plate, canvas, canvas_image, tk_image, pil_image, resize_ratio
    basins = plate.feature_stash['iterated_basins']
    mapped_w = int(round(float(w) / resize_ratio))
    mapped_h = int(round(float(h) / resize_ratio))
    basin = basins[mapped_h, mapped_w]
    if basin == 0:
        print("This is background; not deleting.")
        return
    basins = np.where(basins == basin,
                      0,
                      basins,
                     )
    plate.feature_stash['iterated_basins'] = basins
    remeasure_basins(plate)
    color_image = plate.image_stash['rescaled_image']
    pil_image = make_pil_image(color_image=color_image,
                               basins=plate.feature_stash['iterated_basins'],
                               resize_ratio=resize_ratio,
                              )
    tk_image = ImageTk.PhotoImage(image=pil_image)
    canvas.itemconfig(canvas_image, image=tk_image)
    #grid_background(canvas=canvas,
    #                basins=basins,
    #               )
    stdout.write("complete\n")
    stdout.flush()
    
canvas.bind('<Button 3>', right_click)

def subdivide_spot():
    stdout.write("Subdivision...")
    stdout.flush()
    global plate, resize_ratio, canvas, canvas_image, pil_image, tk_image
    global maxima_distance_entry
    maxima_distance = int(maxima_distance_entry.get())
    if len(left_click_buffer) < 1:
        print("Insufficient points defined")
        return
    w, h = left_click_buffer[-1]
    mapped_w = int(round(float(w) / resize_ratio))
    mapped_h = int(round(float(h) / resize_ratio))
    basins = plate.feature_stash['iterated_basins']
    basin = basins[mapped_h, mapped_w]  # Note h, w reversed vs event
    if basin == 0:
        print("This is background; not splitting.")
        return
    plate.subdivide_basin(tag_in='corrected_rescaled_image',
                          feature_out='iterated_basins',
                          basins_feature='iterated_basins',
                          target_basin=basin,
                          smoothing_sigma=None,
                          maxima_distance=maxima_distance,
                         )
    remeasure_basins(plate)
    color_image = plate.image_stash['rescaled_image']
    pil_image = make_pil_image(color_image=color_image,
                               basins=plate.feature_stash['iterated_basins'],
                               resize_ratio=resize_ratio,
                              )
    tk_image = ImageTk.PhotoImage(image=pil_image)
    canvas.itemconfig(canvas_image, image=tk_image)
    stdout.write("complete\n")
    stdout.flush()

linear_split_button = tk.Button(bottom_frame,
                                text="Watershed subdivide spot",
                                command=subdivide_spot,
                               )
linear_split_button.grid(column=3, row=2)

def left_click(event):
    w, h = event.x, event.y
    global left_click_buffer, left_click_buffer_size, left_click_shapes
    left_click_buffer.append((w, h))
    left_click_buffer = left_click_buffer[-left_click_buffer_size:]
    oval_radius = 3
    for w, h in left_click_buffer:
        oval = canvas.create_oval(w - oval_radius,
                                  h - oval_radius,
                                  w + oval_radius,
                                  h + oval_radius,
                                  width=0,
                                  fill='orange',
                                 )
        left_click_shapes.append(oval)
    for oval in left_click_shapes[:-left_click_buffer_size]:
        canvas.delete(oval)

canvas.bind('<Button 1>', left_click)

def linear_split():
    stdout.write("Subdivision...")
    stdout.flush()
    global left_click_buffer
    global plate, resize_ratio, canvas, canvas_image, pil_image, tk_image
    if len(left_click_buffer) < 2:
        print("Insufficient points defined")
        return
    (w1, h1), (w2, h2) = left_click_buffer[-2:]
    mapped_w1 = int(round(float(w1) / resize_ratio))
    mapped_h1 = int(round(float(h1) / resize_ratio))
    mapped_w2 = int(round(float(w2) / resize_ratio))
    mapped_h2 = int(round(float(h2) / resize_ratio))
    line = (mapped_h1, mapped_w1), (mapped_h2, mapped_w2)
    basins = plate.feature_stash['iterated_basins']
    basin = basins[mapped_h1, mapped_w1]  # Note h, w reversed vs event
    if basin == 0:
        print("This is background; not splitting.")
        return
    plate.linear_split_basin(feature_out='iterated_basins',
                             basins_feature='iterated_basins',
                             line=line,
                             target_basin=basin,
                            )
    remeasure_basins(plate)
    color_image = plate.image_stash['rescaled_image']
    pil_image = make_pil_image(color_image=color_image,
                               basins=plate.feature_stash['iterated_basins'],
                               resize_ratio=resize_ratio,
                              )
    tk_image = ImageTk.PhotoImage(image=pil_image)
    canvas.itemconfig(canvas_image, image=tk_image)
    stdout.write("complete\n")
    stdout.flush()


linear_split_button = tk.Button(bottom_frame,
                                text="Linear split",
                                command=linear_split,
                               )
linear_split_button.grid(column=4, row=1)

solvent_front = None
solvent_front_line = None

def solvent():
    global left_click_buffer, left_click_buffer_size, solvent_front, plate
    global canvas, pil_image, solvent_front_line
    if len(left_click_buffer) < 2:
        print("Insufficient points defined")
        return
    line = (w1, h1), (w2, h2) = left_click_buffer[-2:]
    mapped_w1 = int(round(float(w1) / resize_ratio))
    mapped_h1 = int(round(float(h1) / resize_ratio))
    mapped_w2 = int(round(float(w2) / resize_ratio))
    mapped_h2 = int(round(float(h2) / resize_ratio))
    mapped_line = (mapped_w1, mapped_h1), (mapped_w2, mapped_h2)
    extended_line = appaloosa.Plate.extend_line(
                                  line=mapped_line,
                                  image=plate.feature_stash['iterated_basins'],
                                               )
    plate.feature_stash['solvent_front'] = extended_line
    pil_w, pil_h = pil_image.size
    (ew1, eh1), (ew2, eh2) = appaloosa.Plate.extend_line(line=line,
                                                         image=np.zeros((pil_h, pil_w)),
                                                        )
    if solvent_front is not None:
        canvas.delete(solvent_front)
    solvent_front_line = ((mapped_w1, mapped_h1), (mapped_w2, mapped_h2))
    solvent_front = canvas.create_line(ew1, eh1, ew2, eh2, fill='red')

solvent_front_button = tk.Button(bottom_frame,
                                 text="Solvent front",
                                 command=solvent,
                                )
solvent_front_button.grid(column=5, row=1)

baseline_colors = ['orange',
                   'orange red',
                   'deep pink',
                   'maroon',
                  ]
baselines = []

def add_baseline():
    global left_click_buffer, plate, canvas, pil_image, solvent_front
    global baseline_colors, baselines, solvent_front_line, pil_image
    global resize_ratio
    if solvent_front is None:
        print("Solvent front not defined")
        return
    if len(left_click_buffer) < 1:
        print("Insufficient points defined")
        return
    if len(baselines) >= len(baseline_colors):
        print("All baselines have been defined; ignoring.")
        return
    point = w, h = left_click_buffer[-1]
    mapped_w = int(round(float(w) / resize_ratio))
    mapped_h = int(round(float(h) / resize_ratio))
    mapped_point = mapped_w, mapped_h
    projected_w, projected_h = appaloosa.Plate.project_point_on_segment(
                                                    point=mapped_point,
                                                    segment=solvent_front_line,
                                                                       )
    delta_w, delta_h = mapped_w - projected_w, mapped_h - projected_h
    basins = plate.feature_stash['iterated_basins']
    baseline = appaloosa.Plate.translate_line(line=solvent_front_line,
                                              h=delta_h, w=delta_w,
                                              extend=True,
                                              image=np.zeros_like(basins),
                                             )
    baseline_dict = plate.feature_stash.setdefault('baselines', {})
    baseline_dict[len(baselines)] = baseline
    baselines.append(baseline)
    (sw1, sh1), (sw2, sh2) = baseline
    ew1 = int(round(sw1 * resize_ratio))
    eh1 = int(round(sh1 * resize_ratio))
    ew2 = int(round(sw2 * resize_ratio))
    eh2 = int(round(sh2 * resize_ratio))
    baseline_color = baseline_colors[len(baselines) - 1]
    baseline = canvas.create_line(ew1, eh1, ew2, eh2, fill=baseline_color)

baseline_button = tk.Button(bottom_frame,
                            text="Add baseline",
                            command=add_baseline,
                           )
baseline_button.grid(column=6, row=1)

base_assign_state = 0

def base1_assign():
    global base1_assign_button
    global base2_assign_button
    global base3_assign_button
    global base4_assign_button
    global base_assign_state
    if base_assign_state != 1:
        base_assign_state = 1
        base1_assign_button.config(relief=tk.SUNKEN)
        base2_assign_button.config(relief=tk.RAISED)
        base3_assign_button.config(relief=tk.RAISED)
        base4_assign_button.config(relief=tk.RAISED)
    elif base_assign_state == 1:
        base_assign_state = 0
        base1_assign_button.config(relief=tk.RAISED)
        base2_assign_button.config(relief=tk.RAISED)
        base3_assign_button.config(relief=tk.RAISED)
        base4_assign_button.config(relief=tk.RAISED)

def base2_assign():
    global base1_assign_button
    global base2_assign_button
    global base3_assign_button
    global base4_assign_button
    global base_assign_state
    if base_assign_state != 2:
        base_assign_state = 2
        base1_assign_button.config(relief=tk.RAISED)
        base2_assign_button.config(relief=tk.SUNKEN)
        base3_assign_button.config(relief=tk.RAISED)
        base4_assign_button.config(relief=tk.RAISED)
    elif base_assign_state == 2:
        base_assign_state = 0
        base1_assign_button.config(relief=tk.RAISED)
        base2_assign_button.config(relief=tk.RAISED)
        base3_assign_button.config(relief=tk.RAISED)
        base4_assign_button.config(relief=tk.RAISED)

def base3_assign():
    global base1_assign_button
    global base2_assign_button
    global base3_assign_button
    global base4_assign_button
    global base_assign_state
    if base_assign_state != 3:
        base_assign_state = 3
        base1_assign_button.config(relief=tk.RAISED)
        base2_assign_button.config(relief=tk.RAISED)
        base3_assign_button.config(relief=tk.SUNKEN)
        base4_assign_button.config(relief=tk.RAISED)
    elif base_assign_state == 3:
        base_assign_state = 0
        base1_assign_button.config(relief=tk.RAISED)
        base2_assign_button.config(relief=tk.RAISED)
        base3_assign_button.config(relief=tk.RAISED)
        base4_assign_button.config(relief=tk.RAISED)

def base4_assign():
    global base1_assign_button
    global base2_assign_button
    global base3_assign_button
    global base4_assign_button
    global base_assign_state
    if base_assign_state != 4:
        base_assign_state = 4
        base1_assign_button.config(relief=tk.RAISED)
        base2_assign_button.config(relief=tk.RAISED)
        base3_assign_button.config(relief=tk.RAISED)
        base4_assign_button.config(relief=tk.SUNKEN)
    elif base_assign_state == 4:
        base_assign_state = 0
        base1_assign_button.config(relief=tk.RAISED)
        base2_assign_button.config(relief=tk.RAISED)
        base3_assign_button.config(relief=tk.RAISED)
        base4_assign_button.config(relief=tk.RAISED)

base1_assign_button = tk.Button(bottom_frame,
                                text="1",
                                command=base1_assign,
                               )
base1_assign_button.grid(column=4, row=2)

base2_assign_button = tk.Button(bottom_frame,
                                text="2",
                                command=base2_assign,
                                )
base2_assign_button.grid(column=5, row=2)

base3_assign_button = tk.Button(bottom_frame,
                                text="3",
                                command=base3_assign,
                               )
base3_assign_button.grid(column=6, row=2)

base4_assign_button = tk.Button(bottom_frame,
                                text="4",
                                command=base4_assign,
                               )
base4_assign_button.grid(column=7, row=2)

def assign(event):
    global base_assign_state, plate, canvas, canvas_image, pil_image, tk_image
    global resize_ratio
    if base_assign_state == 0:
        print("No baseline chosen for assignment; ignoring.")
        return
    if len(baselines) < base_assign_state:
        print("baseline #"
              + str(base_assign_state)
              + " not yet defined; ignoring."
             )
        return
    w, h = event.x, event.y
    mapped_w = int(round(float(w) / resize_ratio))
    mapped_h = int(round(float(h) / resize_ratio))
    basins = plate.feature_stash['iterated_basins']
    basin = basins[mapped_h, mapped_w]
    if basin == 0:
        print("Clicked on background; not assigned.")
        return
    assignments = plate.feature_stash.setdefault('base_assignments', {})
    assignments[basin] = base_assign_state
    baseline = plate.feature_stash['baselines'][base_assign_state - 1]
    plate.feature_stash['temp_baseline'] = baseline
    basin_centroids = plate.feature_stash['basin_centroids']
    filtered_basin_centroids = {Label: centroid
                                for Label, centroid
                                in basin_centroids.iteritems()
                                if (Label in assignments
                                    and assignments[Label] == base_assign_state
                                   )
                               }
    plate.feature_stash['temp_basin_centroids'] = filtered_basin_centroids
    plate.compute_basin_rfs(basin_centroids_feature='temp_basin_centroids',
                            baseline_feature='temp_baseline',
                            solvent_front_feature='solvent_front',
                            feature_out='temp_basin_rfs',
                           )
    indexed_basin_rfs = plate.feature_stash.setdefault('indexed_basin_rfs', {})
    indexed_basin_rfs[base_assign_state] = plate.feature_stash['temp_basin_rfs']
    color_image = plate.image_stash['rescaled_image']
    pil_image = make_pil_image(color_image=color_image,
                               basins=plate.feature_stash['iterated_basins'],
                               resize_ratio=resize_ratio,
                               assignment_feature='base_assignments',
                              )
    tk_image = ImageTk.PhotoImage(image=pil_image)
    canvas.itemconfig(canvas_image, image=tk_image)

canvas.bind('<Double-Button-1>', assign)

canvas.focus_set()

basin_texts = {}

def make_boolean_circle(image,
                        h, w,
                        radius,
                       ):
    h, w, radius = int(round(h)), int(round(w)), int(round(radius))
    image_height, image_width = image.shape[:2]
    hh, ww = np.mgrid[:image_height,:image_width]
    radial_distance_sq = (hh - h)**2 + (ww - w)**2
    boolean_circle_array = (radial_distance_sq <= radius**2)
    return boolean_circle_array

def circle_filter(image,
                  basins,
                  basin,
                  radius,
                 ):
    (best_h,
     best_w,
     best_circle,
     best_value,
    ) = appaloosa.Plate.best_circle(
                       image=image,
                       basins=basins,
                       basin=basin,
                       radius=radius,
                                   )
    logical_filter = np.where((basins == basin) & ~best_circle,
                              True,
                              False,
                             )
    updated_basins = np.where(logical_filter,
                              np.zeros_like(basins),
                              basins,
                             )
    return updated_basins

def circle_filter_all(image,
                      basins,
                      radius,
                     ):
    basin_set = set(basins.reshape(-1))
    updated_basins = basins
    for basin in iter(basin_set):
        updated_basins = circle_filter(image=image,
                                       basins=updated_basins,
                                       basin=basin,
                                       radius=radius,
                                      )
    return updated_basins

def circle_filter_all_button():
    mode = 'isolated_LoG_MP'
    if mode == 'LoG':
        stdout.write("Applying circle filter to all basins...")
        stdout.flush()
        global plate, circle_filter_entry
        radius = int(circle_filter_entry.get())
        plate.image_stash['inverted_corrected_rescaled_image'] = \
                          invert(plate.image_stash['corrected_rescaled_image'])
        plate.find_blobs(tag_in='inverted_corrected_rescaled_image',
                         feature_out='blobs_log',
                         min_sigma=5,
                         max_sigma=max_radius,
                         num_sigma=10,
                         threshold=0.01,
                         overlap=0.5,
                        )
        #plate.display(tag_in='corrected_rescaled_image',
        #              figsize=20,
        #              blobs_feature='blobs_log',
        #              output_filename='BLOBS.png',
        #             )
        basins = plate.feature_stash['iterated_basins']
        per_basin_blobs = defaultdict(list)
        for blob in plate.feature_stash['blobs_log']:
            h, w, r = blob
            h, w, r = int(h), int(w), int(r)
            basin = basins[h, w]
            if basin == 0:
                continue
            per_basin_blobs[basin].append((h, w, r))
        largest_per_basin_blobs = {basin: max(blobs, key=lambda x:x[2])
                                   for basin, blobs
                                   in per_basin_blobs.iteritems()
                                  }
        circle_scaling = 1.5
        for basin, (h, w, r) in largest_per_basin_blobs.iteritems():
            boolean_circle_array = appaloosa.Plate.make_boolean_circle(
                           image=plate.image_stash['corrected_rescaled_image'],
                           h=h, w=w,
                           radius=r,
                                                                      )
            updated_basins = circle_filter(image=plate.image_stash['corrected_rescaled_image'],
                                           basins=plate.feature_stash['iterated_basins'],
                                           basin=basin,
                                           radius=r*circle_scaling,
                                          )
            plate.feature_stash['iterated_basins'] = updated_basins
        remeasure_basins(plate)
        global tk_image, canvas, pil_image, canvas_image, resize_ratio
        color_image = plate.image_stash['rescaled_image']
        pil_image = make_pil_image(
                                 color_image=color_image,
                                 basins=plate.feature_stash['iterated_basins'],
                                 resize_ratio=resize_ratio,
                                  )
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.itemconfig(canvas_image, image=tk_image)
        stdout.write("complete\n")
        stdout.flush()
    elif mode == 'isolated_LoG':
        raise NotImplementedError("Superseded by isolated_LoG_MP")
        stdout.write("Applying circle filter to all basins...")
        stdout.flush()
        global plate, circle_filter_entry
        max_radius = int(circle_filter_entry.get())
        plate.image_stash['inverted_corrected_rescaled_image'] = \
                          invert(plate.image_stash['corrected_rescaled_image'])
        basins = plate.feature_stash['iterated_basins']
        per_basin_blobs = defaultdict(list)
        all_blobs = []
        inverted_image_min = \
                np.amin(plate.image_stash['inverted_corrected_rescaled_image'])
        for basin in iter(set(basins.reshape(-1))):
            if basin == 0:
                continue
            plate.image_stash['isolated_image'] = np.where(
                         basins == basin,
                         plate.image_stash['inverted_corrected_rescaled_image'],
                         inverted_image_min,
                                                          )
            feature_name = 'isolated_' + str(basin) + '_blobs'
            plate.find_blobs(tag_in='isolated_image',
                             feature_out=feature_name,
                             min_sigma=5,
                             max_sigma=max_radius,
                             num_sigma=10,
                             threshold=0.01,
                             overlap=0.5,
                            )
            for blob in plate.feature_stash[feature_name]:
                h, w, r = blob
                h, w, r = int(h), int(w), int(r)
                per_basin_blobs[basin].append((h, w, r))
                all_blobs.append((h, w, r))
        plate.feature_stash['all_blobs'] = all_blobs
        #plate.display(tag_in='corrected_rescaled_image',
        #              figsize=20,
        #              blobs_feature='all_blobs',
        #              output_filename='ISOLATED_BLOBS.png',
        #             )
        largest_per_basin_blobs = {basin: max(blobs, key=lambda x:x[2])
                                   for basin, blobs
                                   in per_basin_blobs.iteritems()
                                  }
        circle_scaling = 1.5
        for basin, (h, w, r) in largest_per_basin_blobs.iteritems():
            boolean_circle_array = appaloosa.Plate.make_boolean_circle(
                           image=plate.image_stash['corrected_rescaled_image'],
                           h=h, w=w,
                           radius=r,
                                                                      )
            updated_basins = circle_filter(image=plate.image_stash['corrected_rescaled_image'],
                                           basins=plate.feature_stash['iterated_basins'],
                                           basin=basin,
                                           radius=r*circle_scaling,
                                          )
            plate.feature_stash['iterated_basins'] = updated_basins
        remeasure_basins(plate)
        global tk_image, canvas, pil_image, canvas_image, resize_ratio
        color_image = plate.image_stash['rescaled_image']
        pil_image = make_pil_image(
                                 color_image=color_image,
                                 basins=plate.feature_stash['iterated_basins'],
                                 resize_ratio=resize_ratio,
                                  )
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.itemconfig(canvas_image, image=tk_image)
        stdout.write("complete\n")
        stdout.flush()
    elif mode == 'isolated_LoG_MP':
        stdout.write("Applying circle filter to all basins...")
        stdout.flush()
        global plate, circle_filter_entry
        max_radius = int(circle_filter_entry.get())
        plate.image_stash['inverted_corrected_rescaled_image'] = \
                          invert(plate.image_stash['corrected_rescaled_image'])
        basins = plate.feature_stash['iterated_basins']
        per_basin_blobs = defaultdict(list)
        all_blobs = []
        inverted_image_min = \
                np.amin(plate.image_stash['inverted_corrected_rescaled_image'])
        pool = multiprocessing.Pool(processes=None,
                                    maxtasksperchild=None,
                                   )
        processes = []
        for basin in iter(set(basins.reshape(-1))):
            if basin == 0:
                continue
            plate.image_stash['isolated_image'] = np.where(
                        basins == basin,
                        plate.image_stash['inverted_corrected_rescaled_image'],
                        inverted_image_min,
                                                          )
            isolated_image = plate.image_stash['isolated_image']
            min_sigma = 5
            max_sigma = max_radius
            num_sigma = 10
            threshold = 0.01
            overlap = 0.5
            process = pool.apply_async(blob_log,
                                       (isolated_image,
                                        min_sigma,
                                        max_sigma,
                                        num_sigma,
                                        threshold,
                                        overlap,
                                       )
                                      )
            processes.append((basin, process))
        pool.close()
        pool.join()
        for basin, process in processes:
            blobs = process.get()
            for blob in blobs:
                h, w, r = blob
                h, w, r = int(h), int(w), int(r)
                per_basin_blobs[basin].append((h, w, r))
                all_blobs.append((h, w, r))
        plate.feature_stash['all_blobs'] = all_blobs
        #plate.display(tag_in='corrected_rescaled_image',
        #              figsize=20,
        #              blobs_feature='all_blobs',
        #              output_filename='ISOLATED_BLOBS.png',
        #             )
        #best_per_basin_blobs = {basin: max(blobs, key=lambda x:x[2])
        #                        for basin, blobs
        #                        in per_basin_blobs.iteritems()
        #                       }
        best_per_basin_blobs = {}
        for basin, blobs in per_basin_blobs.iteritems():
            best_h, best_w, best_r, best_value = None, None, None, None
            for blob in blobs:
                h, w, r = blob
                h, w, r = int(h), int(w), int(r)
                image = plate.image_stash['corrected_rescaled_image']
                boolean_circle = make_boolean_circle(image=image,
                                                     h=h, w=w,
                                                     radius=r,
                                                    )
                circle_sum = np.sum(np.where(boolean_circle, image, 0))
                circle_area = np.sum(np.where(boolean_circle, 1, 0))
                circle_value = circle_sum / float(circle_area)
                if best_value is None or circle_value < best_value:
                    best_h = h
                    best_w = w
                    best_r = r
                    best_value = circle_value
            best_per_basin_blobs[basin] = (best_h, best_w, best_r)
        circle_scaling = 1.5
        pool = multiprocessing.Pool(processes=None,
                                    maxtasksperchild=None,
                                   )
        processes = []
        updated_basin_list = []
        for basin, (h, w, r) in best_per_basin_blobs.iteritems():
            image = plate.image_stash['corrected_rescaled_image']
            radius = r
            process = pool.apply_async(make_boolean_circle,
                                       (image,
                                        h, w,
                                        radius*circle_scaling,
                                       )
                                      )
            processes.append((basin, process))
        pool.close()
        pool.join()
        updated_basins = basins
        for basin, process in processes:
            boolean_circle_array = process.get()
            logical_filter = np.where(
                             (updated_basins == basin) & ~boolean_circle_array,
                             False,
                             True,
                                     )
            updated_basins = np.where(logical_filter,
                                      updated_basins,
                                      np.zeros_like(updated_basins),
                                     )
        plate.feature_stash['iterated_basins'] = updated_basins
        remeasure_basins(plate)
        global tk_image, canvas, pil_image, canvas_image, resize_ratio
        color_image = plate.image_stash['rescaled_image']
        pil_image = make_pil_image(
                                 color_image=color_image,
                                 basins=plate.feature_stash['iterated_basins'],
                                 resize_ratio=resize_ratio,
                                  )
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.itemconfig(canvas_image, image=tk_image)
        stdout.write("complete\n")
        stdout.flush()

    elif mode == 'manual':
        stdout.write("Applying circle filter to all basins...")
        stdout.flush()
        global plate, circle_filter_entry
        basins = plate.feature_stash['iterated_basins']
        image = plate.image_stash['corrected_rescaled_image']
        radius = int(circle_filter_entry.get())
        updated_basins = circle_filter_all(image=image,
                                           basins=basins,
                                           radius=radius,
                                          )
        plate.feature_stash['iterated_basins'] = updated_basins
        remeasure_basins(plate)
        global tk_image, canvas, pil_image, canvas_image, resize_ratio
        color_image = plate.image_stash['rescaled_image']
        pil_image = make_pil_image(
                                 color_image=color_image,
                                 basins=plate.feature_stash['iterated_basins'],
                                 resize_ratio=resize_ratio,
                                  )
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.itemconfig(canvas_image, image=tk_image)
        stdout.write("complete\n")
        stdout.flush()
    else:
        print("Invalid circle filter mode...skipping.")

def keyboard(event):
    char = event.char
    global resize_ratio, plate, canvas
    w, h = event.x, event.y
    mapped_w = int(round(float(w) / resize_ratio))
    mapped_h = int(round(float(h) / resize_ratio))
    basins = plate.feature_stash['iterated_basins']
    basin = basins[mapped_h, mapped_w]
    if basin == 0:
        print("Mouse over background; ignoring.")
        return
    height, width = basins.shape
    outside_bounds = (mapped_w < 0
                      or mapped_h < 0
                      or mapped_w >= width
                      or mapped_h >= height
                     )
    if outside_bounds:
        return
    if char == 'd':
        basin_intensity = plate.feature_stash['basin_intensities'][basin]
        ch, cw = basin_centroid = plate.feature_stash['basin_centroids'][basin]
        pcw = int(round(cw * resize_ratio))
        pch = int(round(ch * resize_ratio))
        basin_text = "I = " + str(basin_intensity)
        indexed_basin_rfs = plate.feature_stash.get('indexed_basin_rfs', None)
        rf_text = ""
        if indexed_basin_rfs is not None:
            for base_assign_state, basin_rfs in indexed_basin_rfs.iteritems():
                for b, rf in basin_rfs.iteritems():
                    if b == basin:
                        rf_text = "\nRf = " + str(round(rf, 2))
        basin_text += rf_text
        global basin_texts
        if basin in basin_texts:
            canvas.delete(basin_texts[basin])
        bt = canvas.create_text(pcw, pch,
                                text=basin_text,
                               )
        basin_texts[basin] = bt
    elif char == 'c':
        stdout.write("Applying circle filter...")
        stdout.flush()
        mode = 'isolated_LoG'
        global plate, circle_filter_entry
        if mode == 'isolated_LoG':
            max_radius = int(circle_filter_entry.get())
            plate.image_stash['inverted_corrected_rescaled_image'] = \
                          invert(plate.image_stash['corrected_rescaled_image'])
            inverted_image_min = \
                np.amin(plate.image_stash['inverted_corrected_rescaled_image'])
            plate.image_stash['isolated_image'] = np.where(
                        basins == basin,
                        plate.image_stash['inverted_corrected_rescaled_image'],
                        inverted_image_min,
                                                          )
            plate.find_blobs(tag_in='isolated_image',
                             feature_out='isolated_blobs',
                             min_sigma=5,
                             max_sigma=max_radius,
                             num_sigma=10,
                             threshold=0.01,
                             overlap=0.5,
                            )
            blobs = plate.feature_stash['isolated_blobs']
            best_h, best_w, best_r, best_value = None, None, None, None
            image = plate.image_stash['corrected_rescaled_image']
            for blob in blobs:
                h, w, r = blob
                h, w, r = int(h), int(w), int(r)
                boolean_circle = make_boolean_circle(image=image,
                                                     h=h, w=w,
                                                     radius=r,
                                                    )
                circle_sum = np.sum(np.where(boolean_circle, image, 0))
                circle_area = np.sum(np.where(boolean_circle, 1, 0))
                circle_value = circle_sum / float(circle_area)
                if best_value is None or circle_value < best_value:
                    best_h = h
                    best_w = w
                    best_r = r
                    best_value = circle_value
            circle_scaling = 1.5
            boolean_circle_array = make_boolean_circle(
                                                  image=image,
                                                  h=best_h, w=best_w,
                                                  radius=best_r*circle_scaling,
                                                      )
            updated_basins = basins
            logical_filter = np.where(
                             (updated_basins == basin) & ~boolean_circle_array,
                             False,
                             True,
                                     )
            updated_basins = np.where(logical_filter,
                                      updated_basins,
                                      np.zeros_like(updated_basins),
                                     )
            plate.feature_stash['iterated_basins'] = updated_basins
        elif mode == 'manual':
            radius = int(circle_filter_entry.get())
            updated_basins = circle_filter(
                           image=plate.image_stash['corrected_rescaled_image'],
                           basins=basins,
                           basin=basin,
                           radius=radius,
                                          )
            plate.feature_stash['iterated_basins'] = updated_basins
        else:
            print("Unrecognized circle filter mode; ignoring.")
        remeasure_basins(plate)
        global tk_image, canvas, pil_image, canvas_image, resize_ratio
        color_image = plate.image_stash['rescaled_image']
        pil_image = make_pil_image(
                                 color_image=color_image,
                                 basins=plate.feature_stash['iterated_basins'],
                                 resize_ratio=resize_ratio,
                                  )
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.itemconfig(canvas_image, image=tk_image)

        stdout.write("complete\n")
        stdout.flush()
    else:
        pass


canvas.bind('<Key>', keyboard)

def add_basin():
    global left_click_buffer, resize_ratio
    if len(left_click_buffer) < 2:
        print("Insufficient points defined")
        return
    stdout.write("Adding spot...")
    stdout.flush()
    (w1, h1), (w2, h2) = left_click_buffer[-2:]
    mapped_w1 = int(round(float(w1) / resize_ratio))
    mapped_h1 = int(round(float(h1) / resize_ratio))
    mapped_w2 = int(round(float(w2) / resize_ratio))
    mapped_h2 = int(round(float(h2) / resize_ratio))
    center_h = float(mapped_h1 + mapped_h2) / 2
    center_w = float(mapped_w1 + mapped_w2) / 2
    radius = euclidean((mapped_w1, mapped_h1), (mapped_w2, mapped_h2)) / 2.0
    global plate
    basins = plate.feature_stash['iterated_basins']
    largest_basins_tag = np.amax(basins)
    print("largest_basins_tag = " + str(largest_basins_tag))
    new_basin_tag = largest_basins_tag + 1
    updated_basins = basins.copy()
    #min_h, max_h = min(mapped_h1, mapped_h2), max(mapped_h1, mapped_h2)
    #min_w, max_w = min(mapped_w1, mapped_w2), max(mapped_w1, mapped_w2)
    min_h, max_h = center_h - radius, center_h + radius
    min_w, max_w = center_w - radius, center_w + radius
    for (h, w), basin in np.ndenumerate(basins):
        if not (min_h <= h <= max_h and min_w <= w <= max_w):
            continue
        distance = euclidean((h, w), (center_h, center_w))
        if distance > radius:
            continue
        updated_basins[h, w] = new_basin_tag
    plate.feature_stash['iterated_basins'] = updated_basins
    remeasure_basins(plate)
    global tk_image, canvas, pil_image
    color_image = plate.image_stash['rescaled_image']
    pil_image = make_pil_image(color_image=color_image,
                               basins=plate.feature_stash['iterated_basins'],
                               resize_ratio=resize_ratio,
                               assignment_feature='base_assignments',
                              )
    tk_image = ImageTk.PhotoImage(image=pil_image)
    canvas.itemconfig(canvas_image, image=tk_image)
    stdout.write("complete\n")
    stdout.flush()

add_basin_button = tk.Button(bottom_frame,
                             text="Add spot",
                             command=add_basin,
                            )
add_basin_button.grid(column=7, row=1)

def post_front():
    global plate
    solvent_front_line = plate.feature_stash.get('solvent_front', None)
    if solvent_front_line is None:
        print("Solvent front not defined; ignoring.")
        return
    (mapped_w1, mapped_h1), (mapped_w2, mapped_h2) = solvent_front_line
    basins = plate.feature_stash['iterated_basins']
    split_plate = np.zeros_like(basins).astype(np.bool)
    if mapped_h1 == mapped_h2:
        for (h, w) in np.ndindex(*split_plate.shape):
            if h > mapped_h1:
                split_plate[h, w] = True
    elif mapped_w1 == mapped_w2:
        for (h, w) in np.ndindex(*split_plate.shape):
            if w > mapped_w1:
                split_plate[h, w] = True
    else:
        slope = float(mapped_w2 - mapped_w1) / (mapped_h2 - mapped_h1)
        for (h, w) in np.ndindex(*split_plate.shape):
            coord_value = slope * (h - mapped_h1) + mapped_w1
            if coord_value > w:
                split_plate[h, w] = True
    global left_click_buffer
    if len(left_click_buffer) < 1:
        print("No point defined; ignoring.")
        return
    w, h = left_click_buffer[-1]
    global resize_ratio
    mapped_w = int(round(float(w) / resize_ratio))
    mapped_h = int(round(float(h) / resize_ratio))
    if appaloosa.Plate.point_line_distance(point=(mapped_h, mapped_w),
                                           line=((mapped_h1, mapped_w1),
                                                 (mapped_h2, mapped_w2)),
                                          ) < 1:
        print("Point too close to solvent front; ignoring.")
        return
    basin_centroids = plate.feature_stash['basin_centroids']
    to_delete = []
    for basin, (ch, cw) in basin_centroids.iteritems():
        ich, icw = int(round(ch)), int(round(cw))
        if split_plate[ich, icw] == split_plate[mapped_h, mapped_w]:
            to_delete.append(basin)
    for delete_basin in to_delete:
        basins = np.where(basins == delete_basin,
                          0,
                          basins,
                         )
    plate.feature_stash['iterated_basins'] = basins
    remeasure_basins(plate)
    global pil_image, tk_image, canvas
    color_image = plate.image_stash['rescaled_image']
    pil_image = make_pil_image(color_image=color_image,
                               basins=plate.feature_stash['iterated_basins'],
                               resize_ratio=resize_ratio,
                              )
    tk_image = ImageTk.PhotoImage(image=pil_image)
    canvas.itemconfig(canvas_image, image=tk_image)

post_front_button = tk.Button(bottom_frame,
                              text="Remove spots above front",
                              command=post_front,
                             )
post_front_button.grid(column=3, row=1)

pil_cache = None

def overlay_original(event):
    global plate, pil_image, tk_image, canvas, pil_cache, resize_ratio
    global canvas_image
    pil_cache = pil_image
    color_image = plate.image_stash['rescaled_image']
    uint8_image = np.rint(color_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(uint8_image)
    image_width, image_height = pil_image.size
    resized_width = int(round(image_width * resize_ratio))
    resized_height = int(round(image_height * resize_ratio))
    pil_image = pil_image.resize((resized_width, resized_height))
    tk_image = ImageTk.PhotoImage(image=pil_image)
    canvas.itemconfig(canvas_image, image=tk_image)

def unoverlay_original(event):
    global plate, pil_image, tk_image, canvas, pil_cache, resize_ratio
    global canvas_image
    pil_image = pil_cache
    tk_image = ImageTk.PhotoImage(image=pil_image)
    canvas.itemconfig(canvas_image, image=tk_image)

show_original_button = tk.Button(bottom_frame,
                                 text="Show original",
                                )
show_original_button.grid(column=1, row=3)
show_original_button.bind('<ButtonPress-1>', overlay_original)
show_original_button.bind('<ButtonRelease-1>', unoverlay_original)

circle_filter_label = tk.Label(bottom_frame,
                               text="Circle filter sigma",
                              )
circle_filter_label.grid(column=3, row=3)
circle_filter_entry = tk.Entry(bottom_frame)
circle_filter_entry.insert(0, 10)
circle_filter_entry.grid(column=4, row=3)

circle_filter_all_button = tk.Button(bottom_frame,
                                     text="Circle filter all",
                                     command=circle_filter_all_button,
                                    )
circle_filter_all_button.grid(column=5, row=3)

root.mainloop()
