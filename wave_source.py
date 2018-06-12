import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


# Needed to handle exits properly
import sys, traceback

# Stack Overflow algorithms
def bisection(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0 # Initialize lower
    ju = n-1 # and upper limits.
    while (ju-jl > 1): # If we are not yet done,
        jm=(ju+jl) >> 1 # compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm # and replace either the lower limit
        else:
            ju=jm # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]): # and top
        return n-1
    else:
        return jl

# Here we import the file min_time.txt as an OrderedDict
filename = "/home/paolo/Scrivania/Universita'/Human Brain Project/Slow Waves Project/data/min_time.txt"

width = 50
height = 40

print "Loading...",
# Here we load the file
with open(filename) as f:
    # We built is as a dict with key = pixel-ID ; value = activation-time
    trans_dict = OrderedDict()
    last_up_time = -1
    first_up_time = 1E9

    last_id = -1
    first_id = -1

    for line in f:
        pos = line.find('[')
        key = int(line[:pos])
        try:
            value = [float(x) for x in line[pos + 1:-3].split(",")]

            # We keep track of the first and last up transition
            if last_up_time < max(value):
                last_up_time = max(value)
                last_id = np.argmax(value)
                key_last = key

            if first_up_time > min(value):
                first_up_time = min(value)
                first_id = np.argmin(value)
                key_first = key



        except ValueError as err:
            value = []

        trans_dict[key] = value

print "Dictionary is loaded"

if trans_dict.keys()[-1] + 1 != width * height:
    print "\nError on initiFromFile: pixels-IDs do not math simulation dimentions"
    print "Max pixels-ID is %d while Width x Height is %d\n" % (trans_dict.keys()[-1] + 1, width * height)
    sys.exit(-1)

print "First up transition detected at time T = %lf for ID = %d" % (first_up_time, key_first)
print "Last up transition detected at time T = %lf for ID = %d" % (last_up_time, key_last)

# Here we build two arrays: a time array in which all activation times for all
# the pixels are stored (and ordered) and an idx-array in which the corresponding
# sorted pixels idx values are saved.

up_times = []
up_idx = []
for key in trans_dict.keys():
    up_times.extend(trans_dict[key])
    up_idx.extend([key for i in range(len(trans_dict[key]))])

# Now we order our time and idx arrays
ord_up_times = [x for x,_  in sorted(zip(up_times, up_idx))]
ord_idx = [y for _, y in sorted(zip(up_times, up_idx))]

# Here we defined the number of elements (of pixels idx's) past-to the closest value
# of wave born time to be consider as part of the wave birth itself
k_cluster = 30

# Here we manually define a set of time for wave segmentation
wave_times = np.linspace(start=0, stop = last_up_time, num = 30)

# Here we define the image grid
im_grid = np.zeros((height, width))

for key in trans_dict:
    if trans_dict[key] == []:
        col = key % width
        row = key // width

        im_grid[row][col] = np.nan


# Here we find the closest elements of ord_time to wave_times[i]
for trigger in wave_times:
    j_match = bisection(ord_up_times, trigger)

    # Here we grab the corresponding idx of pixels
    source_idx = np.array(ord_idx[j_match : j_match + k_cluster])

    # Here we deduce the rows and colums idxs
    colums = source_idx % width
    rows = source_idx // width

    for row, col in zip(rows, colums):
        im_grid[row][col] += 1

plt.imshow(im_grid)
plt.colorbar()
plt.show()
