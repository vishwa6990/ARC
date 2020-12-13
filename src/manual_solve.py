
"""
Student name: Vishwa Kumar
Student ID: 20236183
Github URL: https://github.com/vishwa6990/ARC

Summary/Reflection:
Following are the some of the unique python features used in the solve_* functions
    List comprehension
    List indexing, traversals, sorting of lists
    Inline lambda functions
    Tuple unpacking
Numpy library is heavily used in solve_* functions. Few important functions/features used from this library are
    Copying of numpy arrays
    Initializing array and filling with values
    Conditional analysis and replacement of array values
    Rolling the array on a given axis
Commonalities/Differences between the 3 tasks
   For couple of tasks, output grid is initialized with the input grid and then transformations are applied.
   For couple of tasks, unique colors from the input grid are identified and then transformation rules are applied on top of it
   For two out of 3 tasks, completely different approach was used like in one task had to construct the output grid size
   based on the patterns/shapes in input grid and in other task had to traverse the grid from all 4 directions(top,bottom,left,right)
   to find the patterns.
"""

import os
import json
import numpy as np
import re
from enum import Enum


# Enum of color names and values to be used in ARC examples
class Color(Enum):
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    CYAN = 8
    BROWN = 9


def solve_f76d97a5(x):
    """
    Given input grid will have two colors filled in it.
    Input grid will have always grey as one color other could be any other color except black and grey.
    Output grid will have colors modified in it by applying the following transformations.
        1. Initialize the output grid with all contents from input grid
        2. Update all the unique colored squares(except grey color) with black color
        3. Update all the grey colored squares with unique color present in grid

    Completeness - All available training and test grids for this task are solved correctly
    """
    # Generate the output grid by making a copy of the input grid
    output = np.copy(x)

    # Find the unique colors present in input grid, In the given training & test set we will have only two unique colors
    unique_elem = np.unique(output)
    for item in unique_elem:
        # Apply the transformations only when processing the unique color as the required transformations
        if item != Color.GREY.value:
            output = np.where(output == item, Color.BLACK.value, output)  # Update all unique colored squares with black color
            output = np.where(output == Color.GREY.value, item, output)  # Update all grey colored squares with color present in grid
    return output


def solve_c8cbb738(x):
    """
    Given input grid could be of any size m * n
    Output grid will be of size i * i where size i and contents will be determined based on the patterns present in input grid
    Patterns found in input grid
        1. Two or more unique colors with a single color prominently present in cells which doesnt
        2. Shapes like square, rectangle and diamonds could be formed by connecting the same colored cells
    Transformations
        1. Determine the size of the biggest square shaped colored cells and generate output grid of that size filled with dominant color
        2. Now start looking at the different colored cells, map it in the output grid in such a way that it must be in
           into same shape(rectangle / diamond/ color) and size as present in input grid

    Completeness - All available training and test grids for this task are solved correctly
    """
    # Find the unique colors in input grid and number of occurences of each color
    unique_elem = np.unique(x)
    occurences_count = [(elem, (x == elem).sum()) for elem in unique_elem]

    # Find the dominant color and less frequent colors
    # In the given examples, all less frequent colors are present with equal occurence and map to outside edges in output grid
    maximum_num, max_count = sorted(occurences_count, key=lambda tup: tup[1], reverse=True)[0]
    minimum_num, min_count = sorted(occurences_count, key=lambda tup: tup[1])[0]

    # Find the vertices of the less frequent color and determine the maximum distance in X axis.
    # Using the calculated distance, find the centre position and generate the output grid with dominant color
    m, n = np.where(x == minimum_num)
    grid_size = np.max(m) - np.min(m) + 1
    centre = int(np.floor(grid_size / 2))
    output = np.full((grid_size, grid_size), maximum_num)

    # Start filling the output grid with other unique colors by making sure the shapes as present in input grid are preserved
    # Four vertices will also be colored in the below logic
    for item in unique_elem:
        if item != maximum_num:
            m, n = np.where(x == item)
            if len(np.unique(m)) > 2 or len(np.unique(n)) > 2:
                # Filling the diamond shape by coloring the center in the four edges
                output[centre, 0] = item
                output[0, centre] = item
                output[centre, grid_size - 1] = item
                output[grid_size - 1, centre] = item
            elif (np.max(m) - np.min(m) == grid_size-1) and len(np.unique(m)) == 2:
                # Filling the rectangle shaped colored cells by measuring the distance from centre position
                y_diff = int((np.max(n) - np.min(n))/2)
                output[0, centre + y_diff] = item
                output[0, centre - y_diff] = item
                output[grid_size - 1, centre + y_diff] = item
                output[grid_size - 1, centre - y_diff] = item
            elif (np.max(n) - np.min(n) == grid_size-1) and len(np.unique(n)) == 2:
                # Filling the rectangle shaped colored cells by measuring the distance from centre position
                x_diff = int((np.max(m) - np.min(m))/2)
                output[centre + x_diff, 0] = item
                output[centre - x_diff, 0] = item
                output[centre + x_diff, grid_size - 1] = item
                output[centre - x_diff, grid_size - 1] = item
    return output


def solve_1b60fb0c(x):
    """
    Input and output grid are of same size
    Input grid will always have only blue color
    Output grid will have blue and red color (we could generalize this output color detection logic but it will require different color inputs)
    Patterns
        1. If top and bottom portion of the grid are identical then left and right of the grid will also be identical
        2. If top and bottom portion of the grid are shifted at regular interval then left and right of the grid will also be shifted at regular interval
        3. These patterns are to be applied on corresponding portion of the grid,
              Like comparing first and last rows, first + 1 and last - 1 rows ...
              Like comparing first and last columns, first + 1 and last - 1 columns ...
    Transformations
        1. Goal is to fill the output grid with red color on the missing blue colored cells which satisfies the above pattern rules
        2. If bottom row of the grid is shifted by 1 along x axis when compared with top row of the grid
           Then left column of the grid is shifted by 1 along y axis when compared with right column of the grid

    Completeness - All available training and test grids for this task are solved correctly
    """
    # Generate the output grid by copying the input grid
    output = np.copy(x)

    # Find the shape and initialize the positional arguments to traverse the grid from all 4 sides
    i, j = output.shape
    bottom = i - 1
    top = 0
    right = j-1
    left = 0

    # Run the loop to fill the missing cells with the required color until the middle point of grid is reached
    while top < i/2:
        if Color.BLUE.value in output[bottom][:]:
            if Color.BLUE.value in output[top][:]:
                existing_values = output[top:bottom, left]
                if (output[bottom, left:right] == output[top, left:right]).all():
                    # Top and bottom rows are identical, copy the contents from right column to left column
                    new_values = output[top:bottom, right]
                else:
                    # Top and bottom rows are shifted by 1 (only possible scenario with the current test set)
                    # copy the contents from right column to left column after shifting by 1
                    new_values = np.roll(output[top:bottom, right], axis=0, shift=1)
                # Replace the missing blue colored cells with red color
                output[top:bottom, left] = np.where(existing_values == new_values, existing_values, Color.RED.value)
            # Skip the top and left alone as current top row does not have blue colored cell in it
            else:
                top = top + 1
                left = left + 1
                continue
        bottom = bottom - 1
        top = top + 1
        left = left + 1
        right = right - 1
    return output


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict cpontaining all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))


if __name__ == "__main__":
    main()

