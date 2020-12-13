#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

def solve_f76d97a5(x):
    output = np.copy(x)
    unique_elem = np.unique(output)
    for item in unique_elem:
        if item != 5:
            output = np.where(output == item, 0, output)
            output = np.where(output == 5, item, output)
    return output

def solve_c8cbb738(x):
    unique_elem = np.unique(x)
    occurences_count = [(elem, (x == elem).sum()) for elem in unique_elem]
    maximum_num, max_count = sorted(occurences_count, key=lambda tup: tup[1], reverse=True)[0]
    minimum_num, min_count = sorted(occurences_count, key=lambda tup: tup[1])[0]
    m, n = np.where(x == minimum_num)
    grid_size = np.max(m) - np.min(m) + 1
    centre = int(np.floor(grid_size / 2))
    output = np.full((grid_size, grid_size), maximum_num)
    for item in unique_elem:
        if item != maximum_num:
            m, n = np.where(x == item)
            if len(np.unique(m)) > 2 or len(np.unique(n)) > 2:
                output[centre, 0] = item
                output[0, centre] = item
                output[centre, grid_size - 1] = item
                output[grid_size - 1, centre] = item
            elif (np.max(m) - np.min(m) == grid_size-1) and len(np.unique(m)) == 2:
                y_diff = int((np.max(n) - np.min(n))/2)
                output[0, centre + y_diff] = item
                output[0, centre - y_diff] = item
                output[grid_size - 1, centre + y_diff] = item
                output[grid_size - 1, centre - y_diff] = item
            elif (np.max(n) - np.min(n) == grid_size-1) and len(np.unique(n)) == 2:
                x_diff = int((np.max(m) - np.min(m))/2)
                output[centre + x_diff, 0] = item
                output[centre - x_diff, 0] = item
                output[centre + x_diff, grid_size - 1] = item
                output[centre - x_diff, grid_size - 1] = item
    return output

def solve_1b60fb0c(y):
    x = np.copy(y)
    i, j = x.shape
    bottom = i - 1
    top = 0
    right = j-1
    left = 0
    while (top < i/2):
        if 1 in x[bottom][:]:
           if 1 in x[top][:]:
               existing_values = x[top:bottom, left]
               if (x[bottom, left:right] == x[top, left:right]).all():
                   new_values = x[top:bottom, right]
               else:
                    new_values = np.roll(x[top:bottom, right], axis=0, shift=1)
               x[top:bottom, left] = np.where(existing_values == new_values, existing_values, 2)
           else:
                top = top + 1
                left = left + 1
                continue
        bottom = bottom - 1
        top = top + 1
        left = left + 1
        right = right - 1
    return x


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

if __name__ == "__main__": main()

