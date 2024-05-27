import numpy as np
import csv


def check_valid(out, pred_col):
    if pred_col == 0:
        return True
    for i in range(pred_col):
        if not out[i]:
            return False
    return True

def get_all_samples(xs, ys, cls=True):
    if cls:
        ys = [[check_valid(ys[i], pred_col=5)] for i in range(len(ys))]    
    else:
        good_idx = [i for i in range(len(ys)) if check_valid(ys[i], pred_col=5)]
        bad_idx = [i for i in range(len(ys)) if i not in good_idx]
        # print(bad_idx)
        ys = [[ys[i][-1]] for i in good_idx]
        xs = [xs[i] for i in good_idx]
    return xs, ys


def load_dataset(filename, cls=False):
    inputs = []
    outputs = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        idx = 0
        for row in csv_reader:            
            input = []
            input.append(float(row[1]))
            input.append(float(row[2]))
            input.append(float(row[4]))
            input.append(float(row[5]))
            input.append(float(row[7]))
            inputs.append(np.array(input))
            output = [row[10] == 'yes', 'FCC' in row[11], row[12] == 'yes', row[13] == 'yes', row[14] == 'yes']
            try:
                data1 = float(row[16])
                data2 = float(row[17])
            except:
                data1 = 0
                data2 = 0
            output.append(data1)
            output.append(data2)
            outputs.append(output)
    all_dataset = get_all_samples(inputs, outputs, cls=cls)
    return all_dataset