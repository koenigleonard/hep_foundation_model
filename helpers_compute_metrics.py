from argparse import ArgumentParser
import os

def parse_inputs():

    parser = ArgumentParser()
    #### add arguments here
    parser.add_argument("--data_path", type = str, help = "Path to jet data set of which the metrics should be computed ")
    parser.add_argument("--num_const", type = int, default = 100, help = "Number of constituents taken from dataset")
    parser.add_argument("--batch_size", type = int, default = 100, help = "Number of jets used in one computation step")
    parser.add_argument("--output_file", type = str, default = "output/plot_data/metrics.csv", help = "file name of the output csv file")
    parser.add_argument("--input_key", type = str, default = "sampled_jets", help = "if the key of table in the h5 is different it can be specified here")

    args = parser.parse_args()
    return args