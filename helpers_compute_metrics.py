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
    parser.add_argument("--n_pt", type=int, default=40, help="Number of pT bins (log-spaced)")
    parser.add_argument("--n_eta", type=int, default=30, help="Number of eta bins")
    parser.add_argument("--n_phi", type=int, default=30, help="Number of phi bins")
    parser.add_argument("--pt_min", type=float, help="pt min for binning (has to be given as log(pt))")
    parser.add_argument("--pt_max", type=float, help="pt max for binning (has to be given as log(pt))")
    parser.add_argument("--eta_min", default = -0.8, type=float, help="eta min for binning (defaults -0.8)")
    parser.add_argument("--eta_max", default = 0.8, type=float, help="eta max for binning (defaults 0.8)")
    parser.add_argument("--phi_min", default = -0.8, type=float, help="phi min for binning (defaults -0.8)")
    parser.add_argument("--phi_max", default = 0.8, type=float, help="phi max for binning (defaults 0.8)")
    
    args = parser.parse_args()
    return args