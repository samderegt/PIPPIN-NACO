import os
import argparse, configparser
from .pippin import run_pipeline, prepare_calib_files

def main(args=None):

    # Current working directory as SCIENCE path
    path_cwd = os.getcwd()

    # Obtain the master FLATs, BPMs, and DARKs from the
    #path_main            = os.path.realpath(__file__)
    #path_master_FLAT_dir = os.path.join(path_main, './data/master_FLAT/')
    #path_master_BPM_dir  = os.path.join(path_main, './data/master_BPM/')
    #path_master_DARK_dir = os.path.join(path_main, './data/master_DARK/')

    #path_FLAT_dir = '/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_FLAT/'
    #path_BPM_dir  = '/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_BPM/'
    #path_DARK_dir = '/home/sam/Documents/Master-2/MRP/PIPPIN-NACO/pippin/data/master_DARK/'

    path_FLAT_dir       = None
    path_master_BPM_dir = None
    path_DARK_dir       = None

    # All arguments to expect
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_example', action='store_true')

    parser.add_argument('--run_pipeline', action='store_true')
    parser.add_argument('--prepare_calib_files', nargs='?', default=False, const=True)

    parser.add_argument('--path_FLAT_dir', default=path_FLAT_dir,
                        type=str, help='Path to the FLAT directory.')
    parser.add_argument('--path_master_BPM_dir', default=path_master_BPM_dir,
                        type=str, help='Path to the master BPM directory.')
    parser.add_argument('--path_DARK_dir', default=path_DARK_dir,
                        type=str, help='Path to the DARK directory.')

    # Read the arguments in the command line
    args = parser.parse_args()

    if args.prepare_calib_files:
        # Create master FLATs, BPMs and DARKs from the provided paths
        path_master_FLAT_dir, path_master_BPM_dir, path_master_DARK_dir \
        = prepare_calib_files(path_FLAT_dir=args.path_FLAT_dir,
                              path_master_BPM_dir=args.path_master_BPM_dir,
                              path_DARK_dir=args.path_DARK_dir
                              )
    else:
        path_master_FLAT_dir, path_master_BPM_dir, path_master_DARK_dir \
        = args.path_FLAT_dir, args.path_master_BPM_dir, args.path_DARK_dir


    if args.run_example:
        # Run the example reduction
        run_example(path_cwd=path_cwd)

    elif args.run_pipeline:
        # Run the pipeline
        run_pipeline(path_SCIENCE_dir=path_cwd,
                     path_master_FLAT_dir=path_master_FLAT_dir,
                     path_master_BPM_dir=path_master_BPM_dir,
                     path_master_DARK_dir=path_master_DARK_dir
                     )

if __name__ == '__main__':
    main()