#coding;utf-8
"""
* Extract data that are only used in this program
"""
import os, sys, time
import argparse, glob

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
import dataIO as IO


def main():
    parser = argparse.ArgumentParser(description='Copy data')
    parser.add_argument('--base', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Base directory path to program files')
    parser.add_argument('--input_dir', type=str, default='../../data/raw',
                        help='Input directory')
    parser.add_argument('--output_dir', type=str, default='../../data/interim',
                        help='Output directory')
    args = parser.parse_args()

    input_dir = os.path.join(args.base, args.input_dir)
    path_list = [os.path.abspath(i) for i in glob.glob("{}/**/*T1*.gz".format(input_dir), recursive=True)]

    result_dir = os.path.join(args.base, args.output_dir)
    os.makedirs(result_dir, exist_ok=True)

    for i, path in enumerate(path_list):
        img = IO.read_mhd_and_raw(path, False)
        #print('{}/{:04d}.mhd'.format(result_dir, i))
        IO.write_mhd_and_raw(img, '{}/{:04d}.mhd'.format(result_dir, i))



if __name__ == '__main__':
    main()
