#coding;utf-8
import os, sys, time
import argparse, glob, yaml
import SimpleITK as sitk
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
import dataIO as IO
import yaml_utils

def main():
    parser = argparse.ArgumentParser(description='Generate low resolution images from HR images')
    parser.add_argument('--base', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Base directory path to program files')
    parser.add_argument('--config_path', type=str, default='../../configs/base.yml',
                        help='path to config file')
    parser.add_argument('--input_dir', type=str, default='../../data/interim',
                        help='Input directory')
    parser.add_argument('--output_dir', type=str, default='../../data/processed',
                        help='Output directory')
    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))

    input_dir = os.path.join(args.base, args.input_dir)
    path_list = glob.glob('{}/*.mhd'.format(input_dir))

    result_dir = os.path.join(args.base, args.output_dir)
    os.makedirs('{}/HR'.format(result_dir), exist_ok=True)
    os.makedirs('{}/LR'.format(result_dir), exist_ok=True)
    for i, path in enumerate(path_list):
        hr_img = IO.read_mhd_and_raw(path, False)
        hr_img.SetOrigin([0,0,0])
        hr_size = hr_img.GetSize()
        hr_spacing = hr_img.GetSpacing()
        new_spacing = [i*config['upsampling_rate'] for i in hr_spacing]
        new_size = [int(hr_size[0]*(hr_spacing[0]/new_spacing[0])+0.5),
                    int(hr_size[1]*(hr_spacing[1]/new_spacing[1])+0.5),
                    int(hr_size[2]*(hr_spacing[2]/new_spacing[2])+0.5)]
        resampleFilter = sitk.ResampleImageFilter()

        lr_img = resampleFilter.Execute(hr_img,
                                        new_size,
                                        sitk.Transform(),
                                        sitk.sitkBSpline,
                                        hr_img.GetOrigin(),
                                        new_spacing,
                                        hr_img.GetDirection(), 0, hr_img.GetPixelID())

        # Save HR and LR images
        sitk.WriteImage(lr_img, '{}/LR/{:04d}.mhd'.format(result_dir, i))
        sitk.WriteImage(hr_img, '{}/HR/{:04d}.mhd'.format(result_dir, i))


if __name__ == '__main__':
    main()
