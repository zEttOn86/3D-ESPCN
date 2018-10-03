import os, sys, time
import argparse, yaml, shutil, math
import numpy as np
import chainer
import SimpleITK as sitk

from model import ESPCN
import util.yaml_utils as yaml_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/base.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/inference',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data(snapshot)')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')
    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))
    print('GPU: {}'.format(args.gpu))
    print('')

    hr_patchside = config.patch['patchside']
    config.patch['patchside'] = int(config.patch['patchside']/config.upsampling_rate)

    gen = ESPCN(r=config.upsampling_rate)
    chainer.serializers.load_npz(args.model, gen)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
    xp = gen.xp

     # Read test list
    path_pairs = []
    with open(os.path.join(args.base, config.dataset['test_fn'])) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line : continue
            path_pairs.append(line[:])

    for i in path_pairs:
        print('   LR from: {}'.format(i[0]))
        print('   HR from: {}'.format(i[1]))
        sitkLR = sitk.ReadImage(os.path.join(args.root, i[0]))
        lr = sitk.GetArrayFromImage(sitkLR).astype("float32")

        # Calculate maximum of number of patch at each side
        ze,ye,xe = lr.shape
        xm = int(math.ceil((float(xe)/float(config.patch['patchside']))))
        ym = int(math.ceil((float(ye)/float(config.patch['patchside']))))
        zm = int(math.ceil((float(ze)/float(config.patch['patchside']))))

        margin = ((0, config.patch['patchside']),
                  (0, config.patch['patchside']),
                  (0, config.patch['patchside']))
        lr = np.pad(lr, margin, 'edge')
        lr = chainer.Variable(xp.array(lr[np.newaxis, np.newaxis, :], dtype=xp.float32))

        zh,yh,xh = ze*config.upsampling_rate, ye*config.upsampling_rate, xe*config.upsampling_rate
        hr_map = np.zeros((zh+hr_patchside,yh+hr_patchside, xh+hr_patchside))

        # Patch loop
        for s in range(xm*ym*zm):
            xi = int(s%xm)*config.patch['patchside']
            yi = int((s%(ym*xm))/xm)*config.patch['patchside']
            zi = int(s/(ym*xm))*config.patch['patchside']

            # Extract patch from original image
            patch = lr[:,:,zi:zi+config.patch['patchside'],yi:yi+config.patch['patchside'],xi:xi+config.patch['patchside']]
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                hr_patch = gen(patch)

             # Generate HR map
            hr_patch = hr_patch.data
            if args.gpu >= 0:
                hr_patch = chainer.cuda.to_cpu(hr_patch)
            zi,yi,xi = zi*config.upsampling_rate, yi*config.upsampling_rate, xi*config.upsampling_rate
            hr_map[zi:zi+hr_patchside,yi:yi+hr_patchside,xi:xi+hr_patchside] = hr_patch[0,:,:,:]

        print('Save image')
        hr_map = hr_map[:zh,:yh,:xh]

        # Save HR map
        inferenceHrImage = sitk.GetImageFromArray(hr_map)
        lr_spacing = sitkLR.GetSpacing()
        new_spacing = [i/config.upsampling_rate for i in lr_spacing]
        inferenceHrImage.SetSpacing(new_spacing)
        inferenceHrImage.SetOrigin(sitkLR.GetOrigin())
        result_dir = os.path.join(args.base, args.out)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fn = os.path.splitext(os.path.basename(i[0]))[0]
        sitk.WriteImage(inferenceHrImage, '{}/{}.mhd'.format(result_dir, fn))


if __name__ == '__main__':
    main()
