# coding : utf-8
"""
* 3D ESPCN
* This implementaion is based on "Tanno R. et al. (2017) Bayesian Image Quality Transfer with CNNs: Exploring Uncertainty in dMRI Super-Resolution. In: Descoteaux M., Maier-Hein L., Franz A., Jannin P., Collins D., Duchesne S. (eds) Medical Image Computing and Computer Assisted Intervention − MICCAI 2017. Lecture Notes in Computer Science, vol 10433. Springer, Cham"
* Note that this is not official implementaion and I only implement 3D ESPCN!
* @auther tozawa
* @date 20181002
"""
import os, sys, time
import numpy as np
import argparse, yaml, shutil
import chainer
from chainer import training
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))

from model import ESPCN
from updater import EspcnUpdater
from dataset import EspcnDataset
from evaluator import EspcnEvaluator
import util.yaml_utils as yaml_utils

def main():
    parser = argparse.ArgumentParser(description='Training 3D-ESPCN')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/base.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/training',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data')
    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(config.batchsize))
    print('# iteration: {}'.format(config.iteration))
    print('Learning Rate: {}'.format(config.adam['alpha']))
    print('')

    #load the dataset
    print ("load the dataset")
    train = EspcnDataset(args.root,
                        os.path.join(args.base, config.dataset['training_fn']),
                        config.patch['patchside'], config.upsampling_rate)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=config.batchsize)

    val = EspcnDataset(args.root,
                        os.path.join(args.base, config.dataset['val_fn']),
                        config.patch['patchside'], config.upsampling_rate)
    val_iter = chainer.iterators.SerialIterator(val, batch_size=config.batchsize, repeat=False, shuffle=False)

    # Set up a neural network to train
    print ('Set up a neural network to train')
    gen = ESPCN(r=config.upsampling_rate)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()

    #Set up an optimizer
    def make_optimizer(model, alpha=0.00001, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer

    gen_opt = make_optimizer(model = gen,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])

    #set up a trainer
    updater = EspcnUpdater(
        model = (gen),
        iterator = train_iter,
        optimizer = {'gen': gen_opt},
        device=args.gpu
        )

    def create_result_dir(base_dir, output_dir, config_path, config):
        """https://github.com/pfnet-research/sngan_projection/blob/master/train.py"""
        result_dir = os.path.join(base_dir, output_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        def copy_to_result_dir(fn, result_dir):
            bfn = os.path.basename(fn)
            shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

        copy_to_result_dir(
            os.path.join(base_dir, config_path), result_dir)

        copy_to_result_dir(
            os.path.join(base_dir, config.network['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.updater['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.dataset['training_fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.dataset['val_fn']), result_dir)

    create_result_dir(args.base,  args.out, args.config_path, config)

    trainer = training.Trainer(updater,
                            (config.iteration, "iteration"),
                            out=os.path.join(args.base, args.out))

    # Set up logging
    snapshot_interval = (config.snapshot_interval, 'iteration')
    display_interval = (config.display_interval, 'iteration')
    evaluation_interval = (config.evaluation_interval, 'iteration')
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(gen, filename='gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=display_interval))
    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # Evaluate the model with the test dataset for each epoch
    trainer.extend(EspcnEvaluator(val_iter, gen, device=args.gpu), trigger=evaluation_interval)

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['loss', 'val/loss'], 'iteration', file_name='gen_loss.png', trigger=display_interval))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
