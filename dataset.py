#coding:utf-8
'''
* @auther tzw
* @date 2018-10-12
'''
import os, sys, time
import numpy as np
import chainer
import util.dataIO as IO

class EspcnDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, data_list_txt, hr_patch_side, upsampling_rate, augmentation=False):
        print(' Initialize dataset')
        self._root = root
        self._patch_side = hr_patch_side
        self.upsampling_rate = upsampling_rate
        self._augmentation = augmentation
        self._max_batchsize = 500
        assert(self._patch_side%2==0 and self._patch_side%self.upsampling_rate==0)

        self._lr_patch_side = self._patch_side // self.upsampling_rate
        """
        * Read path to org and label data
        hogehoge.txt
        LR.mhd HR.mhd
        """
        path_pairs = []
        with open(data_list_txt) as paths_file:
            for line in paths_file:
                line = line.split()
                if not line : continue
                path_pairs.append(line[:])

        self._num_of_case = len(path_pairs)
        print('    # of cases: {}'.format(self._num_of_case))

        self._dataset=[]
        for i in path_pairs:
            print('   LR from: {}'.format(i[0]))
            print('   HR from: {}'.format(i[1]))
            # Read data
            lr_img = IO.read_mhd_and_raw(os.path.join(self._root, i[0])).astype("float32")
            #(z, y, x) -> (ch, z, y, x) and norm [0, 1]
            lr_img = (lr_img[np.newaxis, :] - np.amin(lr_img)) / (np.amax(lr_img)- np.amin(lr_img))

            hr_img = IO.read_mhd_and_raw(os.path.join(self._root, i[1])).astype("float32")
            hr_img = ((hr_img[np.newaxis, :] - np.amin(hr_img)) / (np.amax(hr_img)- np.amin(hr_img))*2)-1
            self._dataset.append((lr_img, hr_img))

        print(' Initilazation done ')

    def __len__(self):
        return self._max_batchsize

    def transform(self, img):
        # Random right left transform
        if np.random.rand() > 0.5:
            img = img[:, ::-1, ::-1, ::-1]
        #img += np.random.uniform(size=img.shape, low=0, high=1./128)
        return img

    def get_example(self, i):
        '''
        return (LR, HR)
        '''
        pos = np.random.randint(0, len(self._dataset))
        # Get HR img size
        _, d, h, w = self._dataset[pos][1].shape
        x_s = np.random.randint(0, w-self._patch_side)
        x_e = x_s+self._patch_side
        y_s = np.random.randint(0, h-self._patch_side)
        y_e = y_s+self._patch_side
        z_s = np.random.randint(0, d-self._patch_side)
        z_e = z_s+self._patch_side

        # Nearest Neaighbor
        lr_xs = int(x_s/self.upsampling_rate+0.5)
        lr_xe = lr_xs+self._lr_patch_side
        lr_ys = int(y_s/self.upsampling_rate+0.5)
        lr_ye = lr_ys+self._lr_patch_side
        lr_zs = int(z_s/self.upsampling_rate+0.5)
        lr_ze = lr_zs+self._lr_patch_side

        if not self._augmentation:
            return self._dataset[pos][0][:, lr_zs:lr_ze, lr_ys:lr_ye, lr_xs:lr_xe], self._dataset[pos][1][:, z_s:z_e, y_s:y_e, x_s:x_e]

        return self.transform(self._dataset[pos][0][:, lr_zs:lr_ze, lr_ys:lr_ye, lr_xs:lr_xe]), self.transform(self._dataset[pos][1][:, z_s:z_e, y_s:y_e, x_s:x_e])
