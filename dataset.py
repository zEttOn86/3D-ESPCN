#coding:utf-8
'''
* @auther tzw
* @date 2018-10-12
'''
import os, sys, time
import numpy as np
import chainer

class EspcnDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, data_list_txt, hr_patch_side, upsampling_rate, augmentation=False):
        print(' Initialize dataset')
        self._root = root
        self._patch_side = patch_side
        self.upsampling_rate = upsampling_rate
        self._augmentation = augmentation
        assert(self._patch_side%2==0)

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
            lr_img = (lr_img[np.newaxis, :] - min(lr_img)) / (max(lr_img)- min(lr_img))

            hr_img = IO.read_mhd_and_raw(os.path.join(self._root, i[1]))
            hr_img = ((hr_img[np.newaxis, :] - min(hr_img)) / (max(hr_img)- min(hr_img))*2)-1
            self._dataset.append((lr_img, hr_img))

        print(' Initilazation done ')

    def __len__(self):
        return (sys.maxsize)

    def transform(self, image):
        # Random right left transform
        if np.random.rand() > 0.5:
            img = img[:, ::-1, ::-1, ::-1]
        img += np.random.uniform(size=img.shape, low=0, high=1./128)
        return img

    def get_example(self, i):
        '''
        return (LR, HR)
        '''
        _, d, h, w = self._dataset[i][0].shape
        x_s = np.random.randint(0, w-self._patch_side)
        x_e = x_s+self._patch_side
        y_s = np.random.randint(0, h-self._patch_side)
        y_e = y_s+self._patch_side
        z_s = np.random.randint(0, d-self._patch_side)
        z_e = z_s+self._patch_side
        if not self._augmentation:
            return self._dataset[i][0][:, z_s:z_e, y_s:y_e, x_s:x_e], self._dataset[i][1][:, z_s:z_e, y_s:y_e, x_s:x_e]

        return self.transform(self._dataset[i][0][:, z_s:z_e, y_s:y_e, x_s:x_e]), self.transform(self._dataset[i][1][:, z_s:z_e, y_s:y_e, x_s:x_e])
