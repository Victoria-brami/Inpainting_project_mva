
import os
import imghdr
import torch.utils.data as data
from PIL import Image
import random


class ComparisonImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, recursive_search=True, image_type='cn'):
        super(ComparisonImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.imgpaths = self.__load_imgpaths_from_dir(self.data_dir, walk=recursive_search, image_type=image_type)

        super().__init__()
        self._original_train = None


    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index, color_format='RGB'):
        img = Image.open(self.imgpaths[index])
        img = img.convert(color_format)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and imghdr.what(filepath):
            return True
        return False

    def __load_imgpaths_from_dir(self, dirpath, image_type, walk=True):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, _, files) in os.walk(dirpath):
                for file in files:
                    if image_type == 'mask':
                        if file == 'composite.png':
                            file = os.path.join(root, file)
                            if self.__is_imgfile(file):
                                imgpaths.append(file)
                    else:
                        if image_type !='gt':
                            if image_type == 'no_global_cn' or image_type =='no_local_cn' or image_type=='channel0_cn' or image_type=='channel44_cn':
                                if file.endswith('_{}.jpg'.format(image_type)):
                                    file = os.path.join(root, file)
                                    if self.__is_imgfile(file):
                                        imgpaths.append(file)
                            else:
                                if file.endswith('_{}.jpg'.format(image_type)) and len(file) == 13:
                                    file = os.path.join(root, file)
                                    if self.__is_imgfile(file):
                                        imgpaths.append(file)
                        else:
                            if len(file) == 10:
                                file = os.path.join(root, file)
                                if self.__is_imgfile(file):
                                    imgpaths.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if not self.__is_imgfile(path):
                    continue
                imgpaths.append(path)
        return imgpaths

    def shuffle(self):
        random.shuffle(self.imgpaths)

if __name__ == '__main__':
    cn = ComparisonImageDataset(data_dir='../../preprocessed_celebA', image_type='cn')
    patch = ComparisonImageDataset(data_dir='../../preprocessed_celebA', image_type='patch7')
    print(patch.imgpaths[:5])
    print(cn.imgpaths[:5])
    img = Image.open(patch.imgpaths[0])
    img = img.convert('RGB')
    img.show()
