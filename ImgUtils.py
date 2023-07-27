import os
import numpy
from matplotlib import pyplot as plt
import PIL.Image as Image


def npy2img():
    arrayData = numpy.load('./datasets/sdu.npy')
    print(arrayData.shape)

    for type_index in range(arrayData.shape[0]):
        plt.figure()

        for img_index in range(arrayData.shape[1]):
            img = arrayData[type_index, img_index]
            plt.subplot(1, arrayData.shape[1], img_index + 1)
            plt.imshow(img)
            plt.axis('off')

        # 保存图片
        plt.savefig(os.path.join('./imgs/output/sdu', f'type_{type_index}.png'), bbox_inches='tight')
        plt.close()


def img2npy():
    folder_path = './imgs/resultPT500_1600/dataTrain'

    data = numpy.empty((5, 40, 64, 64, 1), dtype=numpy.uint8)

    for type_index in [1, 2, 3, 4, 0]:
        subfolder_path = os.path.join(folder_path, str(type_index))
        for img_index in range(40):
            image_path = os.path.join(subfolder_path, f"{img_index}.png")
            img = Image.open(image_path).convert('L')
            data[type_index - 1, img_index, :, :, 0] = numpy.array(img, dtype=numpy.uint8)

    numpy.save(os.path.join('./datasets', 'sdu.npy'), data)


if __name__ == '__main__':
    # img2npy()
    npy2img()
