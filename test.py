import unittest
import warnings

import torchattacks.attacks.sinifgsm

from attack_model import *
from classifier_efficientnetB3 import *

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


class TestImageTransform(unittest.TestCase):

    def test_1080_to_512(self):
        random_image = np.random.rand(1080, 1080)
        im = Image.fromarray(random_image)
        transformed_image = images_transforms((512, 512))(im)
        expected = (512, 512)
        actual = transformed_image[0].shape
        self.assertEqual(expected, actual)

    def test_1080_to_256(self):
        random_image = np.random.rand(1080, 1080)
        im = Image.fromarray(random_image)
        transformed_image = images_transforms((256, 256))(im)
        expected = (256, 256)
        actual = transformed_image[0].shape
        self.assertEqual(expected, actual)

    def test_1080_to_128(self):
        random_image = np.random.rand(1080, 1080)
        im = Image.fromarray(random_image)
        transformed_image = images_transforms((128, 128))(im)
        expected = (128, 128)
        actual = transformed_image[0].shape
        self.assertEqual(expected, actual)

    def test_128_to_1080(self):
        random_image = np.random.rand(128, 128)
        im = Image.fromarray(random_image)
        transformed_image = images_transforms((1080, 1080))(im)
        expected = (1080, 1080)
        actual = transformed_image[0].shape
        self.assertEqual(expected, actual)

    def test_0_to_128(self):
        random_image = np.random.rand(0, 0)
        self.assertRaises(ValueError, Image.fromarray, random_image)


class TestLoadData(unittest.TestCase):

    def test_data_loader_type(self):
        load_path_, batch_size_ = "./Dataset/brain_tumor_dataset/Testing", 10
        data_loader_ = get_data_loader(load_path_, batch_size_, images_transforms((512, 512)))
        expected = torch.utils.data.dataloader.DataLoader
        actual = type(data_loader_)
        self.assertEqual(expected, actual)

    def test_load_testing(self):
        load_path_, batch_size_ = "./Dataset/brain_tumor_dataset/Testing", 10
        data_loader_ = get_data_loader(load_path_, batch_size_, images_transforms((512, 512)))
        n = 0
        for j, (images, labels) in enumerate(data_loader_):
            n += len(images)
        expected = 1311
        actual = n
        self.assertEqual(expected, actual)

    def test_load_training(self):
        load_path_, batch_size_ = "./Dataset/brain_tumor_dataset/Training", 10
        data_loader_ = get_data_loader(load_path_, batch_size_, images_transforms((512, 512)))
        n = 0
        for j, (images, labels) in enumerate(data_loader_):
            n += len(images)
        expected = 5712
        actual = n
        self.assertEqual(expected, actual)

    def test_load_image_shape(self):
        load_path_, batch_size_, s = "./Dataset/brain_tumor_dataset/Testing", 10, None
        data_loader_ = get_data_loader(load_path_, batch_size_, images_transforms((512, 512)))
        for j, (images, labels) in enumerate(data_loader_):
            s = images[0].shape
            break
        expected = (3, 512, 512)
        actual = s
        self.assertEqual(expected, actual)

    def test_load_saved_image(self):
        load_path_, batch_size_ = "./Dataset/adv_brain_tumor_dataset/SINIFGSM/Testing", 10
        data_loader_ = get_data_loader(load_path_, batch_size_, images_transforms((512, 512)))
        n = 0
        for j, (images, labels) in enumerate(data_loader_):
            n += len(images)
        expected = 1311
        actual = n
        self.assertEqual(expected, actual)

    def test_efficientnetB3_load_data(self):
        x, y = get_data("./Dataset/brain_tumor_dataset/Testing")
        expected = [(1311, 128, 128, 3), (1311,)]
        actual = [x.shape, y.shape]
        self.assertEqual(expected, actual)

    def test_efficientnetB3_load_saved_data(self):
        x, y = get_data("./Dataset/adv_brain_tumor_dataset/SINIFGSM/Testing")
        expected = [(1311, 128, 128, 3), (1311,)]
        actual = [x.shape, y.shape]
        self.assertEqual(expected, actual)


class TestClassifier(unittest.TestCase):

    def test_mobilenetV3(self):
        m = get_classifier_model(4)
        expected = torchvision.models.mobilenetv3.MobileNetV3
        actual = type(m)
        self.assertEqual(expected, actual)

    def test_n_features(self):
        m = get_classifier_model(4)
        expected = len(torchvision.models.mobilenet_v3_large().features)
        actual = len(get_classifier_model(4).features)
        self.assertEqual(expected, actual)

    def test_n_parameters(self):
        m = get_classifier_model(4)

        def count(model_):
            return sum(p.numel() for p in model_.parameters() if p.requires_grad)

        expected = count(m)
        actual = count(get_classifier_model(4))
        self.assertEqual(expected, actual)


class TestAttackModel(unittest.TestCase):

    def test_SINIFGSM(self):
        m = get_classifier_model(4)
        expected = torchattacks.attacks.sinifgsm.SINIFGSM
        actual = type(get_attack("SINIFGSM", m))
        self.assertEqual(expected, actual)

    def test_VNIFGSM(self):
        m = get_classifier_model(4)
        expected = torchattacks.attacks.vnifgsm.VNIFGSM
        actual = type(get_attack("VNIFGSM", m))
        self.assertEqual(expected, actual)

    def test_PGD(self):
        m = get_classifier_model(4)
        expected = torchattacks.attacks.pgd.PGD
        actual = type(get_attack("PGD", m))
        self.assertEqual(expected, actual)

    def test_APGD(self):
        m = get_classifier_model(4)
        expected = torchattacks.attacks.apgd.APGD
        actual = type(get_attack("APGD", m))
        self.assertEqual(expected, actual)

    def test_Jitter(self):
        m = get_classifier_model(4)
        expected = torchattacks.attacks.jitter.Jitter
        actual = type(get_attack("Jitter", m))
        self.assertEqual(expected, actual)

    def test_unknown(self):
        m = get_classifier_model(4)
        expected = None
        actual = get_attack("random", m)
        self.assertEqual(expected, actual)
