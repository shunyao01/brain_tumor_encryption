import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from torchattacks import PGD, SINIFGSM, VNIFGSM, Jitter, APGD
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def images_transforms(size):
    data_transformation = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return data_transformation


def get_data_loader(path, b, data_transform):
    dataset = datasets.ImageFolder(path, transform=data_transform)
    return DataLoader(dataset, batch_size=b)


def get_classifier_model(num_classes):
    m = torchvision.models.mobilenet_v3_large(pretrained=True)
    num_features = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_features=num_features, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    )

    return m


def get_attack(name, c_model):
    if name == "SINIFGSM":
        return SINIFGSM(c_model, eps=0.03137254901960784, alpha=0.00784313725490196, steps=10, decay=1.0, m=5)
    elif name == "PGD":
        return PGD(c_model, eps=0.03137254901960784, alpha=0.00784313725490196, steps=10, random_start=True)
    elif name == "VNIFGSM":
        return VNIFGSM(c_model, eps=0.03137254901960784, alpha=0.00784313725490196, steps=10, decay=1.0, N=5, beta=1.5)
    elif name == "Jitter":
        return Jitter(c_model, eps=0.03137254901960784, alpha=0.00784313725490196, steps=10, scale=10, std=0.1,
                      random_start=True)
    elif name == "APGD":
        return APGD(c_model, norm='Linf', eps=0.03137254901960784, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1,
                    rho=0.75, verbose=False)
    else:
        # print("Attack Model is not valid")
        pass


def atk_save(attack, attack_name, loader, path):
    print(f"Performing {attack_name} attack...")

    # Initialize counter for each class
    class_count = [0, 0, 0, 0]

    # Define class name
    inverse_class_map = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    class_name = ['Te-gl_', 'Te-me_', 'Te-no_', 'Te-pi_']

    # For every batch
    for j, (images, labels) in enumerate(loader):

        adv_images = attack(images, labels)

        # For every image in the batch
        for i in range(len(images)):
            image, label = adv_images[i], labels[i].item()

            # Define save path
            output_dir = os.path.join(path, inverse_class_map[label])
            # If directory does not exist, create it
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = output_dir + "/" + class_name[label] + str(class_count[label]).zfill(4) + ".jpg"

            class_count[label] += 1

            # rescale image from [0..1] to [0..255] and transpose to [H, W, C]
            image = np.array(np.transpose(image.detach().cpu() * 255.0, (1, 2, 0)), dtype=np.uint8)

            # save image
            im = Image.fromarray(image).convert('RGB')
            im.save(output_file)

    print(f"Done saving images for {attack_name} attack")
    print(f"Class count: {class_count}\n")


if __name__ == '__main__':
    atk_name = sys.argv[1]
    load_path = sys.argv[2]
    save_path = sys.argv[3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_SIZE = (512, 512)
    batch_size = 10
    n_classes = 4

    # get classifier model
    model = get_classifier_model(n_classes).to(device)
    model.load_state_dict(torch.load('./models/best_model_512.pt'))

    # get original dataset in loader
    data_loader = get_data_loader(load_path, batch_size, images_transforms(IMAGE_SIZE))

    # define attack model
    atk = get_attack(atk_name, model)

    # perturb image and save into save_path/atk_name
    atk_save(atk, atk_name, data_loader, save_path)
