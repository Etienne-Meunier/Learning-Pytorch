import torch
from torchvision import transforms, datasets


image_size = 224
batch_size = 10
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def load_datas(data_dir='dataset'):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    image_datasets = datasets.ImageFolder(data_dir, data_transforms['train'])

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True,
                                              num_workers=1)

    dataset_sizes = len(image_datasets)

    class_names = image_datasets.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataloaders, dataset_sizes, device, class_names, image_datasets

def unormalize_batch(batch,mean,std):
    '''
    Unormalize batch of data
    :param batch: batch of data to unormilize
    :param mean: mean used for normalization
    :param std: std used for normalization
    :return: batch of data unormalized
    '''
    x = batch.new(*batch.size())
    x[:, 0, :, :] = batch[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = batch[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = batch[:, 2, :, :] * std[2] + mean[2]
    return x