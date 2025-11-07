from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.data import default_collate
from torchvision.transforms import v2

#################### Cutmix or MixUp data augmentation ################################

cutmix = v2.CutMix(num_classes=6)
mixup = v2.MixUp(num_classes=6)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

class CDDCustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        image = self.transform(np.array(image))
        return image, label
    def __len__(self):
        return len(self.labels)

###################################################################

train_dataset = CDDCustomDataset(trainarray_CDD, trainlabel_CDD,transform=data_transform)
test_dataset = CDDCustomDataset(testarray_CDD, testlabel_CDD, transform=data_transform_test)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn = collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
