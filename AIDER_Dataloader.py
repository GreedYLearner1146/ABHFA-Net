from torch.utils.data import Dataset

class AIDERCustomDataset(Dataset):
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

train_dataset = AIDERCustomDataset(new_X, new_y,transform=data_transform)
test_dataset = AIDERCustomDataset(new_X_test, new_y_test, transform=data_transform_test)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
