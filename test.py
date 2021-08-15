import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import shutil
import zipfile
import glob
import os
import time

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = torchvision.models.resnet50(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1024),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.Dropout(0.1),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

model.cuda()
model.load_state_dict(torch.load('/content/gdrive/MyDrive/CCP/catanddog/MODEL_STATE2.pt'))
model.eval()

test_dir = '/content/gdrive/MyDrive/CCP/catanddog/test1'
test_files = [f'{i+1}.jpg' for i in range(12500)] 

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, files, root, mode='test', transform=None):
        self.files = files
        self.root = root
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img = PIL.Image.open(os.path.join(self.root, self.files[index]))
        if self.transform:
            img = self.transform(img)
        return img, self.files[index]

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()
])

test_dataset = CustomDataset(test_files[:12500], test_dir, transform=test_transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

def predict(model, data_loader):
    with torch.no_grad():
        model.eval()
        ret = None
        for img, files in data_loader:
            img = img.to(device)
            pred = model(img)
            if ret is None:
                ret = pred.cpu().numpy()
            else:
                ret = np.vstack([ret, pred.cpu().numpy()])
    return ret
pred = predict(model, test_loader)

sample_pred = pred[:24]
sample_pred[sample_pred >= 0.5] = 1
sample_pred[sample_pred < 0.5] = 0
imgs, files = iter(test_loader).next()
classes = {0:'cat', 1:'dog'}
fig = plt.figure(figsize=(16,24))
for i in range(24):
    a = fig.add_subplot(4,6,i+1)
    a.set_title(classes[sample_pred[i][0]])
    a.axis('off')
    a.imshow(np.transpose(imgs[i].numpy(), (1,2,0)))
plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)

