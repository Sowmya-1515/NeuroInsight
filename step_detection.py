import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- DATASET ----------------

class TumorDataset(Dataset):

    def __init__(self, root):

        self.images = []
        self.labels = []

        for f in os.listdir(root+"/normal"):
            self.images.append(root+"/normal/"+f)
            self.labels.append(0)

        for f in os.listdir(root+"/tumor"):
            self.images.append(root+"/tumor/"+f)
            self.labels.append(1)

        self.transform = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,i):

        img = Image.open(self.images[i]).convert("RGB")
        img = self.transform(img)

        return img, self.labels[i]

# ---------------- LOAD DATA ----------------

dataset = TumorDataset("/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/brats_slices_224")
loader = DataLoader(dataset,batch_size=16,shuffle=True)

# ---------------- MODEL ----------------

model = models.resnet50(weights=None)

model.fc = nn.Linear(2048,2)

model.load_state_dict(torch.load("training_output/best_model.pth"),strict=False)

model = model.to(device)

# freeze backbone
for param in model.parameters():
    param.requires_grad=False

for param in model.fc.parameters():
    param.requires_grad=True

# ---------------- TRAIN ----------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(),lr=1e-4)

EPOCHS=5

for epoch in range(EPOCHS):

    loss_total=0

    for imgs,labels in loader:

        imgs=imgs.to(device)
        labels=torch.tensor(labels).to(device)

        outputs=model(imgs)

        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total+=loss.item()

    print("Epoch",epoch+1,"Loss:",loss_total)

torch.save(model.state_dict(),"tumor_detector.pth")

print("Training complete")