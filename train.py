import pandas as pd
from sklearn import preprocessing
from torch import optim
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from config import *
from utils import *
from model import *
from tqdm import tqdm


dataset_df = pd.read_csv('dataset/train.csv')
dataset_df = dataset_df[dataset_df['recognized'] == True]
transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),  transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

le = preprocessing.LabelEncoder()
le.fit(dataset_df['word'])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
for train_index, test_index in sss.split(dataset_df, dataset_df['word']):
    X_train, X_test = dataset_df.iloc[train_index], dataset_df.iloc[test_index]

train_dataset = ImageDataset(X_train, le, transform)
test_dataset = ImageDataset(X_test, le, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    test_dataset,
    batch_size=1,
)

model = resnet_model(in_channels=3, num_classes=num_classes).to(device)

opt = optim.Adam(model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()
lossfn = nn.CrossEntropyLoss()
scheduler = StepLR(opt, step_size=1, gamma=gamma)
train_iter = 0
val_iter = 0

least_val_loss = 1e5

for epoch in range(50):
    print(f"###################### EPOCH: {epoch} #######################")

    train_loss = 0
    train_accuracy = 0

    val_loss = 0
    val_accuracy = 0

    train_len = len(train_loader)
    trainer_loop = tqdm(train_loader, leave=True, total=train_len)
    model.train()
    for images, labels in trainer_loop:
        label = labels[:, 0]
        opt.zero_grad()

        pred_label = model(images)
        loss = lossfn(pred_label, label)

        loss.backward()
        opt.step()

        pred_softmax = F.softmax(pred_label, dim = 1)
        accuracy = 100 * (pred_softmax.argmax(dim = 1) == label).float().mean().cpu().detach().numpy()
        loss_value = loss.item()
        train_loss += loss_value
        train_accuracy += accuracy

        trainer_loop.set_postfix(
                epoch_loss=loss_value,
                epoch_accuracy = accuracy)
        
    writer.add_scalar('Loss/train', loss_value, train_iter)
    writer.add_scalar('Accuracy/train', accuracy, train_iter)
    train_iter += 1
    
    val_len = len(val_loader)
    model.eval()
    for images_val, labels_val in val_loader:
        label_val = labels_val[:,0]
        with torch.no_grad():
            pred_label_val = model(images_val)
            loss_val = lossfn(pred_label_val, label_val)

        pred_softmax = F.softmax(pred_label_val, dim = 1)
        accuracy_val = 100 * (pred_softmax.argmax(dim = 1) == label_val).float().mean().cpu().detach().numpy()

        val_loss += loss_val.item()
        val_accuracy += accuracy_val

    writer.add_scalar('Loss/val', val_loss / val_len, val_iter)
    writer.add_scalar('Accuracy/val', val_accuracy / val_len, val_iter)
    val_iter += 1

    if val_loss / val_len < least_val_loss:
        least_val_loss = val_loss / val_len

        torch.save(model, 'best.pt')
    torch.save(model, 'last.pt')
    
    print('Train :', train_accuracy / train_len, train_loss / train_len)
    print('Val :', val_accuracy / val_len, val_loss / val_len)