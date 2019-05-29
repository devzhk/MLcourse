import torch
from torch.utils.data import DataLoader
from Dataset import faceData
from torchvision.transforms import transforms
from models import vgg13_bn
import torch.nn as nn
import numpy as np


epoch_num = 30
batchsize = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mode = 'convert'

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x)
    return x


if __name__ == "__main__":
    train_set = faceData(mode='train', transform=data_tf)
    test_set = faceData(mode='test', transform=data_tf)
    train_data = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    test_data = DataLoader(test_set, batch_size=batchsize, shuffle=True)

    model = vgg13_bn()
    optim = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.to(device)
        print('device:{}'.format(device))
    if mode == 'train':
        for e in range(epoch_num):
            model.train()
            train_loss = 0
            train_acc = 0
            for img, label in train_data:
                img, label = img.to(device), label.to(device)
                out = model(img)
                loss = criterion(out, label)
                optim.zero_grad()
                loss.backward()
                optim.step()

                train_loss += loss.item()
                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / img.shape[0]
                train_acc += acc

            eval_loss = 0
            eval_acc = 0
            model.eval()
            for img, label in test_data:
                img, label = img.to(device), label.to(device)

                output = model(img)
                loss = criterion(output, label)
                eval_loss += loss.item()
                _, pred = output.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / img.shape[0]
                eval_acc += acc
            etrain_loss = train_loss / len(train_data)
            etrain_acc = train_acc / len(train_data)
            eeval_loss = eval_loss / len(test_data)
            eeval_acc = eval_acc / len(test_data)
            print('epoch:{}, Train Loss:{:.6f}, Train Acc:{:.6f}, Eval Loss: {:.6f}, Eval Acc:{:.6f}'
                .format(e, etrain_loss, etrain_acc, eeval_loss,
                        eeval_acc))
    model.load_state_dict(torch.load('vgg13.pth'))
    
    print('loading vgg13.pth')
    torch.save(model, 'cpuvgg13.pth')

    print('Finish')
    torch.save(model.state_dict(), 'vgg13.pth')
    from mmdnn.conversion.pytorch.pytorch_parser import PytorchParser
    parser = PytorchParser('model_vgg13.pth', [1, 48, 48])
    IR_file = 'vgg13bn'
    parser.run(IR_file)




