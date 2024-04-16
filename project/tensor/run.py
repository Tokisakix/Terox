from typing import List
from tqdm import tqdm

from terox.tensor import Tensor
from terox.module import Module

from model import ScalarIrisClassifyModel, SGD
from dataset import getDataSet
from function import MSELoss

N      = 10000
EPOCHS = 100
LR     = 1e-3

def test(model:Module, dataset:List) -> float:
    Acc = 0
    for inputs, labels in tqdm(dataset):
        inputs = Tensor(inputs)
        outpus = model(inputs)
        Acc += 1 if int(outpus.item() + 0.5) == labels else 0
    acc = Acc / len(dataset)
    return acc

if __name__ == "__main__":
    dataset   = getDataSet(N)
    model     = ScalarIrisClassifyModel(in_feature=2, hidden_feature=128, out_feature=1)
    criterion = MSELoss
    optimizer = SGD(model.parmeters(), LR)
    print(f"[INFO] Start Acc:{test(model, dataset) * 100:.2f}%")

    for epoch in range(EPOCHS):
        Loss = 0.0
        for inputs, labels in tqdm(dataset):
            inputs = Tensor([inputs])
            labels = Tensor(labels)
            outpus = model(inputs)
            optimizer.zero_grad()
            loss   = criterion(outpus, labels)
            loss.backward()
            Loss += loss.item()[0, 0]
            optimizer.step()
        Loss = Loss / (len(dataset))
        Acc = test(model, dataset)
        print(f"[INFO] Epoch:{epoch} Loss:{Loss:.6f} Acc:{Acc * 100:.2f}%")
    
    print("Finished training!")