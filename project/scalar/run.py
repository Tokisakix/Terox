from typing import List
from tqdm import tqdm

from terox.autodiff import Scalar
from terox.module import Module

from model import ScalarIrisClassifyModel, SGD
from dataset import getDataSet
from function import MSELoss

N      = 1000
EPOCHS = 100
LR     = 5e-1

def test(model:Module, dataset:List) -> float:
    Acc = 0
    for inputs, labels in tqdm(dataset):
        inputs = [Scalar(num) for num in inputs]
        outpus = model(inputs)
        Acc += 1 if int(outpus[0].item() + 0.5) == labels else 0
    acc = Acc / len(dataset)
    return acc

if __name__ == "__main__":
    dataset   = getDataSet(N)
    model     = ScalarIrisClassifyModel(in_feature=2, hidden_feature=16, out_feature=1)
    criterion = MSELoss
    optimizer = SGD(model.parmeters(), LR)

    for epoch in range(EPOCHS):
        Loss = 0.0
        for inputs, labels in tqdm(dataset):
            inputs = [Scalar(num) for num in inputs]
            labels = [Scalar(labels)]
            outpus = model(inputs)
            optimizer.zero_grad()
            loss   = criterion(outpus, labels)
            loss.backward()
            Loss += loss.item()
            optimizer.step()
        Loss /= len(dataset)
        Acc = test(model, dataset)
        print(f"[INFO] Epoch:{epoch} Loss:{Loss:.6f} Acc:{Acc * 100:.2f}%")
    
    print("Finished training!")