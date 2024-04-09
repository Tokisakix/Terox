from tqdm import tqdm

from terox.autodiff import Scalar

from model import ScalarIrisClassifyModel, GD
from dataset import getIrisDataSet
from function import softmax, CrossEntropyLoss, argmax

EPOCHS = 10
LR     = 1e-3

if __name__ == "__main__":
    dataset   = getIrisDataSet()
    model     = ScalarIrisClassifyModel()
    criterion = CrossEntropyLoss
    optimizer = GD(model.parmeters(), LR)

    for epoch in range(EPOCHS):
        Loss = 0.0
        Acc = 0
        for inputs, labels in tqdm(dataset):
            inputs = [Scalar(num) for num in inputs]
            outpus = model(inputs)
            outpus = softmax(outpus)
            optimizer.zero_grad()
            loss   = criterion(outpus, labels)
            loss.backward()
            Loss += loss.item()
            optimizer.step()

            Acc += 1 if argmax(outpus)[0] == labels else 0
        Loss /= len(dataset)
        Acc /= len(dataset)
        print(f"[INFO] Epoch:{epoch} Loss:{Loss:.6f} Acc:{Acc * 100:.2f}%")
    
    print("Finished training!")