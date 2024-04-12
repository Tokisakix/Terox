from tqdm import tqdm

from terox.autodiff import Scalar

from model import ScalarIrisClassifyModel, GD
from dataset import getDataSet
from function import MSELoss, argmax

N      = 10
EPOCHS = 100
LR     = 0.5

if __name__ == "__main__":
    dataset   = getDataSet(N)
    model     = ScalarIrisClassifyModel(in_feature=2, hidden_feature=128, out_feature=4)
    criterion = MSELoss
    optimizer = GD(model.parmeters(), LR)

    for epoch in range(EPOCHS):
        Loss = 0.0
        Acc = 0
        for inputs, labels in tqdm(dataset):
            inputs = [Scalar(num) for num in inputs]
            outpus = model(inputs)
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