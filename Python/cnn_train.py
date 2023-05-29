import cnn_prepare
from blob_generator import generate_samples
from blob_generator import CustomImageDataset
from cnn_prepare import Net
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

import torch.nn.functional as F



train = generate_samples(100)
validation = generate_samples(50)
test = generate_samples(500)

training_set = CustomImageDataset(train)
validation_set = CustomImageDataset(validation)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False)

net = Net()
batch = []
for x in training_loader:
    batch = x

net.forward(batch[0])

learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

def predict(im):
    return net(torch.from_numpy(np.reshape(im.astype(np.single), [1, 1, im.shape[0], im.shape[1]])))

def evaluate(test):
    good = 0
    for sample in test:
        predicted_class = 0
        predicted_vector = predict(sample[0])
        if predicted_vector[0][1] > predicted_vector[0][0]:
            predicted_class = 1
        if predicted_class == sample[1][0][0]:
            good += 1
    return good/test.shape[0]

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    net.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    net.train(False)
    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = net(vinputs)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(net.state_dict(), model_path)

    epoch_number += 1

net.eval()

print(evaluate(test))
