import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


log = logging.getLogger("ttbar_reweighting.nn")


class ReweightingNet(nn.Module):
    def __init__(self, num_inputs, leak=0.1):
        super(ReweightingNet, self).__init__()

        # Size of the leak in leaky relu
        self.leak = leak

        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), self.leak)
        x = F.leaky_relu(self.fc2(x), self.leak)
        x = F.leaky_relu(self.fc3(x), self.leak)
        x = F.leaky_relu(self.fc4(x), self.leak)
        x = F.leaky_relu(self.fc5(x), self.leak)
        return self.fc6(x)


def train(model, loader0, loader1, **kwargs):
    epochs = kwargs.get("epochs", 100)
    lr = kwargs.get("lr", 0.1)
    momentum = kwargs.get("momentum", 0.9)
    lr_schedule_factor = kwargs.get("lr_schedule_factor", 0.95)
    clip_grad_value = kwargs.get("clip_grad_value", None)
    weight_decay = kwargs.get("weight_decay", 0)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda i: lr_schedule_factor)

    log.info("Starting training...")
    log.info("Epochs: " + str(epochs))
    log.info("Initial LR: " + str(lr))
    log.info("Momentum: " + str(momentum))
    log.info("LR schedule factor: " + str(lr_schedule_factor))
    log.info("Gradient value clipping: " + str(clip_grad_value))
    log.info("Weight decay (L2 reg.): " + str(weight_decay))

    for epoch in range(epochs):
        # Running average of loss
        running_loss = 0.0
        running_loss_cnt = 0

        for i, (data0, data1) in enumerate(zip(loader0, loader1)):
            if i % 1000 == 999:
                log.info("[{} {}]: {:.5f}".format(epoch + 1, i + 1, running_loss / running_loss_cnt))
                running_loss = 0.0
                running_loss_cnt = 0

            x0, w0 = data0
            x1, w1 = data1

            if torch.sum(w0) <= 0 or torch.sum(w1) <= 0:
                log.warning("Skipping bad batch...")
                log.warning("sum(w0) = {}".format(torch.sum(w0).item()))
                log.warning("sum(w1) = {}".format(torch.sum(w1).item()))
                log.warning("This should not occur often. If it does it will likely introduce a bias.")
                continue

            optimizer.zero_grad()

            pred0, pred1 = model(x0), model(x1)

            loss = torch.sum(w0 / torch.sqrt(torch.exp(pred0))) / torch.sum(w0) \
                + torch.sum(w1 * torch.sqrt(torch.exp(pred1))) / torch.sum(w1)

            if torch.isnan(loss) or torch.isinf(loss):
                log.error("Loss is nan or inf. Aborting...")
                log.info("pred0")
                log.info(pred0)
                log.info("pred1")
                log.info(pred1)
                log.info("sum w0")
                log.info(torch.sum(w0))
                log.info("sum w1")
                log.info(torch.sum(w1))
                sys.exit(1)

            running_loss += loss.item()
            running_loss_cnt += 1

            loss.backward()
            if clip_grad_value:
                nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
            optimizer.step()

        scheduler.step()

        # Print out learning rate to get a feeling for the scheduler
        params, = optimizer.param_groups
        log.info("Epoch finished. Current LR: {}".format(params["lr"]))
