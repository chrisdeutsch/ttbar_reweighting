import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


log = logging.getLogger("ttbar_reweighting.nn")


class ReweightingNet(nn.Module):
    def __init__(self, num_inputs, hidden_layers=[32, 32, 32, 32, 32], leak=0.1):
        super(ReweightingNet, self).__init__()

        # Size of the leak in leaky relu
        self.leak = leak

        # Build the network
        nodes = [num_inputs] + hidden_layers + [1]

        self.layers = []
        for i, j in zip(nodes, nodes[1:]):
            self.layers.append(nn.Linear(i, j))

        # Register layers (needed so that pytorch knows about these layers)
        for i, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(i), layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), self.leak)
        return self.layers[-1](x)


def train(model, loader0, loader1, **kwargs):
    epochs = kwargs.get("epochs", 100)
    lr = kwargs.get("lr", 0.1)
    momentum = kwargs.get("momentum", 0.9)
    lr_schedule_factor = kwargs.get("lr_schedule_factor", 0.95)
    clip_grad_value = kwargs.get("clip_grad_value", None)
    weight_decay = kwargs.get("weight_decay", 0)
    train_monitor = kwargs.get("train_monitor", None)
    test_monitor = kwargs.get("test_monitor", None)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda i: lr_schedule_factor)

    log.info("Starting training...")
    log.info("Epochs: " + str(epochs))
    log.info("Initial LR: " + str(lr))
    log.info("Momentum: " + str(momentum))
    log.info("LR schedule factor: " + str(lr_schedule_factor))
    log.info("Gradient value clipping: " + str(clip_grad_value))
    log.info("Weight decay (L2 reg.): " + str(weight_decay))

    loss_train = []
    loss_test = []

    for epoch in range(epochs):
        # Running average of loss
        running_loss = 0.0
        running_loss_cnt = 0

        model.train()
        for i, (data0, data1) in enumerate(zip(loader0, loader1)):
            if i % 100 == 99:
                log.info("[{} {}]: Avg. loss = {:.5f}".format(epoch + 1, i + 1, running_loss / running_loss_cnt))
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

            loss_term1 = torch.sum(w0 / torch.sqrt(torch.exp(pred0))) / torch.sum(w0)
            loss_term2 = torch.sum(w1 * torch.sqrt(torch.exp(pred1))) / torch.sum(w1)

            if loss_term1 < 0:
                log.warning("Loss term 1 is negative. Clipping at 0...")
                loss_term1 = torch.max(loss_term1, torch.zeros(1))

            if loss_term2 < 0:
                log.warning("Loss term 2 is negative. Clipping at 0...")
                loss_term1 = torch.max(loss_term2, torch.zeros(1))

            loss = loss_term1 + loss_term2
            log.debug("[{} {}]: Loss = {:.5f}".format(epoch + 1, i + 1, loss))

            if loss < 0:
                log.error("Loss is negative")
                import pdb;pdb.set_trace()
                sys.exit(1)

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
                import pdb; pdb.set_trace()
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

        # Monitor loss
        monitor = []
        if train_monitor:
            monitor.append(("Training", train_monitor))
        if test_monitor:
            monitor.append(("Testing", test_monitor))

        for label, (X0, W0, X1, W1) in monitor:
            model.eval()
            with torch.no_grad():
                pred0 = model(X0)
                pred1 = model(X1)
                l = torch.sum(W0 / torch.sqrt(torch.exp(pred0))) / torch.sum(W0) \
                    + torch.sum(W1 * torch.sqrt(torch.exp(pred1))) / torch.sum(W1)

            log.info("{} loss after epoch {}: {:.6f}".format(label, epoch + 1, l))

            if label == "Training":
                loss_train.append(l.item())
            if label == "Testing":
                loss_test.append(l.item())

            model.train()

    return loss_train, loss_test
