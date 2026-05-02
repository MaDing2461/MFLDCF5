import functools
import torch
import torch.nn as nn
import torch.optim as optim
# from NPR_DeepfakeDetection.resnet import resnet50
# from NPR_DeepfakeDetection.base_model import BaseModel, init_weights
from .resnet import resnet50
from .resnet import ResNet, Bottleneck
# ResNet
from .base_model import BaseModel, init_weights

# class Trainer1(BaseModel):
class Trainer1(ResNet):
    # def name(self):
    #     return 'Trainer'

    # def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        # super(ResNet, self).__init__()
    def __init__(self, args, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=2, zero_init_residual=False):
        super(Trainer1, self).__init__(block=block, layers=layers, num_classes=num_classes, zero_init_residual=zero_init_residual)

        # if self.isTrain and not opt.continue_train:
        #     self.model = resnet50(pretrained=False, num_classes=1)
        # self.model = resnet50(pretrained=False, num_classes=2)
        # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        print(2222222222222222)

        # if not self.isTrain or opt.continue_train:
        #     self.model = resnet50(num_classes=1)

        # if self.isTrain:
        #     # self.loss_fn = nn.BCEWithLogitsLoss()
        #     # initialize optimizers
        #     if args.optimizer == 'ADAM':
        #         self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                                           lr=args.lr, betas=args.betas)
            # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
            #                                      lr=opt.lr, betas=(0.9, 0.999))

            # elif opt.optim == 'sgd':
            #     self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
            #                                      lr=opt.lr, momentum=0.0, weight_decay=0)
            # else:
            #     raise ValueError("optim should be [adam, sgd]")

        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.epoch)
        # self.model.to(0)
 

    # def adjust_learning_rate(self, min_lr=1e-6):
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] *= 0.9
    #         if param_group['lr'] < min_lr:
    #             return False
    #     self.lr = param_group['lr']
    #     print('*'*25)
    #     print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
    #     print('*'*25)
    #     return True

    # def set_input(self, input):
    #     self.input = input[0].to(self.device)
    #     self.label = input[1].to(self.device).float()


    # def forward(self, x):
    #     # self.output = self.model(x)
    #     # out = self.output
    #     # print(11111111111111111111111111111)
    #     # self.loss = self.loss_fn(self.output.squeeze(1), self.label)
    #     # self.optimizer.zero_grad()
    #     # self.loss.backward()
    #     # self.optimizer.step()
    #     # return out
    #     self.output = self.model(self.input)

    # def get_loss(self):
    #     return self.loss_fn(self.output.squeeze(1), self.label)

    # def optimize_parameters(self):
    #     self.forward()
    #     self.loss = self.loss_fn(self.output.squeeze(1), self.label)
    #     self.optimizer.zero_grad()
    #     self.loss.backward()
    #     self.optimizer.step()