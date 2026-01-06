import torch

from torch.optim.lr_scheduler import LambdaLR

def get_customized_schedule_with_warmup(optimizer, num_warmup_steps=4000, d_model=64, last_epoch=-1):
    def lr_lambda(current_step):
        current_step += 1
        return (d_model ** -0.5) * min(current_step ** -0.5,
                                       current_step * (num_warmup_steps ** -1.5))
    lr_lambda2 = lambda epoch: 0.0004 * (0.98 ** ((epoch-1)//2))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, d_model=64, k_list=[0.2, 4e-4], warmup_steps=4000, verbose=False):
        self.optimizer = optimizer
        self.k_list = k_list
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.lr = 0.0
        self.verbose = verbose
        # self.epoch = 0
        # self.visdom_lr = None

    # def zero_grad(self):
    #     self.optimizer.zero_grad()

    def step(self, epoch):
        # self._update_lr(epoch)
        # # self._visdom()
        # self.optimizer.step()

    # def _update_lr(self, epoch):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.k_list[0] * self.init_lr * min(self.step_num ** (-0.5),
                                             self.step_num * (self.warmup_steps ** (-1.5)))
        else:
            lr = self.k_list[1] * (0.98 ** ((epoch-1)//2))
        
        if self.lr != lr:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr                
                if self.verbose:
                    print('Adjusting learning rate of group {} to {:.4e}.'.format(i, lr))

        self.lr = lr
        
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    # def set_k(self, k):
    #     self.k = k

    # def set_visdom(self, visdom_lr, vis):
    #     self.visdom_lr = visdom_lr  # Turn on/off visdom of learning rate
    #     self.vis = vis  # visdom enviroment
    #     self.vis_opts = dict(title='Learning Rate',
    #                          ylabel='Leanring Rate', xlabel='step')
    #     self.vis_window = None
    #     self.x_axis = torch.LongTensor()
    #     self.y_axis = torch.FloatTensor()

    # def _visdom(self):
    #     if self.visdom_lr is not None:
    #         self.x_axis = torch.cat(
    #             [self.x_axis, torch.LongTensor([self.step_num])])
    #         self.y_axis = torch.cat(
    #             [self.y_axis, torch.FloatTensor([self.optimizer.param_groups[0]['lr']])])
    #         if self.vis_window is None:
    #             self.vis_window = self.vis.line(X=self.x_axis, Y=self.y_axis,
    #                                             opts=self.vis_opts)
    #         else:
    #             self.vis.line(X=self.x_axis, Y=self.y_axis, win=self.vis_window,
    #                           update='replace')