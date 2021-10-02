import math

__all__ = ['LR_Scheduler_Head']

class LR_Schedule(object):
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=2, quiet=False):
        self.mode = mode
        self.quiet = quiet
        if not quiet:
            print('Using {} LR schedule with warm-up epochs of {}!'.format(self.mode, warmup_epochs))
        if mode == 'step':
            assert lr_step
        self.base_lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_epochs = warmup_epochs
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch
        # ======================= cos beta ===========================
        self.beta = 1

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == 'cos':
            T = T - self.warmup_iters
            # lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
            # ============================ change =================================
            real_epoch = epoch+1-self.warmup_epochs
            if real_epoch>0:
                if real_epoch%10 == 0:
                    self.beta = self.beta*0.3

            ratio = ((epoch+1)%7)/7
            # 7 epoch
            lr = 0.5 * self.beta * self.base_lr * (1 + math.cos(1.0 * ratio * math.pi))
            # ======================================================================
        elif self.mode == 'poly':
            T = T - self.warmup_iters
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), 0.9)
        elif self.mode == 'step':
            # lr 
            lr = self.base_lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented

        if epoch > self.epoch:
            print('\n=>Epoch %i, learning rate = %.4f' % (epoch, lr))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

class LR_Scheduler_Head(LR_Schedule):
    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10



