import wandb
class TrainInfo:
    # pylint: disable=too-many-instance-attributes
    batch_id = 0
    running_loss = []
    best_loss = float('inf')
    best_las = 0
    best_uas = 0
    best_batch = 0
    lr_reductions = 0
    MAX_REDUCTIONS = 0#10

    def __init__(self, wait_iterations, eval_batches):
        self.wait_iterations = wait_iterations
        self.eval_batches = eval_batches


    @property
    def stuck(self):
        return (self.batch_id - self.best_batch) > self.wait_iterations

    @property
    def reduce_lr(self):
        if self.stuck and (self.lr_reductions < self.MAX_REDUCTIONS):
            self.lr_reductions += 1
            self.best_batch = self.batch_id
            return True

        return False

    @property
    def finish(self):
        #print("is stuck {}".format(self.stuck))
        return self.stuck and (self.lr_reductions >= self.MAX_REDUCTIONS)

    @property
    def eval(self):
        return (self.batch_id % self.eval_batches) == 0

    @property
    def max_epochs(self):
        return self.best_batch + self.wait_iterations

    @property
    def avg_loss(self):
        return sum(self.running_loss) / len(self.running_loss)

    def new_batch(self, loss):
        self.batch_id += 1
        self.running_loss += [loss]

    def is_best(self, dev_results):
        dev_loss, dev_las, dev_uas = dev_results
        # if dev_loss < self.best_loss:
        if dev_las > self.best_las:
            self.best_loss = dev_loss
            self.best_las = dev_las
            self.best_uas = dev_uas
            self.best_batch = self.batch_id
            return True


        return False

    def reset_loss(self):
        self.running_loss = []

    def print_progress(self, dev_results,file):
        dev_loss, dev_las, dev_uas = dev_results

        log_dict = {'Training loss':self.avg_loss,'Dev Loss':dev_loss,'Dev LAS':dev_las,
                    'Dev UAS':dev_uas,'batch_id':self.batch_id,'max_epochs':self.max_epochs}
        wandb.log(log_dict)

        print('(%05d/%05d) Training loss: %.4f Dev loss: %.4f Dev las: %.4f Dev uas: %.4f' %
              (self.batch_id, self.max_epochs, self.avg_loss, dev_loss, dev_las, dev_uas))
        file.write('(%05d/%05d) Training loss: %.4f Dev loss: %.4f Dev las: %.4f Dev uas: %.4f' %
              (self.batch_id, self.max_epochs, self.avg_loss, dev_loss, dev_las, dev_uas))
        self.reset_loss()
