import os
import time
import torch
import datetime
import numpy as np
import torch.nn as nn

from models import *
from models.get_optim import get_Adam_optim, get_Adam_optim_v2

from trainer.utils import multi_acc, multi_mse, load_itr_PSC_trunc

class Trainer:
    def __init__(self, config):
        self.config = config
        pretrained_weights = 'bert-base-uncased'
        self.train_itr, self.dev_itr, self.test_itr = load_itr_PSC_trunc(config, from_scratch=False)
        self.log_file = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt'

        net = models_PSC_single.SBertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=config.num_labels, cus_config=config)
        net.bert.init_personalized()
        model = nn.ModuleList([net])
        self.optim, self.scheduler = get_Adam_optim_v2(config, model)
        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net).to(self.config.device)
        else:
            self.net = net.to(self.config.device)
        self.early_stop = config.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.step_count = 0
        self.oom_time = 0

        training_steps_per_epoch = len(self.train_itr) // (config.gradient_accumulation_steps)
        self.config.num_train_optimization_steps = self.config.max_epoch * training_steps_per_epoch

    def ensureDirs(self, *dir_paths):
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def load_state(self):
        SAVED_MODEL_PATH = self.config.ckpts_path
        net = models_PSC.HieTransformerForClassification(self.config)
        net.load_state_dict(torch.load(os.path.join(SAVED_MODEL_PATH, self.config.dataset, "ckpt.pt")))
        input_embed_model = models_PSC.BertEmbedding.from_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(net).to(self.config.device)
            self.input_embed_model = torch.nn.DataParallel(input_embed_model).to(self.config.device)
        else:
            self.net = net.to(self.config.device)
            self.input_embed_model = input_embed_model.to(self.config.device)

    def save_state(self):
        SAVED_MODEL_PATH = self.config.ckpts_path
        self.ensureDirs(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
        if self.config.n_gpu > 1:
            torch.save(self.net.module.state_dict(), os.path.join(SAVED_MODEL_PATH, self.config.dataset, "ckpt.pt"))
        else:
            torch.save(self.net.state_dict(), os.path.join(SAVED_MODEL_PATH, self.config.dataset, "ckpt.pt"))


    def empty_log(self):
        if (os.path.exists(self.log_file)):
            os.remove(self.log_file)
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def get_logging(self, loss, acc, rmse, eval='training'):
        logs = \
            '==={} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
            "total_loss:{:>2.3f}\ttotal_acc:{:>2.3f}\ttotal_rmse:{:>2.3f}".format(loss, acc, rmse) + "\n"
        return logs

    def run(self, run_mode):
        if run_mode == "train":
            self.empty_log()
            # self.do_statistic()
            start_time = time.time()
            self._train()
            end_time = time.time()
            print("time is " + str(end_time - start_time))
        if run_mode == "test":
            self.load_state()
            # logging test logs
            self.net.eval()
            self.input_embed_model.eval()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse = self.eval(self.test_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse,
                                         eval="testing")
            print(eval_logs)
        # if run_mode == 'train':
        #     self.test_dataset()

    def _train(self):
        # Save log information
        logfile = open(self.log_file, 'a+')
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n' +
            'data:' + str(self.config.dataset) +
            '\n'
        )
        logfile.close()
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        for epoch in range(0, self.config.max_epoch):
            self.optim.zero_grad()
            for step, batch in enumerate(self.train_itr):
                self.step_count += 1
                self.net.train()
                start_time = time.time()
                input_ids, labels, usr_ids, prd_ids = batch['batch_text_indices'], batch['batch_labels'], batch['batch_usr_indeces'], batch['batch_prd_indeces']
                attention_mask = (input_ids != 0).long() # id of [PAD] is 0
                try:
                    logits = self.net(input_ids,
                                      attention_mask=attention_mask,
                                      user_ids=usr_ids,
                                      item_ids=prd_ids,
                                      )[0]
                    loss = loss_fn(logits, labels)
                    metric_acc = acc_fn(labels, logits)
                    metric_mse = mse_fn(labels, logits)
                    total_loss.append(loss.item())
                    total_acc.append(metric_acc.item())
                    total_mse.append(metric_mse.item())

                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                    loss.backward()

                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optim.step()
                        if self.scheduler is not None: self.scheduler.step()
                        self.optim.zero_grad()


                except RuntimeError as exception:
                    print("--"*30)
                    if "out of memory" in str(exception):
                        self.oom_time += 1
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print(labels)
                        print(exception)
                        print("-".center(30, str(exception)))
                        raise exception

                if self.step_count % self.config.moniter_per_step == 0 and epoch > 2:

                    train_loss, train_acc, train_rmse = \
                        np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean())

                    logs = ("    steps:{:^5}    ".format(self.step_count)).center(85, "-") \
                           + "".center(70, " ") + '\n' + \
                           self.get_logging(train_loss, train_acc, train_rmse, eval="training")
                    print("\r" + logs)
                    if self.oom_time > 0:
                        print("num of out of memory is: " + str(self.oom_time))

                    # logging training logs
                    self.logging(self.log_file, logs)

                    # reset monitors
                    total_loss = []
                    total_acc = []
                    total_mse = []

                    # evaluating phase
                    self.net.eval()
                    with torch.no_grad():
                        eval_loss, eval_acc, eval_rmse = self.eval(self.dev_itr)
                    eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval="evaluating")
                    print("\r" + eval_logs)

                    # logging evaluating logs
                    self.logging(self.log_file, eval_logs)

                    # monitoring results on every steps
                    end_time = time.time()
                    span_time = (end_time - start_time) * int(self.config.moniter_per_step - (self.step_count % self.config.moniter_per_step))
                    h = span_time // (60 * 60)
                    m = (span_time % (60 * 60)) // 60
                    s = (span_time % (60 * 60)) % 60 // 1
                    print(
                        "\rIteration: {:>3}/{:^3} ({:>4.1f}%) -- Loss: {:.5f} -ETA {:>2}h-{:>2}m-{:>2}s".format(
                            self.step_count % self.config.moniter_per_step, self.config.moniter_per_step,
                            (self.step_count % self.config.moniter_per_step) / self.config.moniter_per_step,
                            loss, int(h), int(m), int(s)),
                        end="")

                    # early stopping
                    if eval_acc > self.best_dev_acc:
                        self.unimproved_iters = 0
                        self.best_dev_acc = eval_acc

                        # saving models
                        ## getting state
                        self.save_state()
                    else:
                        self.unimproved_iters += 1
                        if self.unimproved_iters >= self.config.patience and self.early_stop == True:
                            early_stop_logs = self.log_file + "\n" + \
                                              "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch,
                                                                                                   self.best_dev_acc)
                            print(early_stop_logs)
                            self.logging(self.log_file, early_stop_logs)

                            # load best model on dev datasets
                            self.load_state()
                            # logging test logs
                            self.net.eval()
                            self.pre_embed_model.eval()
                            self.follow_embed_model.eval()
                            # self.input_embed_model.eval()
                            with torch.no_grad():
                                eval_loss, eval_acc, eval_rmse = self.eval(self.test_itr)
                            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse,
                                                         eval="testing")
                            print("\r" + eval_logs)
                            # logging testt logs
                            self.logging(self.log_file, eval_logs)
                            exit()

    def eval(self, eval_itr):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            input_ids, labels, usr_ids, prd_ids = batch['batch_text_indices'], batch['batch_labels'], batch['batch_usr_indeces'], batch['batch_prd_indeces']
            attention_mask = (input_ids != 0).long()  # id of [PAD] is 0
            logits = self.net(input_ids,
                              attention_mask=attention_mask,
                              user_ids=usr_ids,
                              item_ids=prd_ids,
                              )[0]
            loss = loss_fn(logits, labels)
            metric_acc = acc_fn(labels, logits)
            metric_mse = mse_fn(labels, logits)

            total_loss.append(loss.item())
            total_acc.append(metric_acc.item())
            total_mse.append(metric_mse.item())

            # monitoring results on every steps1
            end_time = time.time()
            span_time = (end_time - start_time) * (int(len(eval_itr)) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)   -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(eval_itr)),
                    100 * (step) / int(len(eval_itr)),
                    int(h), int(m), int(s)),
                end="")

        return np.array(total_loss).mean(0), np.array(total_acc).mean(0), np.sqrt(np.array(total_mse).mean(0))

    def test_dataset(self,):
        for batch in self.train_itr:
            input_ids, labels, usr_ids, prd_ids = batch['batch_text_indices'], batch['batch_labels'], batch['batch_usr_indeces'], batch['batch_prd_indeces']
            print(batch)









