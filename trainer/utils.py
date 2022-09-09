import os
import torch
from dataset import DATASET_MAP, PSC_DATASET_MAP

def multi_acc(y, preds):
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

def multi_mse(y, preds):
    mse_loss = torch.nn.MSELoss()
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
    return mse_loss(y.float(), preds.float())

def load_itr_ABSC(config, from_scratch=False):
    dataset = DATASET_MAP[config.dataset]()
    config.num_labels = dataset.NUM_CLASS
    try:
        if from_scratch: raise Exception
        train_data = torch.load(os.path.join(dataset.PATH, 'train.pt'))
        dev_data = torch.load(os.path.join(dataset.PATH, 'dev.pt'))
        test_data = torch.load(os.path.join(dataset.PATH, 'test.pt'))
    except:
        dataset.generating_datasets(*dataset.get_documents())
        train_data = torch.load(os.path.join(dataset.PATH, 'train.pt'))
        dev_data = torch.load(os.path.join(dataset.PATH, 'dev.pt'))
        test_data = torch.load(os.path.join(dataset.PATH, 'test.pt'))

    from trainer.bucket_iterator import BucketIterator
    train_data_itr = BucketIterator(train_data, config.batch_size,
                                    device=config.device) if train_data is not None else None
    dev_data_itr = BucketIterator(dev_data, config.batch_size,
                                  device=config.device) if dev_data is not None else None
    test_data_itr = BucketIterator(test_data, config.batch_size,
                                   device=config.device) if test_data is not None else None
    return train_data_itr, dev_data_itr, test_data_itr


def load_itr_PSC(config, from_scratch=False):
    dataset = PSC_DATASET_MAP[config.dataset]()
    config.num_labels = dataset.NUM_CLASS
    try:
        if from_scratch: raise Exception
        train_data = torch.load(os.path.join(dataset.PATH, 'h_train.pt'))
        dev_data = torch.load(os.path.join(dataset.PATH, 'h_dev.pt'))
        test_data = torch.load(os.path.join(dataset.PATH, 'h_test.pt'))
        usr2ids = torch.load(os.path.join(dataset.PATH,'usr2ids.pt'))
        prd2ids = torch.load(os.path.join(dataset.PATH,'prd2ids.pt'))
    except:
        dataset.generating_datasets_hie(*dataset.get_documents())
        train_data = torch.load(os.path.join(dataset.PATH, 'h_train.pt'))
        dev_data = torch.load(os.path.join(dataset.PATH, 'h_dev.pt'))
        test_data = torch.load(os.path.join(dataset.PATH, 'h_test.pt'))
        usr2ids = torch.load(os.path.join(dataset.PATH, 'usr2ids.pt'))
        prd2ids = torch.load(os.path.join(dataset.PATH, 'prd2ids.pt'))

    from trainer.hie_bucket_iterator import HieBucketIterator
    config.num_usrs = len(usr2ids)
    config.num_prds = len(prd2ids)
    train_data_itr = HieBucketIterator(train_data, config.batch_size,
                                       device=config.device) if train_data is not None else None
    dev_data_itr = HieBucketIterator(dev_data, config.batch_size,
                                     device=config.device) if dev_data is not None else None
    test_data_itr = HieBucketIterator(test_data, config.batch_size,
                                      device=config.device) if test_data is not None else None

    config.num_train_optimization_steps = int((len(train_data_itr) / config.gradient_accumulation_steps) * config.max_epoch)
    return train_data_itr, dev_data_itr, test_data_itr

def load_itr_PSC_trunc(config, from_scratch=False):
    dataset = PSC_DATASET_MAP[config.dataset]()
    config.num_labels = dataset.NUM_CLASS
    try:
        if from_scratch: raise Exception
        train_data = torch.load(os.path.join(dataset.PATH, 'train.pt'))
        dev_data = torch.load(os.path.join(dataset.PATH, 'dev.pt'))
        test_data = torch.load(os.path.join(dataset.PATH, 'test.pt'))
        usr2ids = torch.load(os.path.join(dataset.PATH,'usr2ids.pt'))
        prd2ids = torch.load(os.path.join(dataset.PATH,'prd2ids.pt'))
    except:
        dataset.generating_dataset_truncate(*dataset.get_documents())
        train_data = torch.load(os.path.join(dataset.PATH, 'train.pt'))
        dev_data = torch.load(os.path.join(dataset.PATH, 'dev.pt'))
        test_data = torch.load(os.path.join(dataset.PATH, 'test.pt'))
        usr2ids = torch.load(os.path.join(dataset.PATH, 'usr2ids.pt'))
        prd2ids = torch.load(os.path.join(dataset.PATH, 'prd2ids.pt'))

    from trainer.trunc_bucket_iterator import TruncBucketIterator
    config.num_usrs = len(usr2ids)
    config.num_prds = len(prd2ids)
    train_data_itr = TruncBucketIterator(train_data, config.batch_size, trunc=config.trunc, trunc_length=config.trunc_length, device=config.device) if train_data is not None else None
    dev_data_itr = TruncBucketIterator(dev_data, config.batch_size, trunc=config.trunc, trunc_length=config.trunc_length,
                                     device=config.device) if dev_data is not None else None
    test_data_itr = TruncBucketIterator(test_data, config.batch_size, trunc=config.trunc, trunc_length=config.trunc_length,
                                      device=config.device) if test_data is not None else None

    config.num_train_optimization_steps = int((len(train_data_itr) / config.gradient_accumulation_steps) * config.max_epoch)
    return train_data_itr, dev_data_itr, test_data_itr


def load_itr_PSC_hie_glove(config, from_scratch=False):
    dataset = PSC_DATASET_MAP[config.dataset]()
    config.num_labels = dataset.NUM_CLASS
    try:
        if from_scratch: raise Exception
        train_data = torch.load(os.path.join(dataset.PATH, 'glove_train.pt'))
        dev_data = torch.load(os.path.join(dataset.PATH, 'glove_dev.pt'))
        test_data = torch.load(os.path.join(dataset.PATH, 'glove_test.pt'))
        vocab = torch.load(os.path.join(dataset.PATH, 'glove_vocab.pt'))
        usr2ids = torch.load(os.path.join(dataset.PATH,'usr2ids.pt'))
        prd2ids = torch.load(os.path.join(dataset.PATH,'prd2ids.pt'))
    except:
        dataset.generating_dataset_hie_glove(*dataset.get_documents())
        train_data = torch.load(os.path.join(dataset.PATH, 'train.pt'))
        dev_data = torch.load(os.path.join(dataset.PATH, 'dev.pt'))
        test_data = torch.load(os.path.join(dataset.PATH, 'test.pt'))
        vocab = torch.load(os.path.join(dataset.PATH, 'glove_vocab.pt'))
        usr2ids = torch.load(os.path.join(dataset.PATH, 'usr2ids.pt'))
        prd2ids = torch.load(os.path.join(dataset.PATH, 'prd2ids.pt'))

    from trainer.glove_hie_bucket_iterator import HieBucketIterator
    config.num_usrs = len(usr2ids)
    config.num_prds = len(prd2ids)
    config.text_embedding = vocab.vectors
    config.batch_size = 32
    config.lr_base = 1e-3
    config.pad_idx = vocab.stoi['<pad>'] # 1
    train_data_itr = HieBucketIterator(train_data, config.batch_size, stoi= vocab.stoi, device=config.device) if train_data is not None else None
    dev_data_itr = HieBucketIterator(dev_data, config.batch_size, stoi= vocab.stoi, device=config.device) if dev_data is not None else None
    test_data_itr = HieBucketIterator(test_data, config.batch_size, stoi= vocab.stoi, device=config.device) if test_data is not None else None

    config.num_train_optimization_steps = int((len(train_data_itr) / config.gradient_accumulation_steps) * config.max_epoch)
    return train_data_itr, dev_data_itr, test_data_itr
