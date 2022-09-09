from cfgs.constants import *
import re
import torch
import pandas as pd

class InputExample(object):
    def __init__(self, d_text=None, h_text=None, user=None, product=None, label=None):
        self.d_text = d_text
        self.h_text = h_text
        self.user = user
        self.product = product
        self.label = label

def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"sssss", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()

def generate_sents(docuemnt, max_length=230):
    if isinstance(docuemnt, list):
        docuemnt = docuemnt[0]
    string = re.sub(r"[!?]", " ", docuemnt)
    string = re.sub(r"\.{2,}", " ", string)
    sents = string.strip().split('.')
    # print(sents)
    sents = [clean_string(sent) for sent in sents]
    n_sents = []
    n_sent = []
    for sent in sents:
        n_sent.extend(sent)
        if len(n_sent) > max_length:
            n_sents.append(" ".join(n_sent))
            n_sent = []
            n_sent.extend(sent)
    n_sents.append(" ".join(n_sent))
    return n_sents

def generate_sents_(document):
    sents = document.split("<sssss>")
    # n_sents = [clean_string(sent) for sent in sents]
    return sents

def clean_document(document):
    string = re.sub(r"<sssss>", "", document)
    string = re.sub(r" n't", "n't", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'.`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()


class IMDB():
    NAME = 'tweet'
    NUM_CLASS = PSC_DATASET_NUMCLASS['imdb']
    PATH = PSC_DATASET_PATH_MAP['imdb']

    def get_documents(self):
        d_train = self._read_tsv(os.path.join(self.PATH, 'imdb.train.txt.ss'))
        d_dev = self._read_tsv(os.path.join(self.PATH, 'imdb.dev.txt.ss'))
        d_test = self._read_tsv(os.path.join(self.PATH, 'imdb.test.txt.ss'))
        return d_train, d_dev, d_test

    def _read_tsv(self, path):
        examples = []
        pd_reader = pd.read_csv(path, header=None, skiprows=0, encoding="utf-8", sep='\t\t', engine='python')
        for i in range(len(pd_reader[0])):
            document = list([pd_reader[0][i], pd_reader[1][i], pd_reader[3][i], pd_reader[2][i]])
            user = document[0]
            product = document[1]
            review = document[2]
            label = int(document[3]) - 1
            # examples.append(InputExample(d_text=review, user=user, product=product, h_text=generate_sents(clean_document(review)), label=label))
            examples.append(InputExample(d_text=review, user=user, product=product, h_text=generate_sents_(review), label=label))
            # if len(generate_sents(clean_document(review))) > 20:
            #     print("="*30)
            #     print(review)
            #     print(generate_sents(review))
        return examples

    def generating_datasets_hie(self, train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        usr2ids, prd2ids = self.generating_attribute(train, dev, test)

        def generating_dataset(dataset):
            def over_one_example(t):
                h_tokens = [tokenizer.tokenize(text) for text in t.h_text]
                h_tokens_index = [[tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')] for tokens in h_tokens]
                return h_tokens, h_tokens_index

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    h_tokens, h_tokens_index = over_one_example(t.h_text)
                    data.append({"h_tokens": h_tokens,
                                 "h_tokens_index": h_tokens_index,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        torch.save(train_data, os.path.join(self.PATH, 'h_train.pt'))
        torch.save(dev_data,  os.path.join(self.PATH, 'h_dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'h_test.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))

    def generating_dataset_truncate(self,train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        usr2ids, prd2ids = self.generating_attribute(train, dev, test)

        def generating_dataset(dataset):
            def over_one_example(t):
                tokens = tokenizer.tokenize(t.d_text)
                seqence = ['[CLS]'] + tokens + ['[SEP]']
                return seqence, tokenizer.convert_tokens_to_ids(seqence)

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    tokens, tokens_index = over_one_example(t)
                    data.append({"tokens": tokens,
                                 "tokens_index": tokens_index,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        torch.save(train_data, os.path.join(self.PATH, 'train.pt'))
        torch.save(dev_data, os.path.join(self.PATH, 'dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'test.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))


    def generating_dataset_hie_glove(self, train=None, dev=None, test=None):
        import spacy
        from collections import Counter
        from torchtext.vocab import Vocab

        usr2ids, prd2ids = self.generating_attribute(train, dev, test)
        tokenizer = spacy.load('en_core_web_lg').tokenizer
        counter = Counter()

        def generating_dataset(dataset):
            def over_one_example(t):
                h_tokens = [[tok.text for tok in tokenizer(text.strip())] for text in t]
                for item in h_tokens:
                    counter.update(item)
                return h_tokens

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    h_tokens = over_one_example(t.h_text)
                    data.append({"h_tokens": h_tokens,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        vocab = Vocab(counter, vectors='glove.840B.300d')
        torch.save(train_data, os.path.join(self.PATH, 'glove_train.pt'))
        torch.save(dev_data, os.path.join(self.PATH, 'glove_dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'glove_test.pt'))
        torch.save(vocab, os.path.join(self.PATH, 'glove_vocab.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))


    def generating_attribute(self, train=None, dev=None, test=None):
        usr_id = 0
        prd_id = 0
        usr2ids = {}
        prd2ids = {}

        datasets = train + dev + test
        for step, i in enumerate(datasets):
            usr = i.user
            prd = i.product
            if usr in usr2ids:
                pass
            else:
                usr2ids[usr] = usr_id
                usr_id += 1
            if prd in prd2ids:
                pass
            else:
                prd2ids[prd] = prd_id
                prd_id += 1

        return usr2ids, prd2ids


class YELP13():
    NAME = 'yelp_13'
    NUM_CLASS = PSC_DATASET_NUMCLASS['yelp_13']
    PATH = PSC_DATASET_PATH_MAP['yelp_13']

    def get_documents(self):
        d_train = self._read_tsv(os.path.join(self.PATH, 'yelp-2013-seg-20-20.train.ss'))
        d_dev = self._read_tsv(os.path.join(self.PATH, 'yelp-2013-seg-20-20.dev.ss'))
        d_test = self._read_tsv(os.path.join(self.PATH, 'yelp-2013-seg-20-20.test.ss'))
        return d_train, d_dev, d_test

    def _read_tsv(self, path):
        examples = []
        pd_reader = pd.read_csv(path, header=None, skiprows=0, encoding="utf-8", sep='\t\t', engine='python')
        for i in range(len(pd_reader[0])):
            document = list([pd_reader[0][i], pd_reader[1][i], pd_reader[3][i], pd_reader[2][i]])
            user = document[0]
            product = document[1]
            review = document[2]
            label = int(document[3]) - 1
            # examples.append(InputExample(d_text=review, user=user, product=product, h_text=generate_sents(clean_document(review)), label=label))
            examples.append(InputExample(d_text=review, user=user, product=product, h_text=generate_sents_(review), label=label))
            # if len(generate_sents(clean_document(review))) > 20:
            #     print("="*30)
            #     print(review)
            #     print(generate_sents(review))
        return examples

    def generating_datasets_hie(self, train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        usr2ids, prd2ids = self.generating_attribute(train, dev, test)

        def generating_dataset(dataset):
            def over_one_example(t):
                h_tokens = [tokenizer.tokenize(text) for text in t.h_text]
                h_tokens_index = [[tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')] for tokens in h_tokens]
                return h_tokens, h_tokens_index

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    h_tokens, h_tokens_index = over_one_example(t.h_text)
                    data.append({"h_tokens": h_tokens,
                                 "h_tokens_index": h_tokens_index,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        torch.save(train_data, os.path.join(self.PATH, 'h_train.pt'))
        torch.save(dev_data,  os.path.join(self.PATH, 'h_dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'h_test.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))

    def generating_dataset_truncate(self,train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        usr2ids, prd2ids = self.generating_attribute(train, dev, test)

        def generating_dataset(dataset):
            def over_one_example(t):
                tokens = tokenizer.tokenize(t.d_text)
                seqence = ['[CLS]'] + tokens + ['[SEP]']
                return seqence, tokenizer.convert_tokens_to_ids(seqence)

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    tokens, tokens_index = over_one_example(t)
                    data.append({"tokens": tokens,
                                 "tokens_index": tokens_index,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        torch.save(train_data, os.path.join(self.PATH, 'train.pt'))
        torch.save(dev_data, os.path.join(self.PATH, 'dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'test.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))


    def generating_attribute(self, train=None, dev=None, test=None):
        usr_id = 0
        prd_id = 0
        usr2ids = {}
        prd2ids = {}

        datasets = train + dev + test
        for step, i in enumerate(datasets):
            usr = i.user
            prd = i.product
            if usr in usr2ids:
                pass
            else:
                usr2ids[usr] = usr_id
                usr_id += 1
            if prd in prd2ids:
                pass
            else:
                prd2ids[prd] = prd_id
                prd_id += 1

        return usr2ids, prd2ids

    def generating_dataset_hie_glove(self, train=None, dev=None, test=None):
        import spacy
        from collections import Counter
        from torchtext.vocab import Vocab

        usr2ids, prd2ids = self.generating_attribute(train, dev, test)
        tokenizer = spacy.load('en_core_web_lg').tokenizer
        counter = Counter()

        def generating_dataset(dataset):
            def over_one_example(t):
                h_tokens = [[tok.text for tok in tokenizer(text.strip())] for text in t]
                for item in h_tokens:
                    counter.update(item)
                return h_tokens

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    h_tokens = over_one_example(t.h_text)
                    data.append({"h_tokens": h_tokens,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        vocab = Vocab(counter, vectors='glove.840B.300d')
        torch.save(train_data, os.path.join(self.PATH, 'glove_train.pt'))
        torch.save(dev_data, os.path.join(self.PATH, 'glove_dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'glove_test.pt'))
        torch.save(vocab, os.path.join(self.PATH, 'glove_vocab.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))


class YELP14():
    NAME = 'yelp_14'
    NUM_CLASS = PSC_DATASET_NUMCLASS['yelp_14']
    PATH = PSC_DATASET_PATH_MAP['yelp_14']

    def get_documents(self):
        d_train = self._read_tsv(os.path.join(self.PATH, 'yelp-2014-seg-20-20.train.ss'))
        d_dev = self._read_tsv(os.path.join(self.PATH, 'yelp-2014-seg-20-20.dev.ss'))
        d_test = self._read_tsv(os.path.join(self.PATH, 'yelp-2014-seg-20-20.test.ss'))
        return d_train, d_dev, d_test

    def _read_tsv(self, path):
        examples = []
        pd_reader = pd.read_csv(path, header=None, skiprows=0, encoding="utf-8", sep='\t\t', engine='python')
        for i in range(len(pd_reader[0])):
            document = list([pd_reader[0][i], pd_reader[1][i], pd_reader[3][i], pd_reader[2][i]])
            user = document[0]
            product = document[1]
            review = document[2]
            label = int(document[3]) - 1
            # examples.append(InputExample(d_text=review, user=user, product=product, h_text=generate_sents(clean_document(review)), label=label))
            examples.append(InputExample(d_text=review, user=user, product=product, h_text=generate_sents_(review), label=label))
            # if len(generate_sents(clean_document(review))) > 20:
            #     print("="*30)
            #     print(review)
            #     print(generate_sents(review))
        return examples

    def generating_datasets_hie(self, train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        usr2ids, prd2ids = self.generating_attribute(train, dev, test)

        def generating_dataset(dataset):
            def over_one_example(t):
                h_tokens = [tokenizer.tokenize(text) for text in t.h_text]
                h_tokens_index = [[tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')] for tokens in h_tokens]
                return h_tokens, h_tokens_index

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    h_tokens, h_tokens_index = over_one_example(t.h_text)
                    data.append({"h_tokens": h_tokens,
                                 "h_tokens_index": h_tokens_index,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        torch.save(train_data, os.path.join(self.PATH, 'h_train.pt'))
        torch.save(dev_data,  os.path.join(self.PATH, 'h_dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'h_test.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))

    def generating_dataset_truncate(self,train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        usr2ids, prd2ids = self.generating_attribute(train, dev, test)

        def generating_dataset(dataset):
            def over_one_example(t):
                tokens = tokenizer.tokenize(t.d_text)
                seqence = ['[CLS]'] + tokens + ['[SEP]']
                return seqence, tokenizer.convert_tokens_to_ids(seqence)

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    tokens, tokens_index = over_one_example(t)
                    data.append({"tokens": tokens,
                                 "tokens_index": tokens_index,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        torch.save(train_data, os.path.join(self.PATH, 'train.pt'))
        torch.save(dev_data, os.path.join(self.PATH, 'dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'test.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))


    def generating_attribute(self, train=None, dev=None, test=None):
        usr_id = 0
        prd_id = 0
        usr2ids = {}
        prd2ids = {}

        datasets = train + dev + test
        for step, i in enumerate(datasets):
            usr = i.user
            prd = i.product
            if usr in usr2ids:
                pass
            else:
                usr2ids[usr] = usr_id
                usr_id += 1
            if prd in prd2ids:
                pass
            else:
                prd2ids[prd] = prd_id
                prd_id += 1

        return usr2ids, prd2ids

    def generating_dataset_hie_glove(self, train=None, dev=None, test=None):
        import spacy
        from collections import Counter
        from torchtext.vocab import Vocab

        usr2ids, prd2ids = self.generating_attribute(train, dev, test)
        tokenizer = spacy.load('en_core_web_lg').tokenizer
        counter = Counter()

        def generating_dataset(dataset):
            def over_one_example(t):
                h_tokens = [[tok.text for tok in tokenizer(text.strip())] for text in t]
                for item in h_tokens:
                    counter.update(item)
                return h_tokens

            if dataset is None:
                return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    usr_index = usr2ids[t.user]
                    prd_index = prd2ids[t.product]
                    h_tokens = over_one_example(t.h_text)
                    data.append({"h_tokens": h_tokens,
                                 "label": label,
                                 "usr_index": usr_index,
                                 "prd_index": prd_index,
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100),
                          end="")
                return data

        print("==loading train datasets")
        train_data = generating_dataset(train)
        print("\rDone!".ljust(60))
        print("==loading dev datasets")
        dev_data = generating_dataset(dev)
        print("\rDone!".ljust(60))
        print("==loading test datasets")
        test_data = generating_dataset(test)
        print("\rDone!".ljust(60))
        vocab = Vocab(counter, vectors='glove.840B.300d')
        torch.save(train_data, os.path.join(self.PATH, 'glove_train.pt'))
        torch.save(dev_data, os.path.join(self.PATH, 'glove_dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'glove_test.pt'))
        torch.save(vocab, os.path.join(self.PATH, 'glove_vocab.pt'))
        torch.save(usr2ids, os.path.join(self.PATH, 'usr2ids.pt'))
        torch.save(prd2ids, os.path.join(self.PATH, 'prd2ids.pt'))



