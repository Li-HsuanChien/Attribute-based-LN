from cfgs.constants import *
import torch

class InputExample(object):
    def __init__(self, l_text=None, r_text=None, aspect=None, label=None):
        self.l_text = l_text
        self.r_text = r_text
        self.aspect = aspect
        self.label = label

class Tweet_DataSet(object):
    NAME = 'tweet'
    NUM_CLASS = ABSC_DATASET_NUMCLASS['twt']
    PATH = ABSC_DATASET_PATH_MAP['twt']

    def get_documents(self):
        d_train =self._read_tsv(os.path.join(self.PATH, 'train.raw'))
        d_dev = None
        d_test =self._read_tsv(os.path.join(self.PATH, 'test.raw'))
        return d_train, d_dev, d_test

    def _read_tsv(self, path):
        examples = []
        print(path)
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            label = int(polarity) + 1
            examples.append(InputExample(l_text=text_left, r_text=text_right, aspect=aspect, label=label))
        return examples

    def generating_datasets(self, train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def generating_dataset(dataset):
            def over_one_example(t):
                l_tokens = tokenizer.tokenize(t.l_text) if len(t.l_text)>0 else []
                aspect_mask = [0]*len(l_tokens)
                aspect_tokens = tokenizer.tokenize(t.aspect)
                aspect_mask.extend([1]*len(aspect_tokens))
                r_tokens = tokenizer.tokenize(t.r_text) if len(t.r_text)>0 else []
                aspect_mask.extend([0]*len(r_tokens))
                tokens = l_tokens + aspect_tokens + r_tokens

                tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                # tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')] + tokenizer.convert_tokens_to_ids(aspect_tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                aspect_tokens_index = tokenizer.convert_tokens_to_ids(aspect_tokens)
                aspect_mask = [0] + aspect_mask + [0]
                return tokens_index, aspect_tokens_index, aspect_mask

            if dataset is None: return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    tokens_index, aspect_tokens_index, aspect_mask = over_one_example(t)
                    data.append({"tokens_index":tokens_index,
                                 "aspect_tokens_index":aspect_tokens_index,
                                 "aspect_mask":aspect_mask,
                                 "label":label
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100), end="")
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
        torch.save(dev_data,  os.path.join(self.PATH, 'dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'test.pt'))


class Restaurant_16_DataSet(object):
    NAME = 'Restaurant_16'
    NUM_CLASS = ABSC_DATASET_NUMCLASS['rst_16']
    PATH = ABSC_DATASET_PATH_MAP['rst_16']

    def get_documents(self):
        d_train =self._read_tsv(os.path.join(self.PATH, 'restaurant_train.raw'))
        d_dev = None
        d_test =self._read_tsv(os.path.join(self.PATH, 'restaurant_test.raw'))
        return d_train, d_dev, d_test

    def _read_tsv(self, path):
        examples = []
        print(path)
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            label = int(polarity) + 1
            examples.append(InputExample(l_text=text_left, r_text=text_right, aspect=aspect, label=label))
        return examples

    def generating_datasets(self, train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def generating_dataset(dataset):
            def over_one_example(t):
                l_tokens = tokenizer.tokenize(t.l_text) if len(t.l_text)>0 else []
                aspect_mask = [0]*len(l_tokens)
                aspect_tokens = tokenizer.tokenize(t.aspect)
                aspect_mask.extend([1]*len(aspect_tokens))
                r_tokens = tokenizer.tokenize(t.r_text) if len(t.r_text)>0 else []
                aspect_mask.extend([0]*len(r_tokens))
                tokens = l_tokens + aspect_tokens + r_tokens

                tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                # tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')] + tokenizer.convert_tokens_to_ids(aspect_tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                aspect_tokens_index = tokenizer.convert_tokens_to_ids(aspect_tokens)
                aspect_mask = [0] + aspect_mask + [0]
                return tokens_index, aspect_tokens_index, aspect_mask

            if dataset is None: return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    tokens_index, aspect_tokens_index, aspect_mask = over_one_example(t)
                    data.append({"tokens_index":tokens_index,
                                 "aspect_tokens_index":aspect_tokens_index,
                                 "aspect_mask":aspect_mask,
                                 "label":label
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100), end="")
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
        torch.save(dev_data,  os.path.join(self.PATH, 'dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'test.pt'))


class Restaurant_15_DataSet(object):
    NAME = 'Restaurant_15'
    NUM_CLASS = ABSC_DATASET_NUMCLASS['rst_15']
    PATH = ABSC_DATASET_PATH_MAP['rst_15']

    def get_documents(self):
        d_train =self._read_tsv(os.path.join(self.PATH, 'restaurant_train.raw'))
        d_dev = None
        d_test =self._read_tsv(os.path.join(self.PATH, 'restaurant_test.raw'))
        return d_train, d_dev, d_test

    def _read_tsv(self, path):
        examples = []
        print(path)
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            label = int(polarity) + 1
            examples.append(InputExample(l_text=text_left, r_text=text_right, aspect=aspect, label=label))
        return examples

    def generating_datasets(self, train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def generating_dataset(dataset):
            def over_one_example(t):
                l_tokens = tokenizer.tokenize(t.l_text) if len(t.l_text)>0 else []
                aspect_mask = [0]*len(l_tokens)
                aspect_tokens = tokenizer.tokenize(t.aspect)
                aspect_mask.extend([1]*len(aspect_tokens))
                r_tokens = tokenizer.tokenize(t.r_text) if len(t.r_text)>0 else []
                aspect_mask.extend([0]*len(r_tokens))
                tokens = l_tokens + aspect_tokens + r_tokens

                tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                # tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')] + tokenizer.convert_tokens_to_ids(aspect_tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                aspect_tokens_index = tokenizer.convert_tokens_to_ids(aspect_tokens)
                aspect_mask = [0] + aspect_mask + [0]
                return tokens_index, aspect_tokens_index, aspect_mask

            if dataset is None: return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    tokens_index, aspect_tokens_index, aspect_mask = over_one_example(t)
                    data.append({"tokens_index":tokens_index,
                                 "aspect_tokens_index":aspect_tokens_index,
                                 "aspect_mask":aspect_mask,
                                 "label":label
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100), end="")
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
        torch.save(dev_data,  os.path.join(self.PATH, 'dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'test.pt'))


class Restaurant_14_DataSet(object):
    NAME = 'Restaurant_14'
    NUM_CLASS = ABSC_DATASET_NUMCLASS['rst_14']
    PATH = ABSC_DATASET_PATH_MAP['rst_14']

    def get_documents(self):
        d_train =self._read_tsv(os.path.join(self.PATH, 'restaurant_train.raw'))
        d_dev = None
        d_test =self._read_tsv(os.path.join(self.PATH, 'restaurant_test.raw'))
        return d_train, d_dev, d_test

    def _read_tsv(self, path):
        examples = []
        print(path)
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            label = int(polarity) + 1
            examples.append(InputExample(l_text=text_left, r_text=text_right, aspect=aspect, label=label))
        return examples

    def generating_datasets(self, train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def generating_dataset(dataset):
            def over_one_example(t):
                l_tokens = tokenizer.tokenize(t.l_text) if len(t.l_text)>0 else []
                aspect_mask = [0]*len(l_tokens)
                aspect_tokens = tokenizer.tokenize(t.aspect)
                aspect_mask.extend([1]*len(aspect_tokens))
                r_tokens = tokenizer.tokenize(t.r_text) if len(t.r_text)>0 else []
                aspect_mask.extend([0]*len(r_tokens))
                tokens = l_tokens + aspect_tokens + r_tokens

                tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                # tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')] + tokenizer.convert_tokens_to_ids(aspect_tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                aspect_tokens_index = tokenizer.convert_tokens_to_ids(aspect_tokens)
                aspect_mask = [0] + aspect_mask + [0]
                return tokens_index, aspect_tokens_index, aspect_mask

            if dataset is None: return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    tokens_index, aspect_tokens_index, aspect_mask = over_one_example(t)
                    data.append({"tokens_index":tokens_index,
                                 "aspect_tokens_index":aspect_tokens_index,
                                 "aspect_mask":aspect_mask,
                                 "label":label
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100), end="")
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
        torch.save(dev_data,  os.path.join(self.PATH, 'dev.pt'))
        torch.save(test_data, os.path.join(self.PATH, 'test.pt'))


class Lap_14_DataSet(object):
    NAME = 'Lap_14'
    NUM_CLASS = ABSC_DATASET_NUMCLASS['lpt_14']
    PATH = ABSC_DATASET_PATH_MAP['lpt_14']

    def get_documents(self):
        d_train =self._read_tsv(os.path.join(self.PATH, 'laptop_train.raw'))
        d_dev = None
        d_test =self._read_tsv(os.path.join(self.PATH, 'laptop_test.raw'))
        return d_train, d_dev, d_test

    def _read_tsv(self, path):
        examples = []
        print(path)
        fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            label = int(polarity) + 1
            examples.append(InputExample(l_text=text_left, r_text=text_right, aspect=aspect, label=label))
        return examples

    def generating_datasets(self, train=None, dev=None, test=None):
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def generating_dataset(dataset):
            def over_one_example(t):
                l_tokens = tokenizer.tokenize(t.l_text) if len(t.l_text)>0 else []
                aspect_mask = [0]*len(l_tokens)
                aspect_tokens = tokenizer.tokenize(t.aspect)
                aspect_mask.extend([1]*len(aspect_tokens))
                r_tokens = tokenizer.tokenize(t.r_text) if len(t.r_text)>0 else []
                aspect_mask.extend([0]*len(r_tokens))
                tokens = l_tokens + aspect_tokens + r_tokens

                tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                # tokens_index = [tokenizer._convert_token_to_id('[CLS]')] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer._convert_token_to_id('[SEP]')] + tokenizer.convert_tokens_to_ids(aspect_tokens) + [tokenizer._convert_token_to_id('[SEP]')]
                aspect_tokens_index = tokenizer.convert_tokens_to_ids(aspect_tokens)
                aspect_mask = [0] + aspect_mask + [0]
                return tokens_index, aspect_tokens_index, aspect_mask

            if dataset is None: return None
            else:
                data = []
                for step, t in enumerate(dataset):
                    label = t.label
                    tokens_index, aspect_tokens_index, aspect_mask = over_one_example(t)
                    data.append({"tokens_index":tokens_index,
                                 "aspect_tokens_index":aspect_tokens_index,
                                 "aspect_mask":aspect_mask,
                                 "label":label
                                 })
                    print("\rIteration: {:>5}/{:>5} ({:.2f}%)".format(step, len(dataset), step / len(dataset) * 100), end="")
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
        torch.save(train_data, os.path.join(self.PATH,  'train.pt'))
        torch.save(dev_data,  os.path.join(self.PATH, 'dev.pt'))
        torch.save(test_data, os.path.join(self.PATH,  'test.pt'))