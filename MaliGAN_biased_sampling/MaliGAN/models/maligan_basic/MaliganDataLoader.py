import numpy as np


class DataLoader():
    def __init__(self, batch_size, seq_length, end_token=0):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token

    def create_batches(self, data_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class DisDataloader():
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.seq_length = seq_length

    def load_train_data(self, positive_file, low_mic_file, negative_file, pos_low_mic_frac):
        # Load data
        positive_examples = []
        low_mic_examples = []
        negative_examples = []
        self.pos_low_mic_frac = pos_low_mic_frac
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    positive_examples.append(parse_line)
        with open(low_mic_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    low_mic_examples.append(parse_line)
            #print('pline:\n',line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
            #print('nline:\n', line)
        
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        #print('positivelabels\n',positive_labels) #added a print statement
        low_mic_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        #print('negativelabels\n',negative_labels) #

        self.positive_num_batch = int(len(self.positive_labels) / self.batch_size)
        self.low_mic_num_batch = int(len(self.low_mic_labels) / self.batch_size)
        self.negative_num_batch = int(len(self.negative_labels) / self.batch_size)

        self.positive_examples = np.array(positive_examples)[:self.positive_num_batch * self.batch_size]
        self.low_mic_examples = np.array(low_mic_examples)[:self.low_mic_num_batch * self.batch_size]
        self.negative_examples = np.array(negative_examples)[:self.negative_num_batch * self.batch_size]

        self.positive_labels = np.array(positive_labels)[:self.positive_num_batch * self.batch_size]
        self.low_mic_labels = np.array(low_mic_labels)[:self.low_mic_num_batch * self.batch_size]
        self.negative_labels = np.array(negative_labels)[:self.negative_num_batch * self.batch_size]

        self.pointer = 0

    def next_batch(self):
        total_positive_examples_per_batch = int(np.round(self.batch_size/2))
        positive_examples_per_batch = int(np.floor(total_positive_examples_per_batch * (1 - self.pos_low_mic_frac)))
        low_mic_examples_per_batch = total_positive_examples_per_batch - positive_examples_per_batch
        negative_examples_per_batch = self.batch_size - total_positive_examples_per_batch
        
        positive_idx = np.random.choice(range(len(self.positive_labels)), positive_examples_per_batch, replace=False)
        low_mic_idx = np.random.choice(range(len(self.low_mic_labels)), low_mic_examples_per_batch, replace=False)
        negative_idx = np.random.choice(range(len(self.negative_labels)), negative_examples_per_batch, replace=False)

        positive_examples = self.positive_examples[positive_idx]
        low_mic_examples = self.low_mic_examples[low_mic_idx]
        negative_examples = self.negative_examples[negative_idx]

        positive_labels = self.positive_labels[positive_idx]
        low_mic_labels = self.low_mic_labels[low_mic_idx]
        negative_labels = self.negative_labels[negative_idx]

        batch_examples = np.concatenate([positive_examples, low_mic_examples, negative_examples])
        batch_labels = np.concatenate([positive_labels, low_mic_labels, negative_labels])

        shuffle_idx = np.random.choice(range(len(batch_examples)), len(batch_examples), replace=False)
        batch_examples = batch_examples[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]
        
        ret = batch_examples, batch_labels
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
