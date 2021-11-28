import torch
def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')
    word_list = set()
    len_sentence = []
    for line in text:
        tokens = [t for t in line.split()]
        len_sentence.append(len(tokens))
        word_list = word_list.union(set(tokens))
    text.close()

    word_list = list(sorted(word_list))
    word2number_dict = {w: i + 4 for i, w in enumerate(word_list)}
    number2word_dict = {i + 4: w for i, w in enumerate(word_list)}

    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk>"] = 1
    number2word_dict[1] = "<unk>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    len_max = max(len_sentence)

    return word2number_dict, number2word_dict, len_max

def make_batch(path, word2number_dict, batch_size, lenmax, state):

    text = open(path, 'r', encoding='utf-8')
    input_batch = []
    all_input_batch = []

    for i, sent in enumerate(text):
        tokens = [t for t in sent.split()]

        # 为每条句子添加<sos>和<eos>符号
        if state == "enc_inputs":
            tokens = tokens
        elif state == "dec_inputs":
            tokens = ['<sos>'] + tokens
        else:
            tokens.append('<eos>')

        # 为长度不足lenmax的句子补上<pad>符号
        if len(tokens) <= lenmax:
            tokens = tokens + ["<pad>"] * (lenmax + 1 - len(tokens))
        tokens = tokens[:80]
        input = [word2number_dict[n] if n in word2number_dict else word2number_dict['<unk>']
                     for n in tokens]
        input_batch.append(input)

        if len(input_batch) == batch_size:
            all_input_batch.append(input_batch)
            input_batch = []

    text.close()

    return torch.LongTensor(all_input_batch)

