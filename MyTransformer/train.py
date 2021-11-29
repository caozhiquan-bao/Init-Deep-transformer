import torch
import torch.nn as nn
import torch.optim as optim
import random
import transformerpre
import data
import math

train_epoch = 30
d_model = 512
d_ff = 1024
d_k = d_v = 128
n_layers = 6
n_heads = 4
learn_rate = 0.0005
batch_size = 50
train_src_path = './data-bin/iwslt14.tokenized.de-en/train.de'
train_tgt_path = './data-bin/iwslt14.tokenized.de-en/train.en'
valid_src_path = './data-bin/iwslt14.tokenized.de-en/valid.de'
valid_tgt_path = './data-bin/iwslt14.tokenized.de-en/valid.en'
seed = 666
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子

src_word2number_dict, src_number2word_dict, src_len = data.make_dict(train_src_path)
src_vocab_size = len(src_word2number_dict)
print("Src_Vocabulary size:", src_vocab_size)
tgt_word2number_dict, tgt_number2word_dict, tgt_len = data.make_dict(train_tgt_path)
tgt_vocab_size = len(tgt_word2number_dict)
print("Tgt_Vocabulary size:", tgt_vocab_size)

train_enc_inputs = data.make_batch(train_src_path, src_word2number_dict, batch_size, 80, "enc_inputs")
train_dec_inputs = data.make_batch(train_tgt_path, tgt_word2number_dict, batch_size, 80, "dec_inputs")
train_dec_outputs = data.make_batch(train_tgt_path, tgt_word2number_dict, batch_size, 80, "dec_outputs")
valid_enc_inputs = data.make_batch(valid_src_path, src_word2number_dict, batch_size, 80, "enc_inputs")
valid_dec_inputs = data.make_batch(valid_tgt_path, tgt_word2number_dict, batch_size, 80, "dec_inputs")
valid_dec_outputs = data.make_batch(valid_tgt_path, tgt_word2number_dict, batch_size, 80, "dec_outputs")


model = transformerpre.Transformer(d_model, d_ff, d_k, n_layers, n_heads, src_vocab_size, tgt_vocab_size).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))

#Training
batch_number = len(train_enc_inputs)
train_set = list(zip(train_enc_inputs, train_dec_inputs, train_dec_outputs))
for epoch in range(train_epoch):
    count_batch = 0
    random.shuffle(train_set)
    for input_batch, target_batch, output in train_set:
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(input_batch.cuda(), target_batch.cuda())
        loss = criterion(outputs, output.cuda().view(-1))
        ppl = math.exp(loss.item())
        if (count_batch + 1) % 50 == 0:
            print('epoch', '%04d,' % (epoch + 1), 'step', f'{count_batch + 1}/{batch_number},',
                    'loss:', '{:.6f},'.format(loss.item()), 'ppl:', '{:.6}'.format(ppl))

        
        loss.backward()
        optimizer.step()
        count_batch += 1

    # 每轮执行一次校验
    model.eval()
    with torch.no_grad():
        total_loss = 0
        count_loss = 0
        for valid_batch, valid_target, out in zip(valid_enc_inputs, valid_dec_inputs, valid_dec_outputs):
            valid_output, _, _, _ = model(valid_batch, valid_target)
            valid_loss = criterion(valid_output, out)
            total_loss += valid_loss.item()
            count_loss += 1

        print(f'\nValidating at epoch', '%04d:' % (epoch + 1), 'loss:',
                '{:.6f},'.format(total_loss / count_loss),
                'ppl:', '{:.6}'.format(math.exp(total_loss / count_loss)))
    print('-' * 80)
    model.train()

    torch.save(model, '.\checkpoints\iwslt-de2en\Transformer.pkl')







