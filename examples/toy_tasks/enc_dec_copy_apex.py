import tqdm
import torch
import torch.optim as optim
from performer_pytorch import PerformerEncDec
from apex import amp

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 16 + 2
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 64 + 1

# helpers

def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1]).bool().cuda()
        yield (src, tgt, src_mask, tgt_mask)

# instantiate model

model = PerformerEncDec(
    dim=512,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=1,
    enc_heads=8,
    enc_max_seq_len=ENC_SEQ_LEN,
    enc_reversible=True,
    enc_feature_redraw_interval=1000,
    enc_nb_features = 64,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=3,
    dec_heads=8,
    dec_max_seq_len=DEC_SEQ_LEN,
    dec_reversible=True,
    dec_feature_redraw_interval=1000,
    dec_nb_features=64
).cuda()

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# amp

model, optim = amp.initialize(model, optim, opt_level = 'O1')

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, tgt, src_mask, tgt_mask = next(cycle())
    loss = model(src, tgt, enc_mask=src_mask, dec_mask=tgt_mask)

    with amp.scale_loss(loss, optim) as scaled_loss:
        scaled_loss.backward()

    print(f'{i}: {loss.item()}')
    optim.step()
    optim.zero_grad()

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, _, src_mask, _ = next(cycle())
        src, src_mask = src[:1], src_mask[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        sample = model.generate(src, start_tokens, ENC_SEQ_LEN, enc_mask=src_mask)
        incorrects = (src != sample).abs().sum()

        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")
