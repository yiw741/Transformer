import torch
import torch.nn as nn
import math

class tokenembedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim, padding_idx):
        super(tokenembedding, self).__init__(vocab_size, embed_dim, padding_idx=padding_idx)

class positonembedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(positonembedding, self).__init__()
        self.em = torch.zeros(max_len, embed_dim)
        self.em.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        i = torch.arange(0, embed_dim, step=2).float()
        self.em[:, 0::2] = torch.sin(position / (10000 ** (i / embed_dim)))
        self.em[:, 1::2] = torch.cos(position / (10000 ** (i / embed_dim)))

    def forward(self, x):
        batch, seq_len = x.size()
        return self.em[:seq_len, :]

class transformerembeding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, padding_idx):
        super(transformerembeding, self).__init__()
        self.tok_emb = tokenembedding(vocab_size, embed_dim, padding_idx)
        self.pos_emb = positonembedding(max_len, embed_dim)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_t = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask):
        batch, size, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 分割为多个头
        q = q.view(batch, size, self.n_head, n_d).permute(0, 2, 1, 3)  # (batch, n_head, size, n_d)
        k = k.view(batch, size, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, size, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)  # (batch, n_head, size, size)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # 使用 -1e9 代替 -10000 以避免数值不稳定

        score = self.softmax(score) @ v  # (batch, n_head, size, n_d)
        score = score.permute(0, 2, 1, 3).contiguous()  # 重新排列为 (batch, size, n_head, n_d)
        score = score.view(batch, size, self.d_model)  # (batch, size, d_model)

        out = self.w_t(score)  # 这里的 score 应该是 (batch_size, seq_length, d_model)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class encoderlayer(nn.Module):
    def __init__(self, d_model, h_head, hidden):
        super(encoderlayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, h_head)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)
        self.ffo = PositionwiseFeedForward(d_model, hidden)

    def forward(self, x, mask):
        x1 = x
        x = self.attention(x, x, x, mask)
        x = self.dropout(x)
        x = self.norm(x + x1)
        x2 = x
        x = self.ffo(x)
        x = self.dropout(x)
        x = self.norm(x + x2)
        return x

class encoder(nn.Module):
    def __init__(self, d_model, vocab_size, embed_dim, max_len, padding_idx, h_head, hidden):
        super(encoder, self).__init__()
        self.embeding = transformerembeding(vocab_size, embed_dim, max_len, padding_idx)
        self.layers = nn.ModuleList()
        for _ in range(6):
            layer = encoderlayer(d_model, h_head, hidden)
            self.layers.append(layer)

    def forward(self, x, mask):
        x = self.embeding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class decoderlayer(nn.Module):
    def __init__(self, d_model, h_head, hidden):
        super(decoderlayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, h_head)
        self.corss_attention = MultiHeadAttention(d_model, h_head)
        self.ffo = PositionwiseFeedForward(d_model, hidden)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, dec, enc, mask):
        x1 = dec
        x = self.attention(x1, x1, x1, mask)
        x = self.dropout(x)
        x = self.norm(x + x1)
        x2 = enc
        x = self.corss_attention(x2, x2, x1, mask)
        x = self.dropout(x)
        x3 = self.norm(x2 + x)
        x = self.ffo(x3)
        x = self.dropout(x)
        x = self.norm(x + x3)
        return x

class decoder(nn.Module):
    def __init__(self, d_model, vocab_size, embed_dim, max_len, padding_idx, h_head, hidden):
        super(decoder, self).__init__()
        self.embedding = transformerembeding(vocab_size, embed_dim, max_len, padding_idx)
        self.layers = nn.ModuleList()
        for _ in range(6):
            layer = decoderlayer(d_model, h_head, hidden)
            self.layers.append(layer)

    def forward(self, enc, dec, mask1, mask2):
        x = self.embedding(dec)
        for layer in self.layers:
            x = layer(x, enc, mask1)
        return x

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 max_len,
                 h_head,
                 hidden):
        super(Transformer, self).__init__()
        self.encoder = encoder(
            d_model=d_model,
            vocab_size=enc_voc_size,
            embed_dim=d_model,
            max_len=max_len,
            padding_idx=src_pad_idx,
            h_head=h_head,
            hidden=hidden
        )
        self.decoder = decoder(
            d_model=d_model,
            vocab_size=dec_voc_size,
            embed_dim=d_model,
            max_len=max_len,
            padding_idx=trg_pad_idx,
            h_head=h_head,
            hidden=hidden
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        mask = q & k
        return mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(q.device)
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_casual_mask(trg, trg)

        enc = self.encoder(src, src_mask)
        out = self.decoder(enc, trg, trg_mask, src_mask)
        return out

if __name__ == '__main__':
    # 创建一个Transformer实例
    transformer = Transformer(
        src_pad_idx=0,
        trg_pad_idx=0,
        enc_voc_size=10000,
        dec_voc_size=10000,
        d_model=512,
        max_len=100,
        h_head=8,
        hidden=2048
    )

    # 创建一些随机输入
    src = torch.randint(0, 10000, (32, 50))
    trg = torch.randint(0, 10000, (32, 50))

    # 调用forward方法
    output = transformer(src, trg)

    # 打印输出
    print(output.shape)
