import torch
from torch import nn
import random

##print_shape_flag = False

cpu = torch.device('cpu')
cuda = torch.device('cuda')
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, src, tar, src_len, teacher_rate, train=True):
        tar = tar.permute(1, 0)  # time_s, batch
        batch_size = src.size(0)
        outputs = torch.zeros(1, batch_size, self.vocab_size, device=device, requires_grad=True)  # Ajuste para una sola salida
        
        out_enc, hidden_enc = self.encoder(src, src_len)
        output = self.one_hot(tar[0].detach())
        hidden = hidden_enc
        attn_weights = torch.zeros(out_enc.shape[1], out_enc.shape[0], requires_grad=True).to(device)

        output, hidden, attn_weights = self.decoder(output, hidden, out_enc, src_len, attn_weights)

        # En lugar de la operación in-place, creamos una nueva variable outputs
        new_outputs = torch.zeros_like(outputs)
        new_outputs[0] = output
        outputs = new_outputs

        return outputs, attn_weights

    def one_hot(self, src):
        ones = torch.eye(self.vocab_size).to(device)
        return ones.index_select(0, src)




