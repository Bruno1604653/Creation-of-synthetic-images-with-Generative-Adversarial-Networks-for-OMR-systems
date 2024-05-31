import numpy as np
import os
import torch
from torch import nn
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from vgg_tro_channel3_modi import vgg19_bn
from recognizer.models.encoder_vgg import Encoder as rec_encoder
from recognizer.models.decoder import Decoder as rec_decoder
from recognizer.models.seq2seq import Seq2Seq as rec_seq2seq
from recognizer.models.attention import locationAttention as rec_attention
from load_data import IMG_HEIGHT, IMG_WIDTH, vocab_size, index2letter, num_tokens, tokens
import cv2

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

def normalize(tar):
    min_val = tar.min()
    max_val = tar.max()
    
    if max_val != min_val:
        tar = (tar - min_val) / (max_val - min_val)
    else:
        tar = np.zeros_like(tar)  # Asigna una matriz de ceros si min y max son iguales
    
    tar = tar * 255
    tar = tar.astype(np.uint8)
    return tar

def fine(label_list):
    if type(label_list) != list:
        return [label_list]
    else:
        return label_list

def write_image(xg, pred_label, gt_img, gt_label, title):
    folder = 'imgs'
    if not os.path.exists(folder):
        os.makedirs(folder)
    batch_size = gt_label.shape[0]
    gt_img = gt_img.cpu().numpy()
    xg = xg.cpu().numpy()
    gt_label = gt_label.cpu().numpy()
    pred_label = torch.topk(pred_label, 1, dim=-1)[1].squeeze(-1)  # b,t,83 -> b,t,1 -> b,t
    pred_label = pred_label.cpu().numpy()

    outs = list()
    for i in range(batch_size):
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        gt = normalize(gt)
        tar = normalize(tar)
        gt_text = gt_label[i].tolist()
        pred_text = pred_label[i].tolist()

        gt_text = fine(gt_text)
        pred_text = fine(pred_text)

        for j in range(num_tokens):
            gt_text = list(filter(lambda x: x != j, gt_text))
            pred_text = list(filter(lambda x: x != j, pred_text))

        gt_text = ''.join([index2letter[c - num_tokens] for c in gt_text])
        pred_text = ''.join([index2letter[c - num_tokens] for c in pred_text])
        gt_text_img = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        cv2.putText(gt_text_img, gt_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(pred_text_img, pred_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out = np.vstack([gt, gt_text_img, tar, pred_text_img])
        outs.append(out)
    final_out = np.hstack(outs)
    cv2.imwrite(folder + '/' + title + '.png', final_out)

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

class DisModel(nn.Module):
    def __init__(self):
        super(DisModel, self).__init__()
        self.n_layers = 6
        self.final_size = 1024
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, padding=3, pad_type='reflect', norm='none', activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = min(nf * 2, 1024)
            cnn_f += [ActFirstResBlock(nf, nf, norm='none', activation='lrelu')]
            cnn_f += [ActFirstResBlock(nf, nf_out, norm='none', activation='lrelu')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
            nf = min(nf * 2, 1024)
        nf_out = min(nf * 2, 1024)
        cnn_f += [ActFirstResBlock(nf, nf, norm='none', activation='lrelu')]
        cnn_f += [ActFirstResBlock(nf, nf_out, norm='none', activation='lrelu')]

        self.cnn_f = nn.Sequential(*cnn_f)

        example_input = torch.randn(1, 1, 128, 128).to(device)  # Mover example_input al dispositivo correcto
        example_feat = self.cnn_f(example_input)
        flattened_size = np.prod(example_feat.shape[1:])

        #print(f"Flattened size: {flattened_size}")

        cnn_c = [
            nn.Flatten(),
            nn.Linear(flattened_size, self.final_size),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        self.cnn_c = nn.Sequential(*cnn_c)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        #print(f"x.shape in forward DisModel: {x.shape}")
        feat = self.cnn_f(x.to(device))
        #print(f"feat.shape after cnn_f: {feat.shape}")
        out = self.cnn_c(feat)
        #print(f"out.shape after cnn_c: {out.shape}")
        return out

    def calc_dis_fake_loss(self, input_fake):
        #print(f"input_fake.shape: {input_fake.shape}")
        label = torch.zeros(input_fake.shape[0], self.final_size).to(device)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

    def calc_dis_real_loss(self, input_real):
        #print(f"input_real.shape: {input_real.shape}")
        label = torch.ones(input_real.shape[0], self.final_size).to(device)
        resp_real = self.forward(input_real)
        real_loss = self.bce(resp_real, label)
        return real_loss

    def calc_gen_loss(self, input_fake):
        label = torch.ones(input_fake.shape[0], self.final_size).to(device)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

class GenModel_FC(nn.Module):
    def __init__(self):
        super(GenModel_FC, self).__init__()
        self.enc_image = ImageEncoder().to(device)
        self.dec = Decoder().to(device)

    def decode(self, content):
        images = self.dec(content)
        return images

    def forward(self, img):
        feat_xs = self.enc_image(img.to(device))  # Codifica la imagen de entrada
        #print(f"Encoded features shape: {feat_xs.shape}")
        generated_img = self.decode(feat_xs)  # Decodifica para generar la imagen
        #print(f"Generated image shape: {generated_img.shape}")
        return generated_img

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = vgg19_bn(False).to(device)
        self.output_dim = 512

    def forward(self, x):
        return self.model(x.to(device))

class Decoder(nn.Module):
    def __init__(self, ups=4, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, "none", activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.to(device))

class RecModel(nn.Module):
    def __init__(self, pretrain=False):
        super(RecModel, self).__init__()
        hidden_size_enc = hidden_size_dec = 512
        embed_size = 60
        self.enc = rec_encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False).to(device)
        self.dec = rec_decoder(hidden_size_dec, embed_size, vocab_size, rec_attention, None).to(device)
        self.seq2seq = rec_seq2seq(self.enc, self.dec, vocab_size).to(device)
        if pretrain:
            model_file = 'recognizer/save_weights/seq2seq-72.model_5.79.bak'
            self.seq2seq.load_state_dict(torch.load(model_file))

    def forward(self, img, label, img_width):
        self.seq2seq.train()
        img = torch.cat([img, img, img], dim=1).to(device)  # b,1,64,128->b,3,64,128
        output, attn_weights = self.seq2seq(img, label.to(device), img_width.to(device), teacher_rate=False, train=False)
        return output.permute(1, 0, 2)  # t,b,83->b,t,83

class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=4096, dim=256, n_blk=3, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1).to(device))

if __name__ == "__main__":
    # Prueba del generador
    gen_model = GenModel_FC().to(device)
    sample_img = torch.randn(128, 1, 128, 128).to(device)  # Ejemplo de entrada
    generated_img = gen_model(sample_img)
    #print(f"Generated image shape: {generated_img.shape}")

    # Prueba del discriminador
    dis_model = DisModel().to(device)
    output = dis_model(generated_img)
    #print(f"Output shape: {output.shape}")
