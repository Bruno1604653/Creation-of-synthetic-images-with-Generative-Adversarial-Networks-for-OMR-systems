import torch
import torch.nn.functional as F
from torch import nn
from load_data import loadData as load_data_func, vocab_size, IMG_WIDTH, IMG_HEIGHT
from modules_tro import GenModel_FC, DisModel, RecModel, write_image
from loss_tro import recon_criterion, crit, log_softmax
import numpy as np
from torchvision import transforms

w_dis = 1.
w_l1 = 0.
w_rec = 1.

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

class ConTranModel(nn.Module):
    def __init__(self, show_iter_num, oov):
        super(ConTranModel, self).__init__()
        self.gen = GenModel_FC().to(device)
        self.dis = DisModel().to(device)
        self.rec = RecModel(vocab_size=vocab_size, pretrain=False).to(device)
        self.iter_num = 0
        self.show_iter_num = show_iter_num
        self.oov = oov

    def forward(self, train_data_list, epoch, mode, cer_func=None):
        tr_img, tr_label = train_data_list
        #print(f"tr_img: {tr_img}")
        tr_img = tr_img.to(device)
        #print(f"tr_label: {tr_label}")
        tr_label = tr_label.to(device)

        if tr_label.dim() == 1:
            tr_label = tr_label.unsqueeze(1)

        batch_size = tr_img.shape[0]

        if mode == 'gen_update':
            self.iter_num += 1
            generated_img = self.gen(tr_img)
            l_dis = self.dis.calc_gen_loss(generated_img)

            pred_xt = self.rec(generated_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
            l_rec = crit(log_softmax(pred_xt.reshape(-1, vocab_size)), tr_label.reshape(-1))
            if cer_func:
                cer_func[0].add(pred_xt, tr_label)
                cer_func[1].add(pred_xt, tr_label)

            l_total = w_dis * l_dis + w_rec * l_rec
            l_total.backward(retain_graph=True)
            return l_total, l_dis, l_rec

        elif mode == 'dis_update':
            sample_img = tr_img
            l_real = self.dis.calc_dis_real_loss(sample_img)
            l_real.backward(retain_graph=True)

            with torch.no_grad():
                generated_img = self.gen(tr_img)

            l_fake = self.dis.calc_dis_fake_loss(generated_img)
            l_fake.backward(retain_graph=True)

            gp = self.dis.gradient_penalty(sample_img, generated_img)
            gp.backward(retain_graph=True)

            l_total = l_real + l_fake
            if self.iter_num % self.show_iter_num == 0:
                with torch.no_grad():
                    pred_xt = self.rec(generated_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
                write_image(generated_img, pred_xt, tr_img, tr_label, 'epoch_' + str(epoch) + '-' + str(self.iter_num))
            return l_total

        elif mode == 'rec_update':
            self.iter_num += 1
            generated_img = self.gen(tr_img)
            pred_xt = self.rec(generated_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
            log_softmax_pred_xt = log_softmax(pred_xt.reshape(-1, vocab_size))
            l_rec = crit(log_softmax_pred_xt, tr_label.reshape(-1))
            if cer_func:
                cer_func.add(pred_xt, tr_label)
            l_rec.backward(retain_graph=True)
            return l_rec

        elif mode == 'eval':
            with torch.no_grad():
                generated_img = self.gen(tr_img)
                pred_xt = self.rec(generated_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
                write_image(generated_img, pred_xt, tr_img, tr_label, 'eval_' + str(epoch) + '-' + str(self.iter_num))
                self.iter_num += 1
                l_dis = self.dis.calc_gen_loss(generated_img)
                l_rec = crit(log_softmax(pred_xt.reshape(-1, vocab_size)), tr_label.reshape(-1))
                if cer_func:
                    cer_func.add(pred_xt, tr_label)
            return l_dis, l_rec

