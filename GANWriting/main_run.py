import os
import torch
import glob
from torch import optim
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from torchvision import transforms
from load_data import loadData as load_data_func, vocab_size, IMG_WIDTH, IMG_HEIGHT, num_tokens
from network_tro import ConTranModel
from loss_tro import CER
from collections import Counter

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
scaler = GradScaler()
OOV = True

NUM_THREAD = 4

EARLY_STOP_EPOCH = None
EVAL_EPOCH = 2
MODEL_SAVE_EPOCH = 10
show_iter_num = 1500
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 28
lr_dis = 5e-5
lr_gen = 1e-5
lr_rec = 1e-5

CurriculumModelID = args.start_epoch
def all_data_loader():
    train_loader, test_loader = load_data_func(OOV)
    train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    balance_data_loader(train_loader)
    return train_loader, test_loader

def balance_data_loader(loader):
    labels = []
    for _, tr_label in loader:
        labels.extend(tr_label.cpu().numpy())
    counter = Counter(labels)
    print(f"Label distribution: {counter}")


def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    img1 = img1.detach().squeeze().cpu().numpy()  # Convert to numpy array
    img2 = img2.detach().squeeze().cpu().numpy()  # Convert to numpy array
    if img1.ndim == 2:  # If grayscale, add channel dimension
        img1 = img1[..., np.newaxis]
    if img2.ndim == 2:  # If grayscale, add channel dimension
        img2 = img2[..., np.newaxis]

    return ssim(img1, img2, channel_axis=-1, win_size=7, data_range=img1.max() - img1.min())

def train(train_loader, model, dis_opt, gen_opt, rec_opt, epoch, log_file):
    model.train()
    loss_dis = list()
    loss_dis_tr = list()
    loss_rec = list()
    loss_rec_tr = list()
    time_s = time.time()
    cer_tr = CER()
    cer_te = CER()
    cer_te2 = CER()
    ssim_scores = []

    for i, train_data_list in enumerate(train_loader):
        print(i)
        tr_img, tr_label = train_data_list
        tr_img = tr_img.to(device)
        tr_label = tr_label.to(device)

        #debug_train_data((tr_img, tr_label), "Original")

        train_data_list = (tr_img, tr_label)

        '''rec update'''
        rec_opt.zero_grad()
        l_rec_tr = model(train_data_list, epoch, 'rec_update', cer_tr)
        scaler.scale(l_rec_tr).backward(retain_graph=True)
        scaler.step(rec_opt)
        scaler.update()

        '''dis update'''
        dis_opt.zero_grad()
        l_dis_tr = model(train_data_list, epoch, 'dis_update')
        scaler.scale(l_dis_tr).backward(retain_graph=True)
        scaler.step(dis_opt)
        scaler.update()

        '''gen update'''
        gen_opt.zero_grad()
        l_total, l_dis, l_rec = model(train_data_list, epoch, 'gen_update', [cer_te, cer_te2])
        scaler.scale(l_total).backward(retain_graph=True)
        scaler.step(gen_opt)
        scaler.update()

        loss_dis.append(l_dis.cpu().item())
        loss_dis_tr.append(l_dis_tr.cpu().item())
        loss_rec.append(l_rec.cpu().item())
        loss_rec_tr.append(l_rec_tr.cpu().item())
        # valores de gradientes para detectar anomalias
        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        pass
                #print(f'{name} gradient mean: {param.grad.mean()}, gradient max: {param.grad.max()}, gradient min: {param.grad.min()}')

        # Chequear valores de pérdida
        #if torch.isnan(l_rec_tr).any() or torch.isinf(l_rec_tr).any():
        #    print(f'l_rec_tr: Anomalous loss detected in rec_update at batch {i}')
        #if torch.isnan(l_dis_tr).any() or torch.isinf(l_dis_tr).any():
        #    print(f'ls_dis_tr: Anomalous loss detected in dis_update at batch {i}')
        #if torch.isnan(l_total).any() or torch.isinf(l_total).any():
        #    print(f'ls_total: Anomalous loss detected in gen_update at batch {i}')

        # Calcular SSIM
        with torch.no_grad():
            # Generar las imágenes
            generated_imgs = model.gen(tr_img)

            # Calcular SSIM para cada par de imágenes
            for gen_img, ref_img in zip(generated_imgs, tr_img):
                ssim_score = compute_ssim(gen_img, ref_img)
                ssim_scores.append(ssim_score)

        # Limpiar memoria no utilizada
        del l_total, l_dis, l_rec, l_dis_tr, l_rec_tr
        torch.cuda.empty_cache()

    fl_dis = np.mean(loss_dis)
    fl_dis_tr = np.mean(loss_dis_tr)
    fl_rec = np.mean(loss_rec)
    fl_rec_tr = np.mean(loss_rec_tr)
    avg_ssim = np.mean(ssim_scores)

    res_cer_tr = cer_tr.fin()
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()

    log_message = 'epo%d <tr>-<gen>: l_dis=%.2f-%.2f, l_rec=%.2f-%.2f, cer=%.2f-%.2f-%.2f, ssim=%.4f, time=%.1f' % (
        epoch, fl_dis_tr, fl_dis, fl_rec_tr, fl_rec, res_cer_tr, res_cer_te, res_cer_te2, avg_ssim, time.time() - time_s)

    print(log_message)
    with open(log_file, "a") as f:
        f.write(log_message + '\n')
    return res_cer_te + res_cer_te2, avg_ssim

def test(test_loader, epoch, modelFile_o_model):
    if type(modelFile_o_model) == str:
        model = ConTranModel(show_iter_num, OOV).to(device)
        print('Loading ' + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model, map_location=device))
    else:
        model = modelFile_o_model
    model.eval()
    loss_dis = list()
    loss_rec = list()
    time_s = time.time()
    cer_te = CER()
    cer_te2 = CER()
    ssim_scores = []
    for test_data_list in test_loader:
        tr_img, tr_label = test_data_list
        tr_img = tr_img.to(device)
        tr_label = tr_label.to(device)

        test_data_list = (tr_img, tr_label)

        l_dis, l_rec = model(test_data_list, epoch, 'eval', cer_te)
        loss_dis.append(l_dis.cpu().item())
        loss_rec.append(l_rec.cpu().item())

        # Calcular SSIM
        with torch.no_grad():
            # Generar las imágenes
            generated_imgs = model.gen(tr_img)

            # Calcular SSIM para cada par de imágenes
            for gen_img, ref_img in zip(generated_imgs, tr_img):
                ssim_score = compute_ssim(gen_img, ref_img)
                ssim_scores.append(ssim_score)
        del l_dis, l_rec
        torch.cuda.empty_cache()

    fl_dis = np.mean(loss_dis)
    fl_rec = np.mean(loss_rec)
    avg_ssim = np.mean(ssim_scores)
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print('epo%d <te>: l_dis=%.2f, l_rec=%.2f, cer=%.2f-%.2f, ssim=%.4f, time=%.1f' % (epoch, fl_dis, fl_rec, res_cer_te, res_cer_te2, avg_ssim, time.time() - time_s))
    return res_cer_te + res_cer_te2, avg_ssim

def main(train_loader, test_loader):
    print(f"Device: {device}")
    model = ConTranModel(show_iter_num, OOV).to(device)
    print(model)
    if CurriculumModelID > 0:
        model_file = 'save_weights/contran-' + str(CurriculumModelID) + '.model'
        print('Loading ' + model_file)
        model.load_state_dict(torch.load(model_file))

    dis_params = list(model.dis.parameters())
    gen_params = list(model.gen.parameters())
    rec_params = list(model.rec.parameters())
    dis_opt = optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)
    epochs = 50001
    min_cer = 1e5
    min_idx = 0
    min_count = 0

    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"execution_{time.strftime('%Y%m%d-%H%M%S')}.txt")

    for epoch in range(CurriculumModelID, epochs):
        cer, avg_ssim = train(train_loader, model, dis_opt, gen_opt, rec_opt, epoch, log_file)

        with open(log_file, "a") as f:
            f.write(f'epo{epoch} <tr>: cer={cer:.2f}, ssim={avg_ssim:.4f}\n')

        if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights + '/contran-%d.model' % epoch)

        if epoch % EVAL_EPOCH == 0:
            test_cer, test_ssim = test(test_loader, epoch, model)
            with open(log_file, "a") as f:
                f.write(f'epo{epoch} <te>: cer={test_cer:.2f}, ssim={test_ssim:.4f}\n')

        if EARLY_STOP_EPOCH is not None:
            if min_cer > cer:
                min_cer = cer
                min_idx = epoch
                min_count = 0
                rm_old_model(min_idx)
            else:
                min_count += 1
            if min_count >= EARLY_STOP_EPOCH:
                print('Early stop at %d and the best epoch is %d' % (epoch, min_idx))
                model_url = 'save_weights/contran-' + str(min_idx) + '.model'
                os.system('mv ' + model_url + ' ' + model_url + '.bak')
                os.system('rm save_weights/contran-*.model')
                break

    with open(log_file, "a") as f:
        f.write(f'Final results: Best CER: {min_cer:.2f} at epoch {min_idx}\n')

def rm_old_model(index):
    models = glob.glob('save_weights/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < index:
            os.system('rm save_weights/contran-' + str(epoch) + '.model')

if __name__ == '__main__':
    print(f"Network_tro.py vocab_size: {vocab_size}")
    print(time.ctime())
    train_loader, test_loader = all_data_loader()
    main(train_loader, test_loader)
    print(time.ctime())


