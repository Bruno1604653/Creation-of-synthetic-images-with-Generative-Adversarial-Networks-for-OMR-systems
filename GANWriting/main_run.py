import os
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
from load_data import loadData as load_data_func
from network_tro import ConTranModel
from loss_tro import CER

parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

OOV = True

NUM_THREAD = 2

EARLY_STOP_EPOCH = None
EVAL_EPOCH = 20
MODEL_SAVE_EPOCH = 200
show_iter_num = 500
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 1  # Reducir tamaño del lote para pruebas
lr_dis = 1 * 1e-4
lr_gen = 1 * 1e-4
lr_rec = 1 * 1e-5

CurriculumModelID = args.start_epoch

def all_data_loader():
    train_loader, test_loader = load_data_func(OOV)
    train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def train(train_loader, model, dis_opt, gen_opt, rec_opt, epoch):
    model.train()
    loss_dis = list()
    loss_dis_tr = list()
    loss_rec = list()
    loss_rec_tr = list()
    time_s = time.time()
    cer_tr = CER()
    cer_te = CER()
    cer_te2 = CER()
    print(len(train_loader))
    for i, train_data_list in enumerate(train_loader):
        train_data_list = [data.to(device) for data in train_data_list]  # Mover los datos al dispositivo
        print(i)
        '''rec update'''
        rec_opt.zero_grad()
        l_rec_tr = model(train_data_list, epoch, 'rec_update', cer_tr)
        l_rec_tr.backward(retain_graph=True)
        rec_opt.step()

        '''dis update'''
        dis_opt.zero_grad()
        l_dis_tr = model(train_data_list, epoch, 'dis_update')
        l_dis_tr.backward(retain_graph=True)
        dis_opt.step()

        '''gen update'''
        gen_opt.zero_grad()
        l_total, l_dis, l_rec = model(train_data_list, epoch, 'gen_update', [cer_te, cer_te2])
        l_total.backward()
        gen_opt.step()

        loss_dis.append(l_dis.cpu().item())
        loss_dis_tr.append(l_dis_tr.cpu().item())
        loss_rec.append(l_rec.cpu().item())
        loss_rec_tr.append(l_rec_tr.cpu().item())

        # Limpiar memoria no utilizada
        del l_total, l_dis, l_rec, l_dis_tr, l_rec_tr
        torch.cuda.empty_cache()
        #if i == 5:
        #    break

    fl_dis = np.mean(loss_dis)
    fl_dis_tr = np.mean(loss_dis_tr)
    fl_rec = np.mean(loss_rec)
    fl_rec_tr = np.mean(loss_rec_tr)

    res_cer_tr = cer_tr.fin()
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print("A")
    print('epo%d <tr>-<gen>: l_dis=%.2f-%.2f, l_rec=%.2f-%.2f, cer=%.2f-%.2f-%.2f, time=%.1f' % (epoch, fl_dis_tr, fl_dis, fl_rec_tr, fl_rec, res_cer_tr, res_cer_te, res_cer_te2, time.time() - time_s))
    return res_cer_te + res_cer_te2

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
    with torch.no_grad():
        for test_data_list in test_loader:
            test_data_list = [data.to(device) for data in test_data_list]  # Mover los datos al dispositivo
            l_dis, l_rec = model(test_data_list, epoch, 'eval', cer_te)
            loss_dis.append(l_dis.cpu().item())
            loss_rec.append(l_rec.cpu().item())
    fl_dis = np.mean(loss_dis)
    fl_rec = np.mean(loss_rec)
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()
    print('epo%d <te>: l_dis=%.2f, l_rec=%.2f, cer=%.2f-%.2f, time=%.1f' % (epoch, fl_dis, fl_rec, res_cer_te, res_cer_te2, time.time() - time_s))
    return res_cer_te + res_cer_te2

def main(train_loader, test_loader):
    print(f"Device: {device}")
    model = ConTranModel(show_iter_num, OOV).to(device)
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

    for epoch in range(CurriculumModelID, epochs):
        cer = train(train_loader, model, dis_opt, gen_opt, rec_opt, epoch)

        if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights + '/contran-%d.model' % epoch)

        if epoch % EVAL_EPOCH == 0:
            test(test_loader, epoch, model)

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

def rm_old_model(index):
    models = glob.glob('save_weights/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < index:
            os.system('rm save_weights/contran-' + str(epoch) + '.model')

if __name__ == '__main__':
    print(time.ctime())
    train_loader, test_loader = all_data_loader()
    main(train_loader, test_loader)
    print(time.ctime())
