import os, sys
import pickle

import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(DIR, '../'))
print(__file__)
SEED = 100
import paddle

paddle.seed(SEED)
np.random.seed(SEED)


def torchRes():
    result = {}
    from torch import nn
    import torch
    import pytorch_VisionLAN.cfgs.cfgs_check as cfgs
    from pytorch_VisionLAN.train_LA import load_network
    from pytorch_VisionLAN.train_LA import generate_optimizer
    from pytorch_VisionLAN.train_LA import Train_or_Eval, flatten_label, _flatten, Zero_Grad, test
    from pytorch_VisionLAN.utils import Attention_AR_counter, cha_encdec
    torch.manual_seed(SEED)
    model = load_network()
    optimizer, optimizer_scheduler = generate_optimizer(model)
    criterion_CE = nn.CrossEntropyLoss().cuda()
    L1_loss = nn.L1Loss().cuda()
    # tools prepare
    train_acc_counter = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                             cfgs.dataset_cfgs['case_sensitive'])
    train_acc_counter_rem = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                                 cfgs.dataset_cfgs['case_sensitive'])
    train_acc_counter_sub = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                                 cfgs.dataset_cfgs['case_sensitive'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    # train
    loss_show = 0
    ratio_res = 0.5
    ratio_sub = 0.5
    best_acc = 0
    loss_ori_show = 0
    loss_mas_show = 0
    for iter, params in enumerate(params_):
        # data_prepare
        data = params['data']
        label = params['label']  # original string
        label_res = params['label_res']  # remaining string
        label_sub = params['label_sub']  # occluded character
        label_id = params['label_id']  # character index

        target = encdec.encode(label)
        target_res = encdec.encode(label_res)
        target_sub = encdec.encode(label_sub)
        Train_or_Eval(model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        label_flatten_res, length_res = flatten_label(target_res)
        label_flatten_sub, length_sub = flatten_label(target_sub)
        target, label_flatten, target_res, target_sub, label_flatten_res = target.cuda(), label_flatten.cuda(), target_res.cuda(), target_sub.cuda(), label_flatten_res.cuda()
        label_flatten_sub, label_id = label_flatten_sub.cuda(), label_id.cuda()
        # prediction
        text_pre, text_rem, text_mas, att_mask_sub = model(data, label_id, cfgs.global_cfgs['step'])
        if iter == 0:
            result['forword'] = [text_pre, text_rem, text_mas, att_mask_sub]
        if iter == 1:
            result['backword'] = [text_pre, text_rem, text_mas, att_mask_sub]
            return result
        # loss_calculation
        if cfgs.global_cfgs['step'] == 'LF_1':
            pre_ori, label_ori = train_acc_counter.add_iter(*params['train_acc_counter'])
            loss_ori = criterion_CE(*params['loss_ori'])
            loss = loss_ori
        else:
            pre_ori, label_ori = train_acc_counter.add_iter(*params['train_acc_counter'])
            pre_rem, label_rem = train_acc_counter_rem.add_iter(*params['train_acc_counter_rem'])
            pre_sub, label_sub = train_acc_counter_sub.add_iter(*params['train_acc_counter_sub'])

            loss_ori = criterion_CE(*params['loss_ori'])
            loss_res = criterion_CE(*params['loss_res'])
            loss_mas = criterion_CE(*params['loss_mas'])
            loss = loss_ori + loss_res * ratio_res + loss_mas * ratio_sub
            loss_ori_show += loss_res
            loss_mas_show += loss_mas
        # loss for display
        loss_show += loss
        if iter == 0:
            result['loss'] = loss
        # optimize
        Zero_Grad(model)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 20, 2)
        optimizer.step()
        best_train_acc, _ = train_acc_counter.show_test(best_acc)
        best_train_acc_rem, _ = train_acc_counter_rem.show_test(best_acc)
        best_train_acc_sub, _ = train_acc_counter_sub.show_test(best_acc)
        result['acc'] = [best_train_acc, best_train_acc_rem, best_train_acc_sub]
        optimizer_scheduler.step()


def paddleRes():
    result = {}
    import paddle
    from paddle import nn
    import paddle_VisionLAN.cfgs.cfgs_check as cfgs
    from paddle_VisionLAN.train_LA import load_network
    from paddle_VisionLAN.train_LA import generate_optimizer
    from paddle_VisionLAN.train_LA import Train_or_Eval, flatten_label, _flatten, Zero_Grad, test
    from paddle_VisionLAN.utils import Attention_AR_counter, cha_encdec
    from paddle_VisionLAN.api import clip_grad_norm_
    paddle.seed(SEED)
    model = load_network()
    optimizer, optimizer_scheduler = generate_optimizer(model)
    criterion_CE = nn.CrossEntropyLoss()
    L1_loss = nn.L1Loss()
    # tools prepare
    train_acc_counter = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                             cfgs.dataset_cfgs['case_sensitive'])
    train_acc_counter_rem = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                                 cfgs.dataset_cfgs['case_sensitive'])
    train_acc_counter_sub = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                                 cfgs.dataset_cfgs['case_sensitive'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    # train
    loss_show = 0
    ratio_res = 0.5
    ratio_sub = 0.5
    best_acc = 0
    loss_ori_show = 0
    loss_mas_show = 0
    for iter, params in enumerate(params_):
        # data_prepare
        data = params['data']
        label = params['label']  # original string
        label_res = params['label_res']  # remaining string
        label_sub = params['label_sub']  # occluded character
        label_id = params['label_id']  # character index

        target = encdec.encode(label)
        target_res = encdec.encode(label_res)
        target_sub = encdec.encode(label_sub)
        Train_or_Eval(model, 'Train')
        data = data
        label_flatten, length = flatten_label(target)
        label_flatten_res, length_res = flatten_label(target_res)
        label_flatten_sub, length_sub = flatten_label(target_sub)
        target, label_flatten, target_res, target_sub, label_flatten_res = target, label_flatten, target_res, target_sub, label_flatten_res
        label_flatten_sub, label_id = label_flatten_sub, label_id
        # prediction
        text_pre, text_rem, text_mas, att_mask_sub = model(data, label_id, cfgs.global_cfgs['step'])
        if iter == 0:
            result['forword'] = [text_pre, text_rem, text_mas, att_mask_sub]
        if iter == 1:
            result['backword'] = [text_pre, text_rem, text_mas, att_mask_sub]
            return result
        # loss_calculation
        if cfgs.global_cfgs['step'] == 'LF_1':
            pre_ori, label_ori = train_acc_counter.add_iter(*params['train_acc_counter'])
            loss_ori = criterion_CE(*params['loss_ori'])
            loss = loss_ori
        else:
            pre_ori, label_ori = train_acc_counter.add_iter(*params['train_acc_counter'])
            pre_rem, label_rem = train_acc_counter_rem.add_iter(*params['train_acc_counter_rem'])
            pre_sub, label_sub = train_acc_counter_sub.add_iter(*params['train_acc_counter_sub'])

            loss_ori = criterion_CE(*params['loss_ori'])
            loss_res = criterion_CE(*params['loss_res'])
            loss_mas = criterion_CE(*params['loss_mas'])
            loss = loss_ori + loss_res * ratio_res + loss_mas * ratio_sub
            loss_ori_show += loss_res
            loss_mas_show += loss_mas
        # loss for display
        loss_show += loss
        if iter == 0:
            result['loss'] = loss
        # optimize
        Zero_Grad(model)
        loss.backward()
        clip_grad_norm_(model.parameters(), 20, 2)
        optimizer.step()
        best_train_acc, _ = train_acc_counter.show_test(best_acc)
        best_train_acc_rem, _ = train_acc_counter_rem.show_test(best_acc)
        best_train_acc_sub, _ = train_acc_counter_sub.show_test(best_acc)
        result['acc'] = [best_train_acc, best_train_acc_rem, best_train_acc_sub]
        optimizer_scheduler.step()


def main():
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()

    pytorch_res = torchRes()
    paddle_res = paddleRes()

    for k in pytorch_res:
        print(type(pytorch_res[k]))
        reprod_log_1.add(k, pytorch_res[k])
        reprod_log_1.save(f"{k}_torch.npy")

        print(type(paddle_res[k]))
        reprod_log_2.add(k, paddle_res[k])
        reprod_log_2.save(f"{k}_paddle.npy")
        check(k)


def check(k):
    diff_helper = ReprodDiffHelper()
    info1 = diff_helper.load_info(f"{k}/{k}_torch.npy")
    info2 = diff_helper.load_info(f"{k}/{k}_paddle.npy")
    diff_helper.compare_info(info1, info2)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path=f"{f}/diff-{k}.txt")


def decode(_params_) -> list or dict:
    if isinstance(_params_, dict):
        params_ = {}
        for k, v in _params_.items():
            """[str(l, encoding="utf-8") for l in label] """
            k = str(k, encoding='utf-8')
            if isinstance(v, (list, tuple)):
                v = decode(v)
            elif isinstance(v, (bytes)):
                v = str(v, encoding="utf-8")
            elif isinstance(v, (dict)):
                v = decode(v)
            params_[k] = v
        return params_
    if isinstance(_params_, (list, tuple)):
        params_ = []
        for i, v in enumerate(_params_):
            if isinstance(v, (list, tuple)):
                v = decode(v)
            elif isinstance(v, (bytes)):
                v = str(v, encoding="utf-8")
            elif isinstance(v, (dict)):
                v = decode(v)
            params_.append(v)
        return params_


if __name__ == "__main__":
    with open('params.pkl', 'rb') as f:
        # params_ = pickle.load(f)
        _params_ = pickle.load(f, encoding='bytes')
    params_ = decode(_params_)
    main()
