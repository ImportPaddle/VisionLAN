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
