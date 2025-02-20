import torch
from utils import *
import copy
import torch.nn as nn

CUDA = torch.cuda.is_available()


def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            #print("q", x_batch.shape, y_batch.shape)

        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        # 打印模型参数的梯度
        # for name, param in net.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")
        #     else:
        #         print(f"Parameter: {name} has no gradient!")
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

def compute_class_weights(y_true, num_classes):
    """
    计算每个类别的权重，基于类别频率反转
    """
    epsilon = 1e-8
    class_counts = torch.bincount(y_true, minlength=num_classes).float() + epsilon
    total_samples = len(y_true)
    class_weights = total_samples / (num_classes * class_counts.float())
    return class_weights

def train(args, data_train, label_train, data_val, label_val, subject, fold):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_trial' + str(fold)
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    model = get_model(args)

    para = get_trainable_parameter_num(model)  # input_shape
    print('Model {} size:{}'.format(args.model, para))

    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # weight_decay=0.01
    # 设置为交叉熵损失
    # class_weights = torch.tensor([0.85, 0.45]).cuda()
    # class_weights = compute_class_weights(label_train, 2).cuda()
    loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()
    start_timer = time.time()

    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train, classes=args.labels)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val, classes=args.labels)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))


        if acc_val > trlog['max_acc']:
            trlog['max_acc'] = acc_val
            save_model('max-acc')

            if args.save_model:
                # save model here for reproduce
                model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
                data_type = 'model_{}_{}_{}'.format(args.dataset, args.data_format, args.label_type)
                save_path = osp.join(args.save_path, data_type)
                ensure_path(save_path)
                model_name_reproduce = osp.join(save_path, model_name_reproduce)
                torch.save(model.state_dict(), model_name_reproduce)

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))
    end_timer = time.time()
    # 计算总运行时间
    total_time = end_timer - start_timer
    print("total_time:", total_time)

    save_name_ = 'trlog' + save_name
    ensure_path(osp.join(args.save_path, 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'log_train', save_name_))

    return trlog['max_acc']


def test(args, data, label, reproduce, subject, fold):
    seed_all(args.random_seed)
    set_up(args)

    test_loader = get_dataloader(data, label, args.batch_size, False)

    model = get_model(args)
    if CUDA:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()

    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}_{}'.format(args.dataset, args.data_format, args.label_type)
        save_path = osp.join(args.save_path, data_type)
        ensure_path(save_path)
        model_name_reproduce = osp.join(save_path, model_name_reproduce)
        model.load_state_dict(torch.load(model_name_reproduce))
    else:
        model.load_state_dict(torch.load(args.load_path))
    s_time = time.time()
    loss, pred, act = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    t_time = time.time() - s_time
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act, classes=args.labels)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}, total time={:f}'.format(loss, acc, f1, t_time))
    return acc, pred, act


