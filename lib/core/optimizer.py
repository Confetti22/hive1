import torch.optim as Opt

def get_optimizer(model, args):

    opt_fns = {
        'adam': Opt.Adam(model.parameters(), lr = args.lr_start, weight_decay=args.weight_decay),
        'sgd': Opt.SGD(model.parameters(), lr = args.lr_start, weight_decay=args.weight_decay),
        'adagrad': Opt.Adagrad(model.parameters(), lr = args.lr_start, weight_decay=args.weight_decay)
    }
    return opt_fns.get(args.solver_name, "Invalid Optimizer")
