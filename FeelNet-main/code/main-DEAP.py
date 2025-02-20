from cross_validation import *
from prepare_data_DEAP import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--data-path', type=str, default='./data_preprocessed_python/')
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='V', choices=['A', 'V'])
    parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')  # 4
    parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--input-shape', type=tuple, default=(1, 32, 512))  #

    parser.add_argument('--data-format', type=str, default='raw_32', choices=['raw', 'base_remove', 'raw_32', 'raw_32_1', 'raw_32_2'])  #raw - raw_2, raw_32 - raw_32_4
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2021)
    parser.add_argument('--max-epoch', type=int, default=100)  # 10
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)

    ###FeelNet
    parser.add_argument('--adaptive_filter', type=bool, default=True)
    parser.add_argument('--labels', type=list, default=[0, 1])

    parser.add_argument('--save-path', default='E:/DEAP/save/')
    parser.add_argument('--load-path', default='E:/DEAP/save/max-acc.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='ERTNet', help='baseline = deepConvNet, shallowConvNet, eegNet, TSception, ERTNet, EEGTCNet, FeelNet')
    parser.add_argument('--T', type=int, default=20)  # 15
    parser.add_argument('--graph-type', type=str, default='O', choices=['TS', 'O'],
                        help='TS for the channel order of TSception, O for the original channel order')
    parser.add_argument('--hidden', type=int, default=32)


    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()

    sub_to_run = np.arange(args.subjects)
    # data-preprocegressing
    save_path = os.getcwd()
    data_type = 'data_{}_{}_{}'.format(args.data_format, args.dataset, args.label_type)
    save_path = osp.join(save_path, data_type)
    if not os.path.exists(save_path):
        pd = PrepareData(args)
        pd.run(sub_to_run, split=True, feature=False, expand=True)  # expand
    else:
        pass

    # train model
    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.n_fold_CV(subject=sub_to_run, fold=10, reproduce=args.reproduce)  # To do leave one trial out please set fold=40
