
from tools import *

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='running_save')
parser.add_argument('--open', type=str, default='r13-16-25_fmnist_SNN_fedDis')
parser.add_argument('--compare', type=str, default=None)
parser.add_argument('--fig', type=str, default=None)
parser.add_argument('--print', type=bool, default=False)
parser.add_argument('--deal', type=bool, default=False)
parser.add_argument('--single', type=str, default=None)
args = parser.parse_args()

dir = './'+ args.d + '/'
file_open = args.open
recorder = torch.load(dir + file_open)
record = recorder['record']

if args.print:
    if args.single is not None:
        if args.single in record[0]:
            print(args.single)
            for i in range(len(record)):
                print('%.6f' % (record[i][args.single], ))
        elif args.single == 'total':
            print(args.single)
            for i in range(len(record)):
                if 'upstream' in record[0]:
                    print('%d' % (record[i]['upstream']+record[i]['downstream'],))
                elif 'total_upstream' in record[0]:
                    print('%d' % (record[i]['total_upstream'] + record[i]['total_downstream'],))
                else:
                    print('保存的数据中无此项目。')
        else:
            print('保存的数据中无此项目。')

    else:
        print('best acc=%f, epoch=%d'
              % (recorder['best_acc'], recorder['best_epoch']))
        if 'learning_rate' in recorder:
            print('learning rate=%f' % (recorder['learning_rate']))
        if 'learning_rate_distillation' in recorder:
            print('learning rate of distillation=%f'% (recorder['learning_rate_distillation']))
        print('epoch\tacc\tupstream\tdownstream\ttotal')
        for i in range(len(record)):
            print('%d\t%.3f\t%d\t%d\t%d'%(i+1, record[i]['acc'], record[i]['upstream'], record[i]['downstream'],
                                          record[i]['upstream']+record[i]['downstream'],))
            #print('%.2f' % (record[i]['acc'], ))


if args.fig is not None:
    fig_dir = './result_save/' + args.fig + '.svg'
    epochs = []
    #accs = [0.1]
    # communications = [np.log(record[0]['downstream'] * 10e-6)]
    accs = []
    communications = []
    sc = 30

    if args.compare is not None:
        file_cp = args.compare
        recorder_cp = torch.load(dir + file_cp)
        record_cp = recorder_cp['record']

        # accs_cp = [0.1]
        # communications_cp = [np.log(record_cp[0]['downstream'] * 10e-6)]
        accs_cp = []
        communications_cp = []

    for i in range(sc):
        epochs.append(i+1)
        accs.append(record[i]['acc'])
        #communications.append(np.log10(record[i]['downstream'] * 10e-6))
        communications.append(record[i]['upstream'] * 10e-6)
        #if record[min(0, i - 1)]['acc'] == recorder['best_acc']:
        #    break

    if args.compare is not None:
        for i in range(sc):
            #if record_cp[i]['acc'] <= recorder['best_acc'] + 0.1:
                accs_cp.append(record_cp[i]['acc'])
                #communications_cp.append(np.log10(record_cp[i]['downstream'] * 10e-6))
                communications_cp.append(record_cp[i]['upstream'] * 10e-6)

    if not os.path.isdir('result_save'):
        os.mkdir('result_save')
    plt.figure()
    plt.title('MNIST')
    plt.ylabel("准确率 [%]")
    plt.xlabel("通信轮次")
    #plt.ylim((0.3, 1))
    plt.plot(epochs, accs, color='r')
    #plt.semilogx(communications, accs, color='r')
    #plt.semilogy(list(range(len(communications))), communications, color='r')

    if args.compare is not None:
        plt.plot(epochs, accs_cp, color='b')
        #plt.semilogx(communications_cp, accs_cp, color='b')
        #plt.semilogy(list(range(len(communications_cp))), communications_cp, color='b')

    #plt.legend(labels=['FSD','FedAvg'],loc='best')
    #plt.axhline(y=recorder['best_acc'] - 0.01, ls="--", c="k")  # 添加水平直线
    #plt.axvline(x=record_cp[0]['upstream'] * 10e-6, ls=":", c="k")  # 添加垂直直线
    plt.savefig(fig_dir, format='svg')