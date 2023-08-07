import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')


parser.add_argument('--dir_name', type=str, default='00001/',help='save_dir')
parser.add_argument('--n_input_features', type=int, default=3, help='num of features per sample')
parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to train (default: 500)')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--decay_steps', type=int, default=300)
parser.add_argument('--eval_frequency', type=int, default=100)
parser.add_argument('--regularization', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=1)
parser.add_argument('--learning_rate', type=float, default=0.0000001)
parser.add_argument('--decay_rate', type=float, default=0.998)
parser.add_argument('--momentum', type=float, default=0.98)
parser.add_argument('--F', default=[16,32,64])
parser.add_argument('--K', default=[5,5,5])
parser.add_argument('--p', default=[2,2,2])
parser.add_argument('--M', default=[128,1])
parser.add_argument('--brelu', type=str,  default='b1relu',  help='Relu function')
parser.add_argument('--pool', type=str, default='mpool1', help='Pooling')
parser.add_argument('--filter', type=str, default='chebyshev5')

args = parser.parse_args()
