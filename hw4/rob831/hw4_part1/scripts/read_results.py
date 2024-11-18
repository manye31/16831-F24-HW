import glob
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    TRAIN = []
    EVAL = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                TRAIN.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                EVAL.append(v.simple_value)
    return TRAIN, EVAL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()


    logdirs = [args.logdir]
    if args.logdir.find(','):
        logdirs = args.logdir.split(',')

    train = []
    eval = []

    for dir in logdirs:
        test = dir.split('/')[-2] # requires trailing '/'
        logdir = os.path.join(dir, 'events*')
        eventfile = glob.glob(logdir)[0]
        print(eventfile)

        train, eval = get_section_results(eventfile)

        iterations = np.arange(1, len(train)+1, 1)
        for i, (x, y) in enumerate(zip(train, eval)):
            print('Iteration {:d} | Train Return: {} | Eval Return: {}'.format(i, x, y))
        
        plt.plot(iterations, train, label="Train")
        plt.plot(iterations, eval, label="Eval")
        plt.xlabel("Iterations")
        plt.ylabel("Average Return")
        plt.legend()
        plt.savefig(f"/home/micah/ri-classes/16831-F24-HW/hw4/figs/{test}.png")