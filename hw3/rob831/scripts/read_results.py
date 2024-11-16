import argparse
import glob
import os
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        # if len(X) > 120:
        #     break
    return X, Y

if __name__ == '__main__':
    """
    Include a learning curve plot showing the performance of your implemen-
    tation on LunarLander-v3. The x-axis should correspond to number of time steps (consider using scientific
    notation) and the y-axis should show the average per-epoch reward as well as the best mean reward so far.
    These quantities are already computed and printed in the starter code. They are also logged to the data
    folder, and can be visualized using Tensorboard as in previous assignments. Be sure to label the y-axis, since
    we need to verify that your implementation achieves similar reward as ours. You should not need to modify
    the default hyperparameters in order to obtain good performance, but if you modify any of the parameters,
    list them in the caption of the figure. Compare the performance of DDQN to vanilla DQN. Since there is
    considerable variance between runs, you must run at least three random seeds for both DQN and DDQN. (To
    be clear, there should be one line for DQN and one line for DDQN on the same plot, each with error bars).
    The final results should use the following experiment names:

    - (time step, (avg_return, best_return))
    - avg of DQN with error bars std 1 line, avg of DDQN with error bars std 1 line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()


    logdirs = [args.logdir]
    if args.logdir.find(','):
        logdirs = args.logdir.split(',')

    steps = []
    avg = []
    best = []

    for dir in logdirs:
        test = dir.split('/')[-2] # requires trailing '/'
        logdir = os.path.join(dir, 'events*')
        # import pdb; pdb.set_trace()
        eventfile = glob.glob(logdir)[0]
        print(eventfile)

        X, Y = get_section_results(eventfile)
        steps.append(X[1:])
        avg.append(Y)
        best_for_experiment = []
        # manual best return
        for data in Y:
            best_for_experiment.append(np.max([*best_for_experiment, data]))
        best.append(best_for_experiment)
    steps = np.asarray(steps)
    avg = np.asarray(avg)
    best = np.asarray(best)

    dqn_steps = steps[0]
    dqn_avg = avg[:3]
    dqn_best = best[:3]
    ddqn_steps = steps[3]
    ddqn_avg = avg[3:]
    ddqn_best = best[3:]

    # Plot 
    fig, ax = plt.subplots()
    ax.errorbar(dqn_steps, np.average(dqn_avg, axis=0), yerr=np.std(dqn_avg), capsize=5.0, color='r', label='DQN Avg')
    ax.errorbar(dqn_steps, np.average(ddqn_avg, axis=0), yerr=np.std(ddqn_avg), capsize=5.0, color='b', label='DDQN Avg')
    # ax.legend()
    # ax.grid(True)
    # ax.set_xlabel("Time step")
    # ax.set_ylabel("Average Return")
    # fig.savefig("/home/micah/ri-classes/16831-F24-HW/hw3/figs/dqn_ddqn_average_return_eval.png")

    # fig, ax = plt.subplots()
    ax.errorbar(dqn_steps, np.average(dqn_best, axis=0), yerr=np.std(dqn_best), capsize=5.0, color='orange', label='DQN Best')
    ax.errorbar(dqn_steps, np.average(ddqn_best, axis=0), yerr=np.std(ddqn_best), capsize=5.0, color='g', label='DDQN Best')
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Return")
    fig.savefig("/home/micah/ri-classes/16831-F24-HW/hw3/figs/dqn_ddqn_return_eval.png")
    # ax.set_ylabel("Best Return")
    # fig.savefig("/home/micah/ri-classes/16831-F24-HW/hw3/figs/dqn_ddqn_best_return_eval.png")
    plt.close()
    