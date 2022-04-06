import matplotlib.pyplot as plt
import os


def plot_training_validation(
    train_loss, train_acc, val_loss, val_acc, experiment_name, PREFIX
):

    # plot the train_loss vs validation_loss in a 5 subplots
    fig, axs = plt.subplots(1, 2)

    axs[0, 0].plot(train_loss[0], label="train_loss")
    axs[0, 0].plot(val_loss[0], label="val_loss")
    axs[0, 0].set_title("Fold 1")

    # set the x axis as epoch and y axis as loss
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")

    axs[0, 1].plot(train_loss[1], label="train_loss")
    axs[0, 1].plot(val_loss[1], label="val_loss")
    axs[0, 1].set_title("Fold 2")

    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")

    # axs[0, 2].plot(train_loss[2], label="train_loss")
    # axs[0, 2].plot(val_loss[2], label="val_loss")
    # axs[0, 2].set_title("Fold 3")

    # axs[0, 2].set_xlabel("Epoch")
    # axs[0, 2].set_ylabel("Loss")

    # axs[1, 0].plot(train_loss[3], label="train_loss")
    # axs[1, 0].plot(val_loss[3], label="val_loss")
    # axs[1, 0].set_title("Fold 4")

    # axs[1, 0].set_xlabel("Epoch")
    # axs[1, 0].set_ylabel("Loss")

    # axs[1, 1].plot(train_loss[4], label="train_loss")
    # axs[1, 1].plot(val_loss[4], label="val_loss")
    # axs[1, 1].set_title("Fold 5")

    # axs[1, 1].set_xlabel("Epoch")
    # axs[1, 1].set_ylabel("Loss")

    # remove the subplot at the bottom right
    # fig.delaxes(axs[1, 2])

    # increase the spacing between subplots
    fig.tight_layout(pad=2.0)

    # make the graphs look nice
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    # increase the width of the image
    fig.set_size_inches(20, 10)

    # save the image and create folder if it doesn't exist
    plt.savefig(
        PREFIX
        + "logs/"
        + experiment_name
        + "/training_loss_"
        + experiment_name
        + ".png"
    )

    # clear the plot
    plt.clf()

    # do the same thing with train_acc and validation_acc
    fig, axs = plt.subplots(1, 2)

    axs[0, 0].plot(train_acc[0], label="train_acc")
    axs[0, 0].plot(val_acc[0], label="val_acc")
    axs[0, 0].set_title("Fold 1")

    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Accuracy")

    axs[0, 1].plot(train_acc[1], label="train_acc")
    axs[0, 1].plot(val_acc[1], label="val_acc")
    axs[0, 1].set_title("Fold 2")

    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy")

    # axs[0, 2].plot(train_acc[2], label="train_acc")
    # axs[0, 2].plot(val_acc[2], label="val_acc")
    # axs[0, 2].set_title("Fold 3")

    # axs[0, 2].set_xlabel("Epoch")
    # axs[0, 2].set_xlabel("Accuracy")

    # axs[1, 0].plot(train_acc[3], label="train_acc")
    # axs[1, 0].plot(val_acc[3], label="val_acc")
    # axs[1, 0].set_title("Fold 4")

    # axs[1, 0].set_xlabel("Epoch")
    # axs[1, 0].set_xlabel("Accuracy")

    # axs[1, 1].plot(train_acc[4], label="train_acc")
    # axs[1, 1].plot(val_acc[4], label="val_acc")
    # axs[1, 1].set_title("Fold 5")

    # axs[1, 1].set_xlabel("Epoch")
    # axs[1, 1].set_xlabel("Accuracy")

    # remove the subplot at the bottom right
    # fig.delaxes(axs[1, 2])

    # increase the spacing between subplots
    fig.tight_layout(pad=2.0)

    # make the graphs look nice
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    # increase the width of the image
    fig.set_size_inches(20, 10)

    # save the image

    plt.savefig(
        PREFIX + "logs/" + experiment_name + "/training_acc_" + experiment_name + ".png"
    )

    # clear the plot
    plt.clf()
