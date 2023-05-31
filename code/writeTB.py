import numpy as np

def write_tensorboard(loss_dict, acc, writer, epoch):
        # for loss in loss_dict.keys():
        #     writer.add_scalar(f"{loss}_{flag}", np.mean(loss_dict[loss]), epoch)
        writer.add_scalar("loss_train", (loss_dict["train"]), epoch)
        writer.add_scalar("loss_valid", (loss_dict["valid"]), epoch)
        writer.add_scalar("acc", acc, epoch)
        # writer.add_scalar('Loss/Classification', classification_loss, iteration)
        # writer.add_scalar('Loss/Regression', regression_loss, iteration)
        # writer.add_scalar('Loss/Running', running_loss, iteration)
        # writer.add_scalar('Accuracy/mAP', mAP, iteration)
        # writer.add_scalar("learning_rate", lr, epoch)
        # writer.add_scalar("learning_rate", np.mean(optimizer.param_groups[0]["lr"]), epoch)
        # writer.add_scalar("time", np.mean(time.time() - t), epoch)

        return