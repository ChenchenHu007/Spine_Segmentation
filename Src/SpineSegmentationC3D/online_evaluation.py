# -*- encoding: utf-8 -*-
from DataLoader.dataloader_3D import val_transform, read_data, pre_processing
from Evaluate.evaluate import *
from model import *
from utils.tools import one_hot_to_img
from Loss.SegLoss.DiceLoss import SoftDiceLoss


def online_evaluation(trainer):
    # FIXME use val_loader
    Spine_Segmentation = '../../Data/Spine_Segmentation'
    cases = sorted(os.listdir(Spine_Segmentation))
    list_case_dirs = [os.path.join(Spine_Segmentation, cases[i]) for i in range(120, 151)]
    list_Dice_score = []
    # val_loader = trainer.setting.val_loader

    with torch.no_grad():
        trainer.setting.network.eval()

        for case_dir in list_case_dirs:
            case_id = case_dir.split('/')[-1]

            dict_images = read_data(case_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]  # MR (1, 16, 256, 256)
            gt_mask = list_images[1]  # Mask (1, 16, 256, 256)
            # mask_original = list_images[2]

            # Forward
            [input_] = val_transform([input_])  # [input_] -> [torch.tensor()]
            input_ = input_.unsqueeze(0).to(trainer.setting.device)  # (1, 1, 16, 256, 256)
            [_, prediction_B] = trainer.setting.network(input_)  # tensor: (1, 20, 16, 256, 256)

            # Post processing and evaluation
            # FIXME convert the prediction to img, Post processing needed
            # prediction_B = np.array(prediction_B.cpu().data[0, :, :, :, :])  # numpy: (20, 16, 256, 256)
            # prediction_B = one_hot_to_img(prediction_B)  # (16, 256, 256)
            # Dice_score = cal_subject_level_dice(prediction_B, gt_mask[0])

            Dice_score = SoftDiceLoss()(prediction_B, gt_mask)  # negative value
            list_Dice_score.append(Dice_score)

            try:
                trainer.print_log_to_file('========> ' + case_id + ':  ' + str(Dice_score), 'a')
            except:
                pass

    try:
        trainer.print_log_to_file('===============================================> mean Dice score: '
                                  + str(np.mean(list_Dice_score)), 'a')
    except:
        pass
    # Evaluation score is the lower the better
    return np.mean(list_Dice_score)  # negative value
