# -*- encoding: utf-8 -*-
from DataLoader.dataloader_3D import val_transform, read_data, pre_processing
from Evaluate.evaluate import *
from model import *


def online_evaluation(trainer):
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

            input_ = list_images[0]  # np.array
            gt_mask = list_images[1]  # (1, 16, 512, 512)
            # mask_original = list_images[2]

            # Forward
            [input_] = val_transform([input_])  # [input_] -> [torch.Tensor()]
            input_ = input_.unsqueeze(0).to(trainer.setting.device)  # (1, 1, 16, 512, 512)
            [_, prediction_B] = trainer.setting.network(input_)  # (1, 1, 16, 512, 512)
            prediction_B = np.array(prediction_B.cpu().data[0, :, :, :, :])

            # Post processing and evaluation
            # Post processing needed
            Dice_score = cal_subject_level_dice(prediction_B, gt_mask)
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
    # Evaluation score is the higher the better
    return - np.mean(list_Dice_score)
