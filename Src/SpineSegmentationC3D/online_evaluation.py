# -*- encoding: utf-8 -*-
from DataLoader.dataloader_3D import val_transform, read_data, pre_processing
from Evaluate.evaluate import *
from model import *
from Loss.SegLoss.DiceLoss import SoftDiceLoss


def online_evaluation(trainer):

    Spine_Segmentation = '../../Data/Spine_Segmentation'
    cases = sorted(os.listdir(Spine_Segmentation))
    list_case_dirs = [os.path.join(Spine_Segmentation, cases[i]) for i in range(120, 151)]
    list_SoftDice_score = []
    list_subject_level_dice_score = []
    # val_loader = trainer.setting.val_loader

    with torch.no_grad():
        trainer.setting.network.eval()

        for case_dir in list_case_dirs:
            case_id = case_dir.split('/')[-1]

            dict_images = read_data(case_dir)
            list_images = pre_processing(dict_images)
            list_images = val_transform(list_images)  # to tensor

            input_ = list_images[0].to(trainer.setting.device)  # MR tensor: (1, 16, H, W) 'cuda:0'
            gt_mask = list_images[1].to(trainer.setting.device)   # Mask tensor: (1, 16, H, W) 'cuda:0'
            # mask_original = list_images[2]

            # Forward
            # [input_] = val_transform([input_])  # [input_] -> [torch.tensor()]
            input_ = input_.unsqueeze(0)  # (1, 1, 16, H, W)
            gt_mask = gt_mask.unsqueeze(0)  # (1, 1, 16, H, W)
            [_, prediction_B] = trainer.setting.network(input_)  # tensor: (1, 20, 16, H, W)

            SoftDice_score = SoftDiceLoss()(prediction_B, gt_mask).cpu().numpy()  # negative value
            list_SoftDice_score.append(SoftDice_score)

            # Post processing and evaluation
            # FIXME convert the prediction to img, Post processing needed
            prediction_B = np.array(prediction_B.cpu())  # numpy: (1, 20, 16, H, W)
            gt_mask = np.array(gt_mask.cpu())
            prediction_B = np.argmax(prediction_B, axis=1).astype(np.int16)  # (1, 16, H, W)
            subject_level_dice_score = cal_subject_level_dice(prediction_B[0], gt_mask[0][0])  # positive value
            list_subject_level_dice_score.append(subject_level_dice_score)

            try:
                trainer.print_log_to_file('========> ' + case_id + ':  ' + 'SoftDice: ' + str(SoftDice_score) +
                                          ', subject_level_dice_score: ' + str(subject_level_dice_score),
                                          'a')
            except:
                pass

    try:
        trainer.print_log_to_file('===============================================> mean SoftDice score: '
                                  + str(np.mean(list_SoftDice_score)), 'a')
        trainer.print_log_to_file('===============================================> mean subject_level_dice score: '
                                  + str(-np.mean(list_subject_level_dice_score)), 'a')
    except:
        pass
    # Evaluation score is the lower the better
    return -np.mean(list_subject_level_dice_score)  # negative value
