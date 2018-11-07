import argparse
import os

from dataset.generate_dataset import GenerateDataset
from trainer.MultitaskTrainer import MultitaskTrainer
from trainer.ClassificationTrainer import ClassificationTrainer


def load_config(config_path):
    config = open(config_path, 'r')
    config_args = {}
    for line in config:
        if line.find(' = ') != -1:
            name, value = line.split(' = ')
            config_args[name] = value.strip('\n')
    config.close()
    return config_args


def get_dataset_generator(args):
    dataset_gen = GenerateDataset(args['DATASET_PATH'])
    dataset_gen.load_dataset(args['EXISTING_DATASET'], float(args['TRAIN_SPLIT']))

    if args['INCLUDE_IGNORE'] == 'True':
        dataset_gen.add_ignore()

    return dataset_gen

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the multitask network')
    parser.add_argument('--config', nargs='?', help='Path of the config file', dest='config_path')
    args = parser.parse_args()

    config_args = load_config(args.config_path)

    dataset_generator = get_dataset_generator(config_args)

    use_k_fold = int(config_args['K_FOLD']) > 1

    if use_k_fold:
        dataset = dataset_generator.get_k_fold_dataset(int(config_args['K_FOLD']))
    else:
        dataset = dataset_generator.all_dataset(train_size=float(config_args['TRAIN_SIZE']), tree_size=float(config_args['TREE_SIZE']))
        dataset = {
            0: dataset
        }

    main_name = args.config_path.split('/')[-1]

    for i in range(int(config_args['K_FOLD'])):
        model_name = str(i)
        if config_args['USE_MULTITASK'] == 'TRUE':

            trainer = MultitaskTrainer(lr=float(config_args['LR']),
                                       batch_size=int(config_args['BATCH_SIZE']),
                                       epoch_lr=[int(x) for x in config_args['EPOCH_LIST'].strip('[]').split(', ')],
                                       lr_decay=float(config_args['LR_DECAY']),
                                       weight_decay=float(config_args['WEIGHT_DECAY']),
                                       n_classes=len(dataset_generator.classes),
                                       pretrained=config_args['PRETRAINED'] == 'TRUE')

        else:
            trainer = ClassificationTrainer(lr=float(config_args['LR']),
                                            batch_size=int(config_args['BATCH_SIZE']),
                                            epoch_lr=[int(x) for x in config_args['EPOCH_LIST'].strip('[]').split(', ')],
                                            lr_decay=float(config_args['LR_DECAY']),
                                            weight_decay=float(config_args['WEIGHT_DECAY']),
                                            n_classes=len(dataset_generator.classes),
                                            pretrained=config_args['PRETRAINED'] == 'TRUE')

        trainer.train(n_epoch=int(config_args['N_EPOCHS']), folder=dataset[i], model_name=config_args['MODEL'],
                      print_info=config_args['PRINT'] == 'TRUE')

        trainer.save_train_data(model_name,
                                os.path.join(config_args['LOG_PATH'], main_name),
                                args.config_path, dataset[i],
                                save_graph=config_args['SAVE_GRAPH'] == 'TRUE')
