from torch.utils.data import DataLoader

from rodnet.datasets.CRDataset import CRDataset
from rodnet.datasets.CRDatasetSM import CRDatasetSM
from rodnet.datasets.CRDataLoader import CRDataLoader
from rodnet.datasets.collate_functions import cr_collate

from rodnet.datasets.CRUW2022Dataset import CRUW2022Dataset
from rodnet.datasets.CRUW20223DDetDataset import CRUW20223DDetDataset


def get_dataloader(dataset_name, config_dict, args, dataset):
    batch_size = config_dict['train_cfg']['batch_size']
    if dataset_name == 'ROD2021':
        print("Building %s dataloader ... (Mode: %s)" % (dataset_name, "save_memory" if args.save_memory else "normal"))
        if not args.save_memory:
            crdata_train = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                     noise_channel=args.use_noise_channel)
            dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)

            # crdata_valid = CRDataset(os.path.join(args.data_dir, 'data_details'),
            #                          os.path.join(args.data_dir, 'confmaps_gt'),
            #                          win_size=win_size, set_type='valid', stride=8)
            # dataloader_valid = DataLoader(crdata_valid, batch_size=batch_size, shuffle=True, num_workers=0)

        else:
            crdata_train = CRDatasetSM(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                       noise_channel=args.use_noise_channel)
            dataloader = CRDataLoader(crdata_train, shuffle=True, noise_channel=args.use_noise_channel)

            # crdata_valid = CRDatasetSM(os.path.join(args.data_dir, 'data_details'),
            #                          os.path.join(args.data_dir, 'confmaps_gt'),
            #                          win_size=win_size, set_type='train', stride=8, is_Memory_Limit=True)
            # dataloader_valid = CRDataLoader(crdata_valid, batch_size=batch_size, shuffle=True)

    elif dataset_name == 'CRUW2022':
        print("Building %s dataloader ..." % dataset_name)
        crdata_train = CRUW2022Dataset(data_dir=args.data_root, dataset=dataset,
                                       config_dict=config_dict,
                                       split='train',
                                       noise_channel=args.use_noise_channel,
                                       old_normalize=args.use_old_norm)
        dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)

    elif dataset_name == 'CRUW2022_3DDet':
        print("Building %s dataloader ..." % dataset_name)
        crdata_train = CRUW20223DDetDataset(data_dir=args.data_root, dataset=dataset,
                                            config_dict=config_dict,
                                            split='train',
                                            noise_channel=args.use_noise_channel,
                                            old_normalize=args.use_old_norm)
        dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)

    else:
        raise NotImplementedError

    return crdata_train, dataloader


def get_dataloader_test(dataset_name, config_dict, args, dataset, subset=None):
    batch_size = 1
    if dataset_name == 'ROD2021':
        print("Building %s dataloader ..." % dataset_name)
        if args.demo:
            crdata_test = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='demo',
                                    noise_channel=args.use_noise_channel, subset=subset, is_random_chirp=False)
        else:
            crdata_test = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='test',
                                    noise_channel=args.use_noise_channel, subset=subset, is_random_chirp=False)
        dataloader = DataLoader(crdata_test, batch_size, shuffle=False, num_workers=0, collate_fn=cr_collate)

    elif dataset_name == 'CRUW2022':
        print("Building %s dataloader ..." % dataset_name)
        crdata_test = CRUW2022Dataset(data_dir=args.data_root, dataset=dataset,
                                      config_dict=config_dict,
                                      split='test',
                                      noise_channel=args.use_noise_channel,
                                      old_normalize=args.use_old_norm)
        dataloader = DataLoader(crdata_test, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)

    elif dataset_name == 'CRUW2022_3DDet':  # TODO: to be tested
        print("Building %s dataloader ..." % dataset_name)
        crdata_test = CRUW20223DDetDataset(data_dir=args.data_root, dataset=dataset,
                                           config_dict=config_dict,
                                           split='test',
                                           noise_channel=args.use_noise_channel,
                                           old_normalize=args.use_old_norm)
        dataloader = DataLoader(crdata_test, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)

    else:
        raise NotImplementedError

    return crdata_test, dataloader
