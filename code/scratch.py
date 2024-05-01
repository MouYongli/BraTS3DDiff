import os


def delete_files(root,keep=10):

    def get_files_to_keep(files):
        keep_interval = int(len(files)/keep)
        keep_files = []
        for i,file in enumerate(files):
            if i % keep_interval == 0:
                keep_files.append(file)
        if not files[-1] in keep_files:
            keep_files.append(files[-1])
        return keep_files
        

    checkpoint_dir_path = os.path.join(root,'checkpoints')
    if os.path.exists(checkpoint_dir_path):
        checkpoints = os.listdir(checkpoint_dir_path)

        ema_files = []
        model_files = []
        opt_files = []
        for checkpoint in checkpoints:
            if checkpoint.startswith('ema'):
                ema_files.append(checkpoint)
            elif checkpoint.startswith('model'):
                model_files.append(checkpoint)
            elif checkpoint.startswith('opt'):
                opt_files.append(checkpoint)

        ema_files = sorted(ema_files)
        model_files = sorted(model_files)
        opt_files = sorted(opt_files)
        ema_files = [f for f in get_files_to_keep(ema_files)]
        model_files = [f for f in get_files_to_keep(model_files)]
        opt_files = [f for f in get_files_to_keep(opt_files)]
        keep_files = ema_files + model_files + opt_files

        for checkpoint in checkpoints:
            if checkpoint not in keep_files:
                checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint)
                os.remove(checkpoint_path)
                print('d')



delete_files(root='/DATA1/sanyal/diffusion-lightning-hydra-out/logs/train/runs/2024-04-29_13-06-00')