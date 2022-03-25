from train_sl import TrainClassifier
# config_file = "configs/SL/config_glister-warm_cifar10.py"
config_file = "configs/SL/config_random_cifar10.py"
classifier = TrainClassifier(config_file)
classifier.cfg.dss_args.fraction = 0.01
<<<<<<< HEAD
classifier.cfg.dss_args.select_every = 20 #20
=======
classifier.cfg.dss_args.select_every = 5
>>>>>>> ff8686775a540bb6f2d193fda4072584b7215454
classifier.cfg.train_args.device = 'cuda'
classifier.cfg.train_args.print_every = 1
classifier.train()
