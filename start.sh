{
            "name": "Python: annotation 2",
            "type": "python",
            "request": "launch",
            "program": "one_bit_annotation_stage2.py",
            "console": "integratedTerminal",
            "args": [
                "--dataset=cifar100",
                "--query-quota=3000"
            ]
        },
        {
            "name": "Python: annotation 1",
            "type": "python",
            "request": "launch",
            "program": "one_bit_annotation_stage1.py",
            "console": "integratedTerminal",
            "args": [
                "--dataset=cifar100",
                "--query-quota=3000"
            ]
        },
        {
            "name": "Python: stage2",
            "type": "python",
            "request": "launch",
            "program": "main_stage2.py",
            "console": "integratedTerminal",
            "args": [
                "--train-subdir=C:/Users/Alex/WorkSpace/pycharm/mymoco/datasets",
                "--test-subdir=C:/Users/Alex/WorkSpace/pycharm/mymoco/datasets",
                "--arch=cifar_shakeshake26",
                "--labeled-batch-size=320",
                "-b=512",
                "--epochs=180",
                "--lr=0.2",
                "--lr-rampdown-epochs=210",
                "--nesterov=true",
                "--ema-decay=0.97",
                "--dataset=cifar100",
                "--consistency=1000",
                "--consistency-rampup=5",
                "--logit-distance-cost=0.01"
            ]
        },
        {
            "name": "Python: stage1",
            "type": "python",
            "request": "launch",
            "program": "main_stage1.py",
            "console": "integratedTerminal",
            "args": [
                "--train-subdir=C:/Users/Alex/WorkSpace/pycharm/mymoco/datasets",
                "--test-subdir=C:/Users/Alex/WorkSpace/pycharm/mymoco/datasets",
                "--arch=cifar_shakeshake26",
                "--labeled-batch-size=200",
                "-b=512",
                "--epochs=180",
                "--lr=0.2",
                "--lr-rampdown-epochs=210",
                "--nesterov=true",
                "--ema-decay=0.97",
                "--dataset=cifar100",
                "--consistency=1000",
                "--consistency-rampup=5",
                "--logit-distance-cost=0.01",
                "--resume=C:/Users/Alex/WorkSpace/pycharm/one-bit-supervision/checkpoint/stage1/checkpoint.ckpt"
            ]
        },
        {
            "name": "Python: stage0",
            "type": "python",
            "request": "launch",
            "program": "main_stage0.py",
            "console": "integratedTerminal",
            "args": [
                "--train-subdir=C:/Users/Alex/WorkSpace/pycharm/mymoco/datasets",
                "--test-subdir=C:/Users/Alex/WorkSpace/pycharm/mymoco/datasets",
                "--arch=cifar_shakeshake26",
                "--labeled-batch-size=50",
                "-b=512",
                "--epochs=180",
                "--lr=0.2",
                "--lr-rampdown-epochs=210",
                "--nesterov=true",
                "--ema-decay=0.97",
                "--dataset=cifar100",
                "--consistency=1000",
                "--consistency-rampup=5",
                "--logit-distance-cost=0.01",
            ],
        }