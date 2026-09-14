"""FisherWiki machine-learning pipeline.

Layout::

    env.py               hardware detection and platform workarounds
    data.py              dataset, augmentation policy, sampling
    models.py            backbones and hierarchical heads
    train_loop.py        training with real resume
    calibration_data.py  real images for INT8 calibration

Entry points live one level up: ``ml/train.py``, ``ml/evaluate.py``,
``ml/export.py``.
"""
