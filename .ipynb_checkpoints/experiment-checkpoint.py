# skip test: data unavailable
from eelbrain.pipeline import *


class ImageNet(MneExperiment):

    ignore_entities = {
        'ignore_runs': ('2', '3'),
        #'ignore_sessions': 'mri',
    }

    raw = {
        'tsss': RawMaxwell('raw', st_duration=10., ignore_ref=True, st_correlation=0.9, st_only=True),
        '1-40': RawFilter('tsss', 1, 40),
        'ica': RawICA('1-40', 'ImageNet', n_components=0.99),
    }

    variables = {
        'event': LabelVar('trigger', {1: 'begin', 2: 'stim_on', 4: 'resp', 8: 'end'}),
    }

    epochs = {
        # 'target': PrimaryEpoch('ImageNet', "(event == 'stim_on')",
         # 'target': PrimaryEpoch ((vent == 'resp')", samplingrate=251.005),
        'stim_on': PrimaryEpoch('ImageNet', "event == 'stim_on'", samplingrate=200),
        # 'resp': SecondaryEpoch('target', "event == 'stim_on'"),
    }

    tests = {
        '=0': TTestOneSample(),
        'connection': TTestRelated('event', 'stim_on', 'resp'),
        'anova': ANOVA('event * subject'),
    }


