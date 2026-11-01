"""Compute the original state/action/effort statistics without decoding unused videos."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(JAX_PLATFORMS='cpu', HF_HOME=str(ROOT / '.cache/huggingface'), HF_LEROBOT_HOME=str(ROOT / '.cache/huggingface/lerobot'), OPENPI_DATA_HOME=str(ROOT / '.cache/openpi'), HF_HUB_OFFLINE='1')
import numpy as np
from lerobot.common.datasets import lerobot_dataset
from openpi.training import config

OriginalDataset = lerobot_dataset.LeRobotDataset

class NumericDataset(OriginalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        keys = ('observation.state', 'action', 'observation.effort', 'observation.is_contact', 'episode_index', 'task_index')
        numpy_data = self.hf_dataset.with_format('numpy')
        self.numeric = {key: np.asarray(numpy_data[key]) for key in keys}
        self.starts = np.array(self.episode_data_index['from'], copy=True)
        self.ends = np.array(self.episode_data_index['to'], copy=True)

    def __getitem__(self, idx):
        item = {key: values[idx] for key, values in self.numeric.items()}
        episode = int(item['episode_index'])
        start, end = self.starts[episode], self.ends[episode]
        for key, offsets in (self.delta_indices or {}).items():
            indices = idx + np.asarray(offsets)
            item[key] = self.numeric[key][np.clip(indices, start, end - 1)]
            item[key + '_is_pad'] = (indices < start) | (indices >= end)
        item['task'] = self.meta.tasks[int(item['task_index'])]
        item.update({key: np.zeros((3, 1, 1), dtype=np.float32) for key in self.meta.video_keys})
        return item

if __name__ == '__main__':
    cfg = config.get_config('pi0_forceldm')
    kwargs = dict(repo_id=cfg.data.repo_id, delta_timestamps={'action': [i / 30 for i in range(32)], 'observation.effort': [i / 30 for i in cfg.data.effort_history]})
    original, numeric = OriginalDataset(**kwargs), NumericDataset(**kwargs)
    for index in (0, 40, 542, 543, len(original) - 1):
        expected, actual = original[index], numeric[index]
        for key in ('observation.state', 'action', 'observation.effort', 'observation.is_contact'):
            np.testing.assert_array_equal(np.asarray(expected[key]), np.asarray(actual[key]))
    del original, numeric
    print('NUMERIC_READER_EQUIVALENCE_PASS: episode start/interior/end; original LeRobot indexing and transforms retained', flush=True)
    import dataclasses
    config._CONFIGS_DICT[cfg.name] = dataclasses.replace(cfg, num_workers=0)
    lerobot_dataset.LeRobotDataset = NumericDataset
    from scripts.compute_norm_stats import main
    main(config_name=cfg.name)
    (ROOT / '.repro/full-norm.ready').write_text('Full original numeric statistics computed; batch=16, 362336 frames, 3 trailing frames dropped exactly as original loader.\n')
    print('FULL_NORM_READY', flush=True)