"""Check explicit sequence windows and legacy preprocessing without retraining."""
import tempfile
from pathlib import Path
import joblib
import numpy as np
import run_sequences as seq
from cmapps.utils import digest, write_json
from cmapps.evaluation import threshold
from train_all import ROOT, DATASETS, load_inputs

seq.tf.config.threading.set_intra_op_parallelism_threads(2)
seq.tf.config.threading.set_inter_op_parallelism_threads(1)
train, test, _ = load_inputs()
processor = joblib.load(ROOT/'outputs/tcn_full/preprocessor.joblib')
assert not hasattr(processor, 'window')  # Old saved objects need no migration.
small = test[test.engine_id.eq(1)].copy()
a30, i30 = processor.windows(small, window=30)
a50, i50 = processor.windows(small, window=50)
again, indices = processor.windows(small, window=30)
np.testing.assert_array_equal(i30, i50)
np.testing.assert_array_equal(indices, i30)
np.testing.assert_array_equal(again, a30)
np.testing.assert_array_equal(a50[:, -30:], a30)

# Per-engine prefix oracle catches future use, padding and engine mixing.
for dataset in DATASETS:
    history = small[small.dataset.eq(dataset)].sort_values('cycle')
    for length in [1, 30, 50]:
        for observed in [1, 23, min(55, len(history))]:
            prefix = history.iloc[:observed]
            endpoint = prefix.index[-1]
            expected = np.zeros((length, 26), dtype=np.float32)
            rows = processor.transform(prefix)[-length:]
            expected[-len(rows):] = rows
            actual, returned = processor.windows(history, [endpoint], window=length)
            np.testing.assert_array_equal(actual[0], expected)
            np.testing.assert_array_equal(returned, [endpoint])
            changed = history.copy()
            changed.loc[changed.index[observed:], seq.SENSORS] = 1e6
            changed['RUL'] = -999
            np.testing.assert_array_equal(processor.windows(changed, [endpoint], window=length)[0], actual)
for bad in [0, -1, 2.5, True]:
    try:
        processor.windows(small, window=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Accepted invalid window {bad}')
try:
    processor.windows(small)
except TypeError:
    pass
else:
    raise AssertionError('Window must be explicit')
for length in [30, 50, 30]:
    model = seq.network('TCN', 26, 42, window=length)
    assert model.input_shape == (None, length, 26)
    assert model.count_params() == 26241
assert threshold([10, 20, 80], [15, 35, 1]) == 35
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)/'record.json'
    write_json(path, {'complete': True})
    before = digest(path)
    write_json(path, {'complete': True})
    assert digest(path) == before and not path.with_suffix('.json.tmp').exists()
print('PASS: legacy preprocessing, interleaved 30/50 windows, prefix oracle, padding, engine/future/target isolation, explicit model shapes and shared helpers.')
