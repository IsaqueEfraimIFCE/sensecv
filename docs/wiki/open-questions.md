# Open Questions

## Validation

- Has the PyTorch port been numerically compared against the original Keras model on the same input image? The implementation is designed for parity, but this wiki only found smoke tests, not a saved parity report.
- Are the supermarket clips labeled with ground truth expected actions or obstacle states? Without labels, the generated CSVs are descriptive model outputs, not accuracy measurements.
- In the [[pilotguru-10fps-folder-check]], should STOP clips be judged by collision probability only, by steering plus collision, or by a separate hand-labeled stop/avoidance target? The current check uses collision probability because DroNet has no explicit STOP output.

## Data Quality

- Why do clips `06` and `07` have identical aggregate statistics in `summary.json`? Check whether they are duplicate videos, duplicate exports, or coincidentally identical after preprocessing.
- Should clip `01` be recovered or removed from future runs? It is reported as a 261-byte broken/placeholder file.

## Domain Adaptation

- Would camera orientation correction or aspect-ratio-preserving preprocessing reduce the out-of-distribution effect?
- Would recalibrating the collision threshold on supermarket data make the collision output more useful?
- Should indoor examples be collected and labeled for a small validation set?
- Why did the 10 FPS pass produce no RIGHT aggregate predictions even for folders whose names expected `desvio_direita`? Possible causes include domain shift, camera orientation, sign convention mismatch, label interpretation, or preprocessing mismatch.

## Wiki Maintenance

- Add pages for training data and loss functions if future work touches `dronet/repo/cnn.py` and `dronet/repo/utils.py` in detail.
- Add a parity page if a Keras-vs-PyTorch comparison becomes available.
