import os

import h5py

base_dir = os.path.dirname(os.path.abspath(__file__))
weights = os.environ.get("DRONET_WEIGHTS", os.path.join(base_dir, "repo", "model", "model_weights.h5"))
f = h5py.File(weights, "r")
print("=== root attrs ===")
for k in f.attrs:
    v = f.attrs[k]
    print(k, "=", v)

print("\n=== datasets (name -> shape) ===")
def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name, obj.shape)
f.visititems(show)
f.close()
