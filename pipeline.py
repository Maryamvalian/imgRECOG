# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from experiment import ImageNet

root = '~/Data/ds005810'
e = ImageNet(root)
e.set(rej='', epoch='used')
# e.load_raw(preload=True)
# print(e.load_raw(preload=True))
print(e.load_evoked_stc(subjects=-1,parc="aparc+aseg"))
# print(e.load_test('connection', 0.3, 0.5, 0.05, data='meg', baseline=False, epoch='used', make=True))




# %%

# %%
