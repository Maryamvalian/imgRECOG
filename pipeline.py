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
#e.set(rej='')
# print(e.load_evoked(subjects=-1, data='meg'))
#print(e.load_test('connection', 0.3, 0.5, 0.05, data='meg', baseline=False, epoch='target', make=True))


e.show_subjects()



# %%

# %%
