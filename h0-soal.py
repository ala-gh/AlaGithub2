#1
import numpy as np
idx_test = np.random.randint(0, 245056, 5000)
idx_train = list()

for i in range(245056):
    if not i in idx_test:
        idx_train.append(i)

print("len idx test:",len(idx_test))
print("len idx train",len(idx_train))

# ? chera len(idx_train) barabar ba (245056- 5000) nist ?

#**********************************************************************
