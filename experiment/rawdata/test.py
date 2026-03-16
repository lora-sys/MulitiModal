import pandas as pd 



if __name__ == "__main__":
    import numpy as np

    data = np.load('experiment/model/unified_dataset_100.npz', allow_pickle=True)
    lst = data.files

    for item in lst:
        print(item)
        print(data[item])

    import numpy as np
    d = np.load('experiment/model/unified_dataset_100.npz')
    print(list(d.keys()))
    for k in d:
        print(k, d[k].shape, d[k].dtype)
        

