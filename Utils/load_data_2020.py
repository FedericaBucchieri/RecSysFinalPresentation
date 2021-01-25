def load_URM(file_path):
    import pandas as pd

    data = pd.read_csv(file_path)

    import scipy.sparse as sps

    user_list = data['row'].tolist()
    item_list = data['col'].tolist()
    rating_list = data['data'].tolist()

    return sps.coo_matrix((rating_list, (user_list, item_list))).tocsr()

def get_user_list(file_path):
    import pandas as pd
    import scipy.sparse as sps

    data = pd.read_csv(file_path)
    user_list = data['row']

    return user_list


def get_item_list(file_path):
    import pandas as pd
    import scipy.sparse as sps

    data = pd.read_csv(file_path)
    item_list = data['col']

    return item_list




def load_ICM(file_path):
    import pandas as pd

    data = pd.read_csv(file_path)

    import scipy.sparse as sps

    item_icm_list = data['row'].tolist()
    feature_list = data['col'].tolist()
    weight_list = data['data'].tolist()

    return sps.coo_matrix((weight_list, (item_icm_list, feature_list)))