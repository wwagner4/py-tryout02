import pandas as pd
import matplotlib.pyplot as plt

"""
     clf             features  recall  precission      f1
0     lr             mean_all  0.9301      0.9299  0.9298
1     lr        mean_most_rel  0.9371      0.9369  0.9369
2     lr         mean_two_rel  0.8951      0.8949  0.8924
3     lr         mean_one_rel  0.8182      0.8180  0.8154
4     lr               se_all  0.8741      0.8772  0.8707
5     lr          se_most_rel  0.8951      0.8977  0.8922
6     lr           se_two_rel  0.7902      0.8097  0.7770
....
"""


def bar(df: pd.DataFrame, legend=False, values="recall", is_fs=True, is_class=True):
    p1 = df.pivot(
        index="clf",
        columns="features",
        values=values
    )
    if is_class:
        print(p1)
        p1.plot.bar(legend=legend, ylim=(0.8, 1.0), figsize=(10, 3),
                    title="{} : classifiers / feature sets".format(values))
        ax1 = plt.axes()
        x_axis = ax1.axes.get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)
        plt.show()

    if is_fs:
        p2 = df.pivot(
            index="features",
            columns="clf",
            values=values
        )
        print(p2)
        p2.plot.bar(legend=legend, ylim=(0.8, 1.0), figsize=(10, 3),
                    title="{} : feature sets / classifiers".format(values))
        ax1 = plt.axes()
        x_axis = ax1.axes.get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)
        plt.show()


def df_filter(df: pd.DataFrame, clfs, fs):
    df_tmp = df.loc[df['clf'].isin(clfs)]
    return df_tmp.loc[df_tmp['features'].isin(fs)]

def df_filter_clf(df: pd.DataFrame, clfs):
    return df.loc[df['clf'].isin(clfs)]

def df_filter_fs(df: pd.DataFrame, fs):
    return df.loc[df['features'].isin(fs)]


fs_rel = ["worst_most_rel", "all", "se_all", "se_two_rel"]

clf_rel = ["lr", "lda", "nm_g", "xgb", "gb"]

fs_rel1 = ["worst_most_rel", "all"]

clf_rel1 = ["lda", "xgb"]

_data = pd.read_csv('../tmp/result.csv', header=0)

_data1 = df_filter_fs(_data, fs_rel)
_data2 = df_filter(_data, clf_rel, fs_rel1)
_data3 = df_filter(_data, clf_rel1, fs_rel1)

bar(_data1)
bar(_data2, legend=True, is_fs=False)
bar(_data3, legend=True, is_fs=False)
bar(_data3, legend=True, values="precission", is_fs=False)
bar(_data3, legend=True, values="f1", is_fs=False)
