

import os
import pickle
from urllib.request import urlopen, FancyURLopener
from urllib.error import URLError

import numpy as np
import scipy
import pandas as pd

import outpyr.helpers as h
from outpyr import helpers_kremer
import outpyr.train_cpu_singleprocess, outpyr.train_cpu_multiprocess


def validate_web_url(url="http://google"):
    try:
        urlopen(url)
        return True
    except (URLError, ValueError):
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('data_file', metavar='data_file', type=str, nargs=1, help='file with count data')


    parser.add_argument("-s", "--settings", default=None)
    parser.add_argument("-p", "--preprocessing", choices=['normalize_sf'],
                        default=None)
    parser.add_argument("-n", "--nosubset", action='store_true')
    args = parser.parse_args()


















    if args.settings is not None:
        settings = h.load_settings(args.settings)
    else:
        import outpyr.settings as settings

    if validate_web_url(args.data_file[0]):
        if args.data_file[0].endswith('/'):
            data_file = args.data_file[0][:-1].rsplit('/', 1)[1]
        else:
            data_file = args.data_file[0].rsplit('/', 1)[1]

        if os.path.isfile(data_file):
            print('It seems that the provided URL was already downloaded to file %s. Skipping downloading.' % data_file)
        else:
            print('Downloading URL', args.data_file[0], '...')

            opener = FancyURLopener({})
            with opener.open(args.data_file[0]) as f:
                text = f.read().decode('utf-8')
            print('Finished!')

            with open('tmp.txt', 'w', encoding='utf-8') as f:
                f.write(text)



            df = h.csv_to_df('tmp.txt')
            if not args.nosubset and (list(df.index)) == helpers_kremer.INDEX_FULL and (list(df.columns)) == helpers_kremer.COLUMNS:
                print('Kremer dataset recognized, filtering genes...')
                df = df.loc[helpers_kremer.INDEX_FILTERED, :]
                print('Done!')
            h.save_df_to_csv(df, data_file)
            os.remove('tmp.txt')
    else:
        data_file = args.data_file[0]
    base_name, ext = os.path.splitext(os.path.basename(data_file))
    dir_ = os.path.abspath(os.path.dirname(data_file))

    pvalues_file = os.path.join(dir_, base_name + '-opx-pv.csv')

    if os.path.isfile(data_file):
        print('Running OutPyRX on', data_file, '...')
        gene_subset = None
        output_dir = outpyr.train_cpu_singleprocess.run(data_file, settings, args.preprocessing, gene_subset, 'p_j_r_j_numba')

        dir_abs = os.path.join(dir_, output_dir)
        from outpyr import helpers_tensorflow as htf
        ti = htf.TraceInspector(dir_abs)
        if 'p_values_mean' not in ti.v:
            print('Post-sampling: calculating p-values...')
            ti.set_final_values_from_trace()
            print('Done!')
            print('Saving p-values inside of the trace directory...')
            with open(os.path.join(ti.trace_data_folder, 'variables.pickle'), 'wb') as f_variables:
                pickle.dump(ti.v, f_variables)
            print('Finished!')

        print('Saving scores as CSV files...')
        delta__ = ti.v['r_avg_diff_matrix']/(ti.v['iteration'] + 1)
        zs__ = np.empty_like(delta__)
        for j in range(delta__.shape[0]):
            row = delta__[j, :]
            mu = row.mean()
            std = row.std()
            if std == 0:
                std = 0.000000000000001
            zs_row_ = (row - mu) / std
            _tmp = np.copy(zs_row_)
            _tmp[np.isinf(_tmp)] = 0
            _tmp[np.isneginf(_tmp)] = 0
            zs_row_[np.isinf(zs_row_)] = max(7, np.abs(_tmp).max())
            zs_row_[np.isneginf(zs_row_)] = min(-7, -np.abs(_tmp).max())
            zs__[j, :] = zs_row_
        pv__ = 2 * scipy.stats.norm.cdf(-np.abs(zs__))
        h.save_df_to_csv(pd.DataFrame(pv__, index=ti.df.index, columns=ti.df.columns), pvalues_file)






    else:
        parser.error("The file (%s) you provided does not exist." % data_file)


if __name__ == '__main__':
    main()
