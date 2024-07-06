import argparse
import pandas as pd
from scipy import stats


def main(args):

    data_path = args.data_path
    file_name = data_path.split("/")[-1]
    pv_threshold = args.pval

    df = pd.read_csv(data_path)

    df_results = pd.DataFrame(columns=['Delta', 'Original', 'Pval', 'Reject', 'CI_low','CI_high'])

    grouped_df = df.groupby('Original',sort=False)

    for key, item in grouped_df:
        delta = grouped_df.get_group(key)['Delta'].unique()
        if len(delta)  > 1:
            raise
        else:
            delta = delta[0]
        # Compute p value and confidence intervals
        res = stats.ttest_1samp(grouped_df.get_group(key)['New'], popmean=key)
        ci = res.confidence_interval(confidence_level=0.95)
        df_results.loc[key] = [delta, key, res.pvalue, int(res.pvalue < pv_threshold), ci.low, ci.high]

           
    df_results.to_csv(f"{args.log_path}/ttest_{file_name}", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Confidence intervals generation script.')
    parser.add_argument('data_path', metavar='dp', default=None, help='Path to original results.')
    parser.add_argument('log_path', metavar='dp', default=None, help='Path to save new results.')
    parser.add_argument('--pval', type=float, default=0.05, help='Threshold for pvalue. Default: 0.05.')


    args = parser.parse_args()

    main(args)