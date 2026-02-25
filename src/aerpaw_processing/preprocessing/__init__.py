from aerpaw_processing.resources.config.config_init import CONFIG, load_env

load_env()


def combine_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
