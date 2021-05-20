from tabulate import tabulate

def to_fwf(df, path, drop_index=True):
    """Write Pandas Dataframe into fixed-width file (FWF).

    Parameters
    ----------
    df : pandas.Dataframe object
        The dataframe to be written to file.
    path : str or pathlib.Path object
        The path to be written into fixed-width file.
    drop_index : bool, default True
        If `True`, the index will not be included;
        if `False`, the index will be written to file as one of the columns.
    """
    df = df.copy()
    if not drop_index:
        df.reset_index(inplace=True)

    content = tabulate(df.values.tolist(), list(df.columns), tablefmt='plain')
    with open(path, 'w') as f:
        f.write(content + '\n')