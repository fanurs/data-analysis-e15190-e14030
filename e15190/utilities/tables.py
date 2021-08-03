from tabulate import tabulate

def to_fwf(df, path, drop_index=True, **tabulate_kwargs):
    """Write Pandas Dataframe into fixed-width file (FWF).

    Parameters
    ----------
    df : pandas.Dataframe object
        The dataframe to be written to file.
    path : str or pathlib.Path object
        The path to be written into fixed-width file.
    drop_index : bool, default True
        If `True`, the index will not be included; if `False`, the index will be
        written to file as one of the columns.
    tabulate_kwargs
        Keyword arguments for the tabulate function. See more at
        https://pypi.org/project/tabulate/
    """
    df = df.copy()
    if not drop_index:
        df.reset_index(inplace=True)

    kw = dict(tablefmt='plain')
    kw.update(tabulate_kwargs)
    content = tabulate(df.values.tolist(), list(df.columns), **kw)
    with open(path, 'w') as f:
        f.write(content + '\n')