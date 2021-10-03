from tabulate import tabulate

def to_fwf(
    df,
    path,
    drop_header=False,
    drop_index=True,
    comment=None,
    **tabulate_kwargs,
):
    """Write Pandas Dataframe into fixed-width file (FWF).

    Parameters
    ----------
    df : pandas.Dataframe
        The dataframe to be written to file.
    path : str or pathlib.Path
        The path to be written into fixed-width file.
    drop_header : bool, default False
        If `True`, table header will be dropped; if `False`, dataframe's columns
        will be used as header.
    drop_index : bool, default True
        If `True`, the index will not be included; if `False`, the index will be
        written to file as one of the columns.
    comment : str or None
        If None, no comment will be written to file. Otherwise, the comment will
        be written to the beginning of the file, in lines before the table.
    **tabulate_kwargs
        Keyword arguments for the tabulate function. See more at
        https://pypi.org/project/tabulate/
    """
    df = df.copy()
    if not drop_index:
        df.reset_index(inplace=True)

    header = [] if drop_header else list(df.columns)
    kw = dict(tablefmt='plain')
    kw.update(tabulate_kwargs)
    content = tabulate(df.values.tolist(), header, **kw)
    with open(path, 'w') as f:
        if comment is None:
            pass
        else:
            f.write(comment.strip() + '\n')

        f.write(content + '\n')