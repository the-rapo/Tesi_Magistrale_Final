def terna_preprox(file: str, folderout: str = 'IO', verbose: bool = False) -> None:
    """
        Effettua il pre-processo dei dati Terna.

        :param folderout: Cartella di output
        :param file: File da processare
        :param verbose: Flag per console-output

    """

    # LIBs
    import pandas as pd
    from pathlib import Path
    from cust_libs.misc import mk_dir
    #

    filename = Path(file).stem
    print('---------- ' 'Dataset ' + filename + 'PrePROX' + ' ---------- ')
    data = pd.read_excel(file, engine="openpyxl", sheet_name=0)
    data.drop('Forecast Total load [MW]', inplace=True, axis=1)
    data.drop('Bidding zone', inplace=True, axis=1)
    data.rename(columns={'Date': 'DateTime', 'Total Load [MW]': 'Terna_Load'}, inplace=True)
    data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
    data = data.dropna(subset=['DateTime'])
    # Max_Daily
    rel_load_daily = []
    day_start = 0
    for i in range(1, data.shape[0], 1):
        if data['DateTime'].iloc[i].day != data['DateTime'].iloc[i - 1].day:
            max_day = max(data['Terna_Load'].iloc[day_start: i])
            rel_load_daily[day_start: i] = data['Terna_Load'].iloc[day_start: i] / max_day
            day_start = i
        if i == data.shape[0] - 1:
            max_day = max(data['Terna_Load'].iloc[day_start:])
            rel_load_daily[day_start: i + 1] = data['Terna_Load'].iloc[day_start:] / max_day
    data['Terna_Load_Rel'] = rel_load_daily
    mk_dir(folderout, False, verbose)
    mk_dir(folderout + '/Terna_Load', False, verbose)
    if verbose:
        print('Salvando ' + folderout + '/Terna_Load/' + filename + '_Processed.csv')
    data.to_csv(folderout + '/Terna_Load/' + filename + '_Processed.csv', index=False)

    return
