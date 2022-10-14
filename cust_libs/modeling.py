"""
##### MODELING V2 #####
In questa libreria sono presenti una serie di funzioni relative allla modellazione, divise nelle sezioni:
    - METRICHE
    - PLOTTING
    - AUTOML

"""

# METRICHE


def RMSE(model, x_test, y_test, parameters=None, x_transform=None):
    """
    Calcola la radice dello scarto quadratico medio ("RMSE") del modello specificato a partire dal database
    specificato in x_test, y_test

    :param model: modello di cui eseguire il plot
    :param x_test: features dati storici
    :param y_test: risultato dati storici
    :param parameters: specifico per modelli di regressione custom [optz.]
    :param x_transform: specifico per modelli che lo presentano [optz.]
    """
    # LIBs
    from sklearn.metrics import mean_squared_error
    #
    if parameters is None:
        if x_transform is not None:
            x_test = x_transform.transform(x_test)
        y_pred = model.predict(x_test)
    else:
        y_pred = model(x_test, *parameters)

    err = mean_squared_error(y_test, y_pred, squared=False)
    return err


def MSE(model, x_test, y_test, parameters=None, x_transform=None):
    """
    Calcola l'errore quadratico medio ("MSE") del modello specificato a partire dal database specificato
    in x_test, y_test

    :param model: modello di cui eseguire il plot
    :param x_test: features dati storici
    :param y_test: risultato dati storici
    :param parameters: specifico per modelli di regressione custom [optz.]
    :param x_transform: specifico per modelli che lo presentano [optz.]
    """
    # LIBs
    from sklearn.metrics import mean_squared_error
    #
    if parameters is None:
        if x_transform is not None:
            x_test = x_transform.transform(x_test)
        y_pred = model.predict(x_test)
    else:
        y_pred = model(x_test, *parameters)
    err = mean_squared_error(y_test, y_pred, squared=True)
    return err


def MAE(model, x_test, y_test, parameters=None, x_transform=None):
    """
    Calcola l'errore medio assoluto (MAE) del modello specificato a partire dal database specificato in x_test, y_test

    :param model: modello di cui eseguire il plot
    :param x_test: features dati storici
    :param y_test: risultato dati storici
    :param parameters: specifico per modelli di regressione custom [optz.]
    :param x_transform: specifico per modelli che lo presentano [optz.]
    """
    # LIBs
    from sklearn.metrics import mean_absolute_error
    #
    if parameters is None:
        if x_transform is not None:
            x_test = x_transform.transform(x_test)
        y_pred = model.predict(x_test)
    else:
        y_pred = model(x_test, *parameters)
    err = mean_absolute_error(y_test, y_pred)
    return err


# PLOTTING


def RMSE_MAE_plot(model, data_full, parameters=None, x_transform=None, title=None, Single_Param=False, add_text=True,
                  save=None):
    """
    Esegue il plot dell'errore in funzione della potenza relativa.

    :param model: modello di cui eseguire il plot
    :param title: titolo del grafico [optz.]
    :param data_full: database per analisi errore
    :param parameters: specifico per modelli di regressione custom [optz.]
    :param x_transform: specifico per modelli che lo presentano [optz.]
    :param Single_Param: Flag per analisi monovariata [False]
    :param add_text: Aggiunge il testo riguardo l'errore per P>0.45
    :param save: Piuttosto che visualizzare l'immagine la salva nel percorso specificato
        """
    # LIBs
    import pandas as pd
    import numpy as np
    from cust_libs.data_processing import filter_data
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    mpl.rcParams["font.size"] = 18
    #
    data = pd.read_csv(data_full)
    pwr = np.linspace(0, 1, 21, endpoint=True)
    pwr = [round(num, 3)for num in pwr]
    labels = []
    error_rmse = []
    error_mae = []
    n_elem = []
    mta = filter_data(data, None, None, 0.45, 1)
    if Single_Param:
        x_tot = data['PwrTOT_rel'].values
        x_tot = x_tot.reshape(-1, 1)
        x_mta = mta['PwrTOT_rel'].values
        x_mta = x_mta.reshape(-1, 1)
    else:
        x_tot = data[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
        x_mta = mta[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
    y_tot = data['Rendimento'].values
    y_mta = mta['Rendimento'].values
    avg_rmse = RMSE(model, x_tot, y_tot, parameters, x_transform)
    avg_mae = MAE(model, x_tot, y_tot, parameters, x_transform)
    mta_rmse = RMSE(model, x_mta, y_mta, parameters, x_transform)
    mta_mae = MAE(model, x_mta, y_mta, parameters, x_transform)

    RMSE_text = 'Avg. RMSE = ' "{:.4f}".format(avg_rmse)
    MAE_text = ' Avg. MAE = ' "{:.4f}".format(avg_mae)
    RMSE_text_mta = 'RMSE (P > 0.45) = ' "{:.4f}".format(mta_rmse)
    MAE_text_mta = 'MAE (P > 0.45) = ' "{:.4f}".format(mta_mae)

    for i in range(1, 21, 1):
        filtered = filter_data(data, None, None, pwr[i-1], pwr[i])
        if Single_Param:
            x = filtered['PwrTOT_rel'].values
            x = x.reshape(-1, 1)
        else:
            x = filtered[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
        y = filtered['Rendimento'].values

        if filtered.empty:
            continue
        else:
            n_elem.append(filtered.shape[0])
            error_rmse.append(RMSE(model, x, y, parameters, x_transform))
            error_mae.append(MAE(model, x, y, parameters, x_transform))
            labels.append(str(pwr[i - 1]) + ' - ' + str(pwr[i]))

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(range(len(error_rmse)), error_rmse, color='red', linewidth=2, marker='o', label='RMSE')
    ax.plot(range(len(error_rmse)), error_mae, color='blue', linewidth=2, marker='^', label='MAE')
    ax.axhline(y=avg_rmse, color='red', linestyle=(0, (5, 10)), label='RMSE_avg')
    ax.axvline(x=9, color='black', linestyle='dashed', linewidth=1, alpha=0.5)
    ax.text(3, avg_rmse + 0.0005, RMSE_text, fontsize='medium', color='red', ha='left')
    ax.text(3, avg_mae + 0.0003, MAE_text, fontsize='medium', color='blue',  ha='left')
    if add_text:
        ax.text(11, 0.0150, RMSE_text_mta, fontsize='medium', color='red', ha='left')
        ax.text(11, 0.0120, MAE_text_mta, fontsize='medium', color='blue',  ha='left')
    ax.axhline(y=avg_mae, color='blue', linestyle=(0, (5, 10)), label='MAE_avg')
    plt.xticks(range(len(error_rmse)), labels, rotation=45)
    ax2 = ax.twinx()
    ax2.plot(range(len(error_rmse)), n_elem, color='black', linewidth=1.5, label='Campione')
    ax2.set_yscale('log')
    ax.set_yscale('log')
    ax2.set_ylabel("# Entries", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.set_xlabel('Range Potenza Relativa', fontsize=14)
    ax.grid(color='black', which='major', axis='y', alpha=0.5, linestyle='dashed', linewidth=1)
    ax.grid(color='black', which='minor', axis='y', alpha=0.5, linestyle='dashed', linewidth=0.5)
    ax.tick_params(axis='y', which='minor', left=True)
    # ax.legend(loc='best')
    fig.legend(loc="center right", bbox_to_anchor=(1, 0.5), bbox_transform=ax.transAxes, fontsize='large')

    if title is not None:
        fig.suptitle(title)

    if save is None:
        plt.show()
    else:
        plt.savefig(save, bbox_inches='tight')
    return labels, error_rmse, error_mae, n_elem, avg_rmse, avg_mae


def Model_Plot(model, modelname=None, n=1000, parameters=None, x_transform=None, scatter=True, save=None,
               Single_Param=False, censura=True):
    """
    Esegue il plot del modello specificato

    :param model: modello di cui eseguire il plot
    :param modelname: nome del modello per titolo [optz.]
    :param n: numero di punti per grafico [1000]
    :param parameters: specifico per modelli di regressione custom [optz.]
    :param x_transform: specifico per modelli che lo presentano [optz.]
    :param scatter: plot sullo sfondo della curva di rendimento storica [True]
    :param save: Piuttosto che visualizzare l'immagine la salva nel percorso specificato [optz.]
    :param Single_Param: Flag per analisi monovariata [False]

    """
    # LIBs
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from cust_libs.data_processing import filter_data
    from cust_libs.misc import transf_fun
    import matplotlib as mpl
    #
    data = pd.read_csv('Data/Processed/Corsini2021/Corsini2021_Processed_ON.csv')
    alpha = 0.1
    mpl.rcParams["font.size"] = 18
    rend_max = 0.545
    fig, ax = plt.subplots(figsize=(18, 9))
    if scatter:
        gradh = filter_data(data, 0.02, None, None, None)  # 8.2 MW / min
        gradmh = filter_data(data, 0.005, 0.02, None, None)
        grad0 = filter_data(data, -0.005, 0.005, None, None)
        gradml = filter_data(data, -0.02, -0.005, None, None)
        gradl = filter_data(data, None, -0.02, None, None)
        gradh.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  >  8.5$ MW/min ',
                                         marker='o', color='tab:red', ax=ax, alpha=alpha)
        gradmh.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento',
                                          label=r'$ \nabla  P  \in $ [ 2.0;  8.5 ]  MW/min',
                                          marker='o', color='tab:orange', ax=ax, alpha=alpha)
        grad0.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento',
                                         label=r'$ \nabla  P  \in $ [-2.0;  2.0 ]  MW/min',
                                         marker='o', color='tab:green', ax=ax, alpha=alpha)
        gradml.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento',
                                          label=r'$ \nabla  P  \in $ [-8.5; -2.0 ]  MW/min',
                                          marker='o', color='tab:cyan', ax=ax, alpha=alpha)
        gradl.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  <  8.5$ MW/min ',
                                         marker='o', color='tab:blue', ax=ax, alpha=alpha)

    if Single_Param:
        x_1 = np.linspace(0.1, 0.9, n, endpoint=True).reshape(-1, 1)
        if parameters is None:
            Eff_Pred = x_1
            if x_transform is not None:
                Eff_Pred = x_transform.transform(x_1)
            y_pred = model.predict(Eff_Pred).reshape(-1, 1)
        else:
            y_pred = model(x_1, *parameters).reshape(-1, 1)
        if scatter:
            ax.plot(x_1, y_pred, label=r'Curva di rendimento', linewidth=3.5, color='r')
        else:
            ax.plot(x_1, y_pred, linewidth=2, color='r')
    else:
        x_1 = np.linspace(0.1, 0.9, n, endpoint=True)
        gradients = [-0.04, -0.01, 0, 0.01, 0.04]
        labels = [-16, -4, 0, 4, 16]
        colors = ['b', 'c', 'k', 'm', 'r']
        for grad, label, col in zip(gradients, labels, colors):
            grad_column = np.linspace(grad, grad, n)
            Eff_Pred = np.vstack((x_1.flatten(), grad_column.flatten())).T
            if parameters is None:
                if x_transform is not None:
                    Eff_Pred = x_transform.transform(Eff_Pred)
                y_pred = model.predict(Eff_Pred).reshape(-1, 1)
            else:
                y_pred = model(Eff_Pred, *parameters).reshape(-1, 1)
            if scatter:
                ax.plot(x_1, y_pred, label=r'$ \nabla  P$  = ' + str(label) + ' MW/min', linewidth=3.5, color=col)
            else:
                ax.plot(x_1, y_pred, label=r'$ \nabla  P$  = ' + str(label) + ' MW/min', linewidth=2, color=col)

    rel2tot, tot2rel = transf_fun(data)
    ax.set_xlim(0.1, 0.9)
    ax.set_xlabel(r'$P_{rel}$')
    secax1_x = ax.secondary_xaxis('top', functions=(rel2tot, tot2rel))
    secax1_x.set_xlabel(r'$P\ [MW]$')
    if censura:
        ax.yaxis.set_ticks(np.linspace(0, rend_max, 6, endpoint=True))
        locs_y = ax.get_yticks()
        ax.set_yticks(locs_y, np.round(locs_y * 100 / rend_max, 1))
        ax.set_ylabel('Rendimento Rel. [%]')
    else:
        locs_y = ax.get_yticks()
        ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
        ax.set_ylabel('Rendimento [%]')
    if modelname is not None:
        plt.title(modelname)
    plt.legend(loc='best')
    if save is None:
        plt.show()
    else:
        plt.savefig(save, bbox_inches='tight')

    return


def Model_Plot_3D(model, modelname=None, n=1000, parameters=None, x_transform=None):
    """
    Esegue il plot 3D del modello specificato

    :param model: modello di cui eseguire il plot
    :param modelname: nome del modello per titolo [optz.]
    :param n: numero di punti per grafico [1000]
    :param parameters: specifico per modelli di regressione custom [optz.]
    :param x_transform: specifico per modelli che lo presentano [optz.]
    """
    # LIBs
    import matplotlib.pyplot as plt
    import numpy as np
    #
    # Grid for 3D plot
    x_1 = np.linspace(0.1, 0.9, n, endpoint=True)
    x_2 = np.linspace(-0.05, 0.05, n, endpoint=True)
    X1, X2 = np.meshgrid(x_1, x_2)
    X12 = np.vstack((X1.flatten(), X2.flatten())).T
    # Effciency - Pred
    if parameters is None:
        if x_transform is not None:
            X12 = x_transform.transform(X12)
        Z = (model.predict(X12)).reshape(-1, 1)
    else:
        Z = model(X12, *parameters)
    z = np.array(Z)
    z = z.reshape((len(X1), len(X2)))
    # 3D view
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X1, X2, z, 50, cmap='binary')
    if modelname is not None:
        plt.title(modelname)
    plt.show()


def Pred_Plot(model, x_test, y_test, x_transform=None, modelname=None, parameters=None, save=None, Single_Param=False,
              censura=True):
    """
    Esegue un grafico che confronta i dati storici con quelli predetti dal modello

    :param model: modello di cui eseguire il plot
    :param modelname: nome del modello per titolo [optz.]
    :param x_test: numero di punti per grafico [1000]
    :param y_test: plot sullo sfondo della curva di rendimento storica
    :param parameters: specifico per modelli di regressione custom [optz.]
    :param x_transform: specifico per modelli che lo presentano [optz.]
    :param y_test: plot sullo sfondo della curva di rendimento storica
    :param save: Piuttosto che visualizzare l'immagine la salva nel percorso specificato [optz.]

    """
    # LIBs
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    #
    rend_max = 0.545
    if len(x_test) == len(y_test):
        if parameters is None:
            if x_transform is not None:
                x_test = x_transform.transform(x_test)
            y_pred = model.predict(x_test).reshape(-1, 1)
        else:
            y_pred = model(x_test, *parameters).reshape(-1, 1)
        mpl.rcParams["font.size"] = 18
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.plot(y_test, color="cornflowerblue", label="Data", linewidth=2)
        ax.plot(y_pred, color="yellowgreen", label="Prediction", linewidth=2)
        ax.set_ylim([0.46, 0.53])
        plt.xlabel('Tempo[min]')
        if censura:
            ax.yaxis.set_ticks(np.linspace(0.45, rend_max, 6, endpoint=True))
            locs_y = ax.get_yticks()
            ax.set_yticks(locs_y, np.round(locs_y * 100 / rend_max, 1))
            ax.set_ylabel('Rendimento Rel. [%]')
        else:
            locs_y = ax.get_yticks()
            ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
            ax.set_ylabel('Rendimento [%]')
        if modelname is not None:
            plt.title(modelname)
        plt.legend()
        if save is None:
            plt.show()
        else:
            plt.savefig(save, bbox_inches='tight')
    else:
        print('Errore nel dataset Test')
    return


def Delibr_Plot(model, data, x_transform=None, modelname=None, parameters=None, save=None, Single_Param=False,
                censura=True):
    """
    Esegue un grafico che confronta i dati storici con quelli predetti dal modello

    :param model: modello di cui eseguire il plot
    :param modelname: nome del modello per titolo [optz.]
    :param data
    :param parameters: specifico per modelli di regressione custom [optz.]
    :param x_transform: specifico per modelli che lo presentano [optz.]
    :param save: Piuttosto che visualizzare l'immagine la salva nel percorso specificato [optz.]

    """
    # LIBs
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    #
    rend_max = 0.545
    if Single_Param:
        x = data['PwrTOT_rel'].values.reshape(-1, 1)
    else:
        x = data[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
    y_true = data['Rendimento'].values.reshape(-1, 1)
    if parameters is None:
        if x_transform is not None:
            x = x_transform.transform(x)
        y_pred = model.predict(x).reshape(-1, 1)
    else:
        y_pred = model(x, *parameters).reshape(-1, 1)
    mpl.rcParams["font.size"] = 18
    fig, ax = plt.subplots(figsize=(10, 10))

    plt.scatter(y_true, y_pred, c='crimson')

    p1 = max(max(y_true), max(y_pred))
    p2 = min(min(y_true), min(y_pred))
    ax.plot([p1, p2], [p1, p2], 'b-')

    if censura:
        locs_y = ax.get_yticks()
        ax.set_yticks(locs_y, np.round(locs_y * 100 / rend_max, 1))
        ax.set_ylabel('Predictions - Rendimento Rel. [%]')
        locs_x = ax.get_xticks()
        ax.set_xticks(locs_x, np.round(locs_x * 100 / rend_max, 1))
        ax.set_xlabel('True Values - Rendimento Rel. [%]')
    else:
        locs_y = ax.get_yticks()
        ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
        ax.set_ylabel('Predictions - Rendimento [%]')
        locs_x = ax.get_xticks()
        ax.set_xticks(locs_x, np.round(locs_x * 100, 1))
        ax.set_xlabel('True Values - Rendimento Rel. [%]')

    ax.axis('equal')
    if modelname is not None:
        plt.title(modelname)
    plt.legend()
    if save is None:
        plt.show()
    else:
        plt.savefig(save, bbox_inches='tight')
    return


# SAVE / LOAD


def Save_Poly_Model(model, transformer, out_folder):
    """
       Salva il modello di regressione polinomiale nella cartella specificata

       :param transformer: X-Transformer associato al modello
       :param model: modello da salvare
       :param out_folder: cartella di output del modello
    """
    # LIBs
    from joblib import dump
    from cust_libs.misc import mk_dir
    #
    mk_dir(out_folder)
    dump(model, out_folder + '/model.joblib')
    dump(transformer, out_folder + '/transformer.joblib')
    return


def Save_ML_Model(model, attributes, output_folder):
    """
       Salva il modello di regressione ML nella cartella specificata assieme a un file che ne salva le caratteristiche
       fondamentali

       :param model: modello da salvare
       :param attributes: diziaonario contenente le caratteristiche del modello
       :param output_folder: cartella di output del modello
    """
    # LIBs
    from joblib import dump
    from cust_libs.misc import mk_dir, folder_tree
    #

    mk_dir(output_folder)

    if attributes['regressor'] == '*':
        regressor = 'ALL'
    else:
        regressor = attributes['regressor']

    folder = output_folder + '/' + regressor
    mk_dir(folder)

    files = folder_tree(folder, from_root=False)

    num = 1
    modelname = regressor + '_' + str(num).zfill(2)
    filename = modelname + '.joblib'
    while filename in files:
        num += 1
        modelname = regressor + '_' + str(num).zfill(2)
        filename = modelname + '.joblib'

    print('Salvando ' + filename)
    dump(model, folder + '/' + filename)
    mk_dir(folder + '/' + modelname)

    with open(folder + '/' + modelname + "/Description.txt", 'w') as f:
        for key, value in attributes.items():
            f.write('%s:%s\n' % (key, value))
    return


def Load_Poly_model(folder):
    """
    Carica il modello di regressione polinomiale dalla cartella specificata precedentemente salvato mediante
    Save_Poly_model

    :param folder: cartella relatva al modello
    """
    # LIBs
    from joblib import load
    #
    model = load(folder + '/model.joblib')
    transformer = load(folder + '/transformer.joblib')
    return model, transformer


def Load_ML_Model(model_file: str):
    """
    Carica il modello di regressione ML dal file specficato precedentemente salvato mediante
    Save_ML_model assieme a file contenente i suoi attributi.

    :param model_file: file del modello
    """
    # LIBs
    from joblib import load
    from pathlib import Path
    #
    model = load(model_file)
    modelname = Path(model_file).stem
    folder = Path(model_file).parent
    desc_folder = str(folder) + '/' + modelname
    desc_file = desc_folder + '/Description.txt'
    attributes = {}
    with open(desc_file) as f:
        for line in f:
            line = line.replace('\n', '')
            (key, val) = line.split(':', 1)
            attributes[key] = val
    return model, attributes


# AUTOML

def HyperOpt(x_train, y_train, x_test, y_test, regressor, preprocessing, loss_fn, n_iter, timeout):
    # LIBs
    import hyperopt
    from hpsklearn import HyperoptEstimator
    from hpsklearn import any_regressor, svr, random_forest_regressor, ada_boost_regressor,  mlp_regressor
    from hpsklearn import any_preprocessing
    from hyperopt import tpe
    from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error

    attributes = dict([
        ('regressor', regressor),
        ('preprocessing', preprocessing),
        ('loss_fn', loss_fn),
        ('n_iter', str(n_iter)),
        ('timeout', str(timeout))
    ])

    if regressor == '*':
        regressor = any_regressor('reg')
    elif regressor == 'SVR':
        regressor = svr('svr')
    elif regressor == 'RND_FOR':
        regressor = random_forest_regressor('rnd_frst')
    elif regressor == 'ADA_BST':
        regressor = ada_boost_regressor('ada_bst')
    elif regressor == 'MLP':
        regressor = mlp_regressor('ml_perceptron')
    else:
        print('Il regressore selezionato "' + regressor + '"non è supportato')
        return

    if preprocessing == '*':
        preprocessing = any_preprocessing('pre')
    else:
        print('Il metodo di pre-prox dati selezionato "' + preprocessing + '"non è supportato')
        return

    if loss_fn == 'MSE':
        loss_fn = mean_squared_error
    elif loss_fn == 'MAE':
        loss_fn = mean_absolute_error
    elif loss_fn == 'MAX':
        loss_fn = max_error
    else:
        print('La funzione di costo selezionata "' + loss_fn + '" non è supportata')
        return

    y_train = y_train.ravel()

    model = HyperoptEstimator(
        regressor=regressor, preprocessing=preprocessing, loss_fn=loss_fn, algo=tpe.suggest, max_evals=n_iter,
        trial_timeout=timeout, n_jobs=-1
    )

    count = 1

    while count <= 10:
        try:
            print('TRIAL # ' + str(count))
            model.fit(x_train, y_train)
            print('TRIAL # ' + str(count) + ' ESEGUITO CORRETTAMENTE')
            count = 100
        except (hyperopt.exceptions.AllTrialsFailed, ValueError):
            print('TRIAL # ' + str(count) + 'FALLITO')
            count += 1

    if count != 100:
        print('Errore nel training')
        return

    # SCORE TEST
    y_pred = model.predict(x_test)
    mse_test = mean_squared_error(y_test, y_pred)
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)
    mae_test = mean_absolute_error(y_test, y_pred)

    y_pred_tr = model.predict(x_train)
    mse_train = mean_squared_error(y_train, y_pred_tr)
    rmse_train = mean_squared_error(y_train, y_pred_tr, squared=False)
    mae_train = mean_absolute_error(y_train, y_pred_tr)

    attributes['MSE-TRAIN'] = str(mse_train) + ' ( RMSE: ' + str(rmse_train) + ' )'
    attributes['MAE-TRAIN'] = str(mae_train)
    attributes['MSE-TEST'] = str(mse_test) + ' ( RMSE: ' + str(rmse_test) + ' )'
    attributes['MAE-TEST'] = str(mae_test)
    return model, attributes

# OTHER


def Predict_Point_Poly(model, transformer, point):
    """
        Equivalente di model.predict() per modelli polinomiali

        :param model: modello polinomiale
        :param transformer: X-Transformer associato al modello
        :param point: punto da cui effettuare la previsione

        """
    # LIBs
    import numpy as np
    #
    to_predict = np.array([[point]])
    x = transformer.transform(to_predict)
    y = model.predict(x)
    return y
