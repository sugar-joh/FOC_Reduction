from os import system
from os.path import join as path_join, exists as path_exists
from astroquery.mast import MastMissions, Observations
from astropy.table import unique
import numpy as np


def get_product_list(target, proposal_id=None):
    """
    Retrieve products list for a given target from the MAST archive
    """
    mission = MastMissions(mission='hst')
    radius = '3'
    select_cols = [
        'sci_data_set_name',
        'sci_spec_1234',
        'sci_actual_duration',
        'sci_start_time',
        'sci_stop_time',
        'sci_central_wavelength',
        'sci_instrume',
        'sci_aper_1234',
        'sci_targname',
        'sci_pep_id',
        'sci_pi_last_name']

    cols = [
        'Dataset',
        'Filters',
        'Exptime',
        'Start',
        'Stop',
        'Central wavelength',
        'Instrument',
        'Size',
        'Target name',
        'Proposal ID',
        'PI last name']

    # Use query_object method to resolve the object name into coordinates
    results = mission.query_object(
        target,
        radius=radius,
        select_cols=select_cols,
        sci_spec_1234='POL*',
        sci_obs_type='image',
        sci_aec='S',
        sci_instrume='foc')

    for c, n_c in zip(select_cols, cols):
        results.rename_column(c, n_c)

    obs = results.copy()

    for pid in np.unique(obs['Proposal ID']):
        used_pol = []
        for dataset in obs[obs['Proposal ID'] == pid]:
            filtnam = dataset["Filters"].split(";")
            obs["Filters"][obs["Dataset"] ==
                           dataset["Dataset"]] = ";".join(filtnam[1:])
            used_pol.append(filtnam[0])
        if np.unique(used_pol).size < 3:
            del obs[obs['Proposal ID'] == pid]

    obs["Obs"] = [np.argmax(unique(obs, 'Proposal ID')[
                            'Proposal ID'] == data['Proposal ID'])+1 for data in obs]
    try:
        obs = unique(obs[["Obs", "Filters", "Start", "Central wavelength", "Instrument",
                     "Size", "Target name", "Proposal ID", "PI last name"]], 'Proposal ID')
    except IndexError:
        raise ValueError(
            "There is no observation with POL0, POL60 and POL120 for {0:s} in HST/FOC Legacy Archive".format(target))

    b = np.zeros(len(results), dtype=bool)
    if not proposal_id is None and proposal_id in obs['Proposal ID']:
        b[results['Proposal ID'] == proposal_id] = True
    else:
        print(obs)
        try:
            a = [np.array(i.split(":"), dtype=int) for i in input(
                "select observations to be downloaded ('1,3,4,5' or '1,3:5' default to 1)\n>").split(',')]
        except ValueError:
            a = [[1]]
        for i in a:
            if len(i) > 1:
                for j in range(i[0], i[1]+1):
                    b[results['Proposal ID'] == obs['Proposal ID']
                        [obs["Obs"] == j]] = True
            else:
                b[results['Proposal ID'] == obs['Proposal ID']
                    [obs['Obs'] == i[0]]] = True

    observations = Observations.query_criteria(obs_id=list(results['Dataset'][b]))
    products = Observations.filter_products(Observations.get_product_list(observations),
                                            productType=['SCIENCE'],
                                            dataproduct_type=['image'],
                                            calib_level=[2],
                                            description="DADS C0F file - Calibrated exposure WFPC/WFPC2/FOC/FOS/GHRS/HSP")

    return products


def retrieve_products(target, proposal_id=None):
    """
    Given a target name and a proposal_id, create the local directories and retrieve the fits files from the MAST Archive
    """
    products = get_product_list(target=target,proposal_id=proposal_id)
    prodpaths = []
    data_dir = path_join("../data", target)
    out = ""
    for obs_id in unique(products, 'proposal_id')['proposal_id']:
        filepaths = []
        obs_dir = path_join(data_dir, obs_id)
        if not path_exists(obs_dir):
            system("mkdir -p {0:s} {1:s}".format(obs_dir,path_join("../plots",target,obs_id)))
        for file in products['productFilename'][products['proposal_id'] == obs_id]:
            fpath = path_join(obs_dir, file)
            if not path_exists(fpath):
                out += "{0:s} : {1:s}\n".format(file, Observations.download_file(
                    products['dataURI'][products['productFilename'] == file][0], local_path=fpath)[0])
            else:
                out += "{0:s} : Exists\n".format(file)
            filepaths.append([obs_dir,file])
        prodpaths.append(np.array(filepaths,dtype=str))

    return prodpaths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Query MAST for target products')
    parser.add_argument('-t','--target', metavar='targetname', required=False,
                        help='the name of the target', type=str, default=None)
    parser.add_argument('-p','--proposal_id', metavar='proposal_id', required=False,
                        help='the proposal id of the data products', type=int, default=None)
    args = parser.parse_args()
    prodpaths = retrieve_products(target=args.target, proposal_id=args.proposal_id)
    print(prodpaths)