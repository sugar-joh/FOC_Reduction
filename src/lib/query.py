#!/usr/bin/python3
#-*- coding:utf-8 -*-
"""
Library function to query and download datatsets from MAST api.
"""
from os import system
from os.path import join as path_join, exists as path_exists
from astroquery.mast import MastMissions, Observations
from astropy.table import unique, Column
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np


def divide_proposal(products):
    """
    Divide observation in proposals by time or filter
    """
    for pid in np.unique(products['Proposal ID']):
        obs = products[products['Proposal ID']==pid].copy()
        close_date = np.unique(np.array([TimeDelta(np.abs(Time(obs['Start']).unix-date.unix),format='sec') < 7.*u.d for date in obs['Start']], dtype=bool), axis=0)
        if len(close_date)>1:
            for date in close_date:
                products['Proposal ID'][np.any([products['Dataset']==dataset for dataset in obs['Dataset'][date]],axis=0)] = "_".join([obs['Proposal ID'][date][0],str(obs['Start'][date][0])[:10]])
    for pid in np.unique(products['Proposal ID']):
        obs = products[products['Proposal ID']==pid].copy()
        same_filt = np.unique(np.array(np.sum([obs['Filters'][:,1:]==filt[1:] for filt in obs['Filters']],axis=2)<3,dtype=bool),axis=0)
        if len(same_filt)>1:
            for filt in same_filt:
                products['Proposal ID'][np.any([products['Dataset']==dataset for dataset in obs['Dataset'][filt]],axis=0)] = "_".join([obs['Proposal ID'][filt][0],"_".join([fi for fi in obs['Filters'][filt][0][1:] if fi[:-1]!="CLEAR"])])
    return products


def get_product_list(target=None, proposal_id=None):
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

    if target is None:
        target = input("Target name:\n>")

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
    results['Proposal ID'] = Column(results['Proposal ID'],dtype='U35')
    results['Filters'] = Column(np.array([filt.split(";") for filt in results['Filters']],dtype=str))
    results['Start'] = Column(Time(results['Start']))
    results['Stop'] = Column(Time(results['Stop']))

    results = divide_proposal(results)
    obs = results.copy()

    polfilt = {"POL0":0,"POL60":1,"POL120":2}
    for pid in np.unique(obs['Proposal ID']):
        used_pol = np.zeros(3)
        for dataset in obs[obs['Proposal ID'] == pid]:
            used_pol[polfilt[dataset['Filters'][0]]] += 1
        if np.all(used_pol < 1):
            obs.remove_rows(np.arange(len(obs))[obs['Proposal ID'] == pid])

    obs["Obs"] = [np.argmax(unique(obs, 'Proposal ID')[
                            'Proposal ID'] == data['Proposal ID'])+1 for data in obs]
    try:
        obs = unique(obs[["Obs", "Filters", "Start", "Central wavelength", "Instrument",
                     "Size", "Target name", "Proposal ID", "PI last name"]], 'Proposal ID')
    except IndexError:
        raise ValueError(
            "There is no observation with POL0, POL60 and POL120 for {0:s} in HST/FOC Legacy Archive".format(target))

    b = np.zeros(len(results), dtype=bool)
    if not proposal_id is None and str(proposal_id) in obs['Proposal ID']:
        b[results['Proposal ID'] == str(proposal_id)] = True
    else:
        print(obs)
        a = [np.array(i.split(":"), dtype=str) for i in input("select observations to be downloaded ('1,3,4,5' or '1,3:5' or 'all','*' default to 1)\n>").split(',')]
        if a[0][0]=='':
            a = [[1]]
        if a[0][0] in ['a','all','*']:
            b = np.ones(len(results),dtype=bool)
        else:
            a = [np.array(i,dtype=int) for i in a]
            for i in a:
                if len(i) > 1:
                    for j in range(i[0], i[1]+1):
                        b[results['Proposal ID'] == obs['Proposal ID'][obs["Obs"] == j]] = True
                else:
                    b[results['Proposal ID'] == obs['Proposal ID'][obs['Obs'] == i[0]]] = True

    observations = Observations.query_criteria(obs_id=list(results['Dataset'][b]))
    products = Observations.filter_products(Observations.get_product_list(observations),
                                            productType=['SCIENCE'],
                                            dataproduct_type=['image'],
                                            calib_level=[2],
                                            description="DADS C0F file - Calibrated exposure WFPC/WFPC2/FOC/FOS/GHRS/HSP")
    products['proposal_id'] = Column(products['proposal_id'],dtype='U35')
    
    for pid in np.unique(results['Proposal ID']):
        rpid = results['Proposal ID']==pid
        ppid = np.argmax([results['Dataset'][rpid] == prod[:len(results['Dataset'][0])].upper() for prod in products['productFilename']],axis=0)
        products['proposal_id'][ppid] = pid

    return target, products


def retrieve_products(target=None, proposal_id=None, output_dir='./data'):
    """
    Given a target name and a proposal_id, create the local directories and retrieve the fits files from the MAST Archive
    """
    target, products = get_product_list(target=target,proposal_id=proposal_id)
    prodpaths = []
    data_dir = path_join(output_dir, target)
    out = ""
    for obs_id in unique(products, 'proposal_id')['proposal_id']:
        filepaths = []
        obs_dir = path_join(data_dir, obs_id)
        if not path_exists(obs_dir):
            system("mkdir -p {0:s} {1:s}".format(obs_dir,obs_dir.replace("data","plots")))
        for file in products['productFilename'][products['proposal_id'] == obs_id]:
            fpath = path_join(obs_dir, file)
            if not path_exists(fpath):
                out += "{0:s} : {1:s}\n".format(file, Observations.download_file(
                    products['dataURI'][products['productFilename'] == file][0], local_path=fpath)[0])
            else:
                out += "{0:s} : Exists\n".format(file)
            filepaths.append([obs_dir,file])
        prodpaths.append(np.array(filepaths,dtype=str))

    return target, prodpaths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Query MAST for target products')
    parser.add_argument('-t','--target', metavar='targetname', required=False,
                        help='the name of the target', type=str, default=None)
    parser.add_argument('-p','--proposal_id', metavar='proposal_id', required=False,
                        help='the proposal id of the data products', type=int, default=None)
    parser.add_argument('-o','--output_dir', metavar='directory_path', required=False,
                        help='output directory path for the data products', type=str, default="./data")
    args = parser.parse_args()
    prodpaths = retrieve_products(target=args.target, proposal_id=args.proposal_id)
    print(prodpaths)