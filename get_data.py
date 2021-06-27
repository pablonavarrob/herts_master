from astropy.io import fits
import matplotlib.pyplot as plt
import requests
import json
import numpy as np

data = fits.open('/Users/pab.nb/Desktop/GALAH_DR2.1_catalog.fits.txt')[1].data

# Make some plots, see what all the fuzz is about lol
# Get 10K stars that are between 4500 and 6000K and that don't have crazy loggs
mask = ((data.teff < 6000) & (data.teff > 5000))
# filter
data_filtered = data[mask][0:10]
# sample 10K
id_list = list(map(str, data_filtered.sobject_id.tolist()))
v
content = {
    "source_list": ','.join(id_list), # list of ids
    "data_releases": [16], # what data releases to check
    "loose_matching": False, # false by default
    "data_products_ifs": [],
    "data_products_spectra": [59],
    "email": "pablonavarrobarrachina@gmail.com"
}

resp = requests.post('https://datacentral.org.au/api/services/download/',
                     json = content)

if resp.status_code == 201:
    result = resp.json()['url']
    # Another get request to get the information for the donwload of the
    # spectral packages
    resp_link = requests.get(result)
else:
    print('Something went wrong.')

# To trigger the donwload:
task_id = resp_link.json()['tasks'][0]['id']
object_id = resp_link.json()['tasks'][0]['object_id']
download_link = ('https://datacentral.org.au/services_data/public/download/' +
                 '{}'.format(object_id) + '/' +
                 '{}'.format(task_id) + '.tar.gz')

import shutil

# Snippet obtained from: 
# https://stackoverflow.com/questions/16694907/
# download-large-file-in-python-with-requests
def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f, length=16*1024*1024)

    return local_filename
