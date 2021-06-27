from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# import the data from the stars
star_data = fits.open('../GALAH_DR3_main_allstar_v2.fits')
star_selected = (star_data[1]
                    .data[star_data[1].data['sobject_id'] == 140711003901014])

# import the first channel of the hermes spectrograph
# 1 blue, 2 green, 3 red, 4 near-infrared
flux_vales = fits.open('../../../Desktop/1407110039010141.fits')[4].data
# the fifth table is the normalize
# spectra with sky substraction, pseudo-normalized?
waverange = np.linspace(4718, 4903, 4096)
fig, ax = plt.subplots(1, figsize=[10, 10])
ax.plot(waverange, flux_vales)
plt.show()
