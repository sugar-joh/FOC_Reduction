"""
Library functions for displaying  informations using matplotlib

prototypes :
    - plot_obs(data_array, headers, shape, vmin, vmax, savename, plots_folder)
        Plots whole observation raw data in given display shape

    - polarization_map(Stokes_hdul, SNRp_cut, SNRi_cut, step_vec, savename, plots_folder, display)
        Plots polarization map of polarimetric parameters saved in an HDUList
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.transforms import Bbox
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar, AnchoredDirectionArrows
from astropy.wcs import WCS


def princ_angle(ang):
    """
    Return the principal angle in the 0-180° quadrant.
    """
    while ang < 0.:
        ang += 180.
    while ang > 180.:
        ang -= 180.
    return ang


def sci_not(v,err,rnd=1):
    """
    Return the scientifque error notation as a string.
    """
    power = - int(('%E' % v)[-3:])+1
    output = r"({0}".format(round(v*10**power,rnd))
    if type(err) == list:
        for error in err:
            output += r" $\pm$ {0}".format(round(error*10**power,rnd))
    else:
        output += r" $\pm$ {0}".format(round(err*10**power,rnd))
    return output+r")e{0}".format(-power)


def plot_obs(data_array, headers, shape=None, vmin=0., vmax=6., rectangle=None,
        savename=None, plots_folder=""):
    """
    Plots raw observation imagery with some information on the instrument and
    filters.
    ----------
    Inputs:
    data_array : numpy.ndarray
        Array of images (2D floats, aligned and of the same shape) of a
        single observation with multiple polarizers of an instrument
    headers : header list
        List of headers corresponding to the images in data_array
    shape : array-like of length 2, optional
        Shape of the display, with shape = [#row, #columns]. If None, defaults
        to the optimal square.
        Defaults to None.
    vmin : float, optional
        Min pixel value that should be displayed.
        Defaults to 0.
    vmax : float, optional
        Max pixel value that should be displayed.
        Defaults to 6.
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    if shape is None:
        shape = np.array([np.ceil(np.sqrt(data_array.shape[0])).astype(int),]*2)
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(10,10), dpi=200,
            sharex=True, sharey=True)

    for i, enum in enumerate(list(zip(ax.flatten(),data_array))):
        ax = enum[0]
        data = enum[1]
        instr = headers[i]['instrume']
        rootname = headers[i]['rootname']
        exptime = headers[i]['exptime']
        filt = headers[i]['filtnam1']
        #plots
        im = ax.imshow(data, vmin=vmin, vmax=vmax, origin='lower')
        if not(rectangle is None):
            x, y, width, height, angle, color = rectangle[i]
            ax.add_patch(Rectangle((x, y), width, height, angle=angle,
                edgecolor=color, fill=False))
        #position of centroid
        ax.plot([data.shape[1]/2, data.shape[1]/2], [0,data.shape[0]-1], lw=1,
                color='black')
        ax.plot([0,data.shape[1]-1], [data.shape[1]/2, data.shape[1]/2], lw=1,
                color='black')
        ax.annotate(instr+":"+rootname,color='white',fontsize=5,xy=(0.02, 0.95),
                xycoords='axes fraction')
        ax.annotate(filt,color='white',fontsize=10,xy=(0.02, 0.02),
                xycoords='axes fraction')
        ax.annotate(exptime,color='white',fontsize=5,xy=(0.80, 0.02),
                xycoords='axes fraction')

    fig.subplots_adjust(hspace=0, wspace=0, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.75])
    fig.colorbar(im, cax=cbar_ax)

    if not (savename is None):
        fig.suptitle(savename)
        fig.savefig(plots_folder+savename+".png",bbox_inches='tight')
    plt.show()
    return 0


def plot_Stokes(Stokes, savename=None, plots_folder=""):
    """
    Plots I/Q/U maps.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    """
    # Get data
    stkI = Stokes[np.argmax([Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])].data
    stkQ = Stokes[np.argmax([Stokes[i].header['datatype']=='Q_stokes' for i in range(len(Stokes))])].data
    stkU = Stokes[np.argmax([Stokes[i].header['datatype']=='U_stokes' for i in range(len(Stokes))])].data

    wcs = WCS(Stokes[0]).deepcopy()

    # Plot figure
    fig = plt.figure(figsize=(30,10))

    ax = fig.add_subplot(131, projection=wcs)
    im = ax.imshow(stkI, origin='lower')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$I_{stokes}$")

    ax = fig.add_subplot(132, projection=wcs)
    im = ax.imshow(stkQ, origin='lower')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$Q_{stokes}$")

    ax = fig.add_subplot(133, projection=wcs)
    im = ax.imshow(stkU, origin='lower')
    plt.colorbar(im)
    ax.set(xlabel="RA", ylabel="DEC", title=r"$U_{stokes}$")

    if not (savename is None):
        fig.suptitle(savename+"_IQU")
        fig.savefig(plots_folder+savename+"_IQU.png",bbox_inches='tight')
    plt.show()
    return 0


class crop_map(object):
    """
    Class to interactively crop I_stokes map to desired Region of Interest
    """
    def __init__(self, Stokes, data_mask, SNRp_cut=3., SNRi_cut=30.):
        #Get data
        stkI = Stokes[np.argmax([Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])]
        stk_cov = Stokes[np.argmax([Stokes[i].header['datatype']=='IQU_cov_matrix' for i in range(len(Stokes))])]
        pol = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_debiased' for i in range(len(Stokes))])]
        pol_err = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_err' for i in range(len(Stokes))])]
        self.Stokes = Stokes
        self.data_mask = data_mask

        wcs = WCS(Stokes[0]).deepcopy()

        #Compute SNR and apply cuts
        pol.data[pol.data == 0.] = np.nan
        SNRp = pol.data/pol_err.data
        SNRp[np.isnan(SNRp)] = 0.
        pol.data[SNRp < SNRp_cut] = np.nan
        SNRi = stkI.data/np.sqrt(stk_cov.data[0,0])
        SNRi[np.isnan(SNRi)] = 0.
        pol.data[SNRi < SNRi_cut] = np.nan

        convert_flux = Stokes[0].header['photflam']

        #Plot the map
        plt.rcParams.update({'font.size': 16})
        self.fig = plt.figure(figsize=(15,15))
        self.ax = self.fig.add_subplot(111, projection=wcs)
        self.ax.set_facecolor('k')
        self.fig.subplots_adjust(hspace=0, wspace=0, right=0.9)
        cbar_ax = self.fig.add_axes([0.95, 0.12, 0.01, 0.75])

        self.extent = [-stkI.data.shape[1]/2.,stkI.data.shape[1]/2.,-stkI.data.shape[0]/2.,stkI.data.shape[0]/2.]
        self.center = [stkI.data.shape[1]/2.,stkI.data.shape[0]/2.]

        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = self.ax.imshow(stkI.data*convert_flux,extent=self.extent, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        levelsI = np.linspace(SNRi_cut, np.max(SNRi[SNRi > 0.]), 10)
        cont = self.ax.contour(SNRi, extent=self.extent, levels=levelsI, colors='grey', linewidths=0.5)

        fontprops = fm.FontProperties(size=16)
        px_size = wcs.wcs.get_cdelt()[0]
        px_sc = AnchoredSizeBar(self.ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax.add_artist(px_sc)

        self.ax.set_title(
        "Click and drag to crop to desired Region of Interest.\n"
        "Press 'c' to toggle the selector on and off")

    def onselect_crop(self, eclick, erelease) -> None:
        # Obtain (xmin, xmax, ymin, ymax) values
        self.RSextent = self.rect_selector.extents
        self.RScenter = [self.center[i]+self.rect_selector.center[i] for i in range(2)]

        # Zoom to selection
        print("CROP TO : ",self.RSextent)
        print("CENTER : ",self.RScenter)
        #self.ax.set_xlim(self.extent[0], self.extent[1])
        #self.ax.set_ylim(self.extent[2], self.extent[3])

    def run(self) -> None:
        self.rect_selector = RectangleSelector(self.ax, self.onselect_crop,
                drawtype='box', button=[1], interactive=True)
        #self.fig.canvas.mpl_connect('key_press_event', self.toggle_selector)
        plt.show()

    def crop(self):
        Stokes_crop = copy.deepcopy(self.Stokes)
        # Data sets to crop
        stkI = Stokes_crop[np.argmax([Stokes_crop[i].header['datatype']=='I_stokes' for i in range(len(Stokes_crop))])]
        stkQ = Stokes_crop[np.argmax([Stokes_crop[i].header['datatype']=='Q_stokes' for i in range(len(Stokes_crop))])]
        stkU = Stokes_crop[np.argmax([Stokes_crop[i].header['datatype']=='U_stokes' for i in range(len(Stokes_crop))])]
        stk_cov = Stokes_crop[np.argmax([Stokes_crop[i].header['datatype']=='IQU_cov_matrix' for i in range(len(Stokes_crop))])]
        pol = Stokes_crop[np.argmax([Stokes_crop[i].header['datatype']=='Pol_deg_debiased' for i in range(len(Stokes_crop))])]
        pol_err = Stokes_crop[np.argmax([Stokes_crop[i].header['datatype']=='Pol_deg_err' for i in range(len(Stokes_crop))])]
        pang = Stokes_crop[np.argmax([Stokes_crop[i].header['datatype']=='Pol_ang' for i in range(len(Stokes_crop))])]
        pang_err = Stokes_crop[np.argmax([Stokes_crop[i].header['datatype']=='Pol_ang_err' for i in range(len(Stokes_crop))])]
        # Crop datasets
        vertex = [int(self.RScenter[0]+self.RSextent[0]), int(self.RScenter[0]+self.RSextent[1]), int(self.RScenter[1]+self.RSextent[2]), int(self.RScenter[1]+self.RSextent[3])]
        for dataset in [stkI, stkQ, stkU, pol, pol_err, pang, pang_err]:
            dataset.data = dataset.data[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        data_mask = self.data_mask[vertex[2]:vertex[3], vertex[0]:vertex[1]]
        new_stk_cov = np.zeros((3,3,stkI.data.shape[0],stkI.data.shape[1]))
        for i in range(3):
            for j in range(3):
                new_stk_cov[i,j] = stk_cov.data[i,j][vertex[2]:vertex[3], vertex[0]:vertex[1]]
        stk_cov.data = new_stk_cov

        return Stokes_crop, data_mask


def polarization_map(Stokes, data_mask=None, rectangle=None, SNRp_cut=3., SNRi_cut=30.,
        step_vec=1, savename=None, plots_folder="", display=None):
    """
    Plots polarization map from Stokes HDUList.
    ----------
    Inputs:
    Stokes : astropy.io.fits.hdu.hdulist.HDUList
        HDUList containing I, Q, U, P, s_P, PA, s_PA (in this particular order)
        for one observation.
    rectangle : numpy.ndarray, optional
        Array of parameters for matplotlib.patches.Rectangle objects that will
        be displayed on each output image. If None, no rectangle displayed.
        Defaults to None.
    SNRp_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on P.
        Any SNR < SNRp_cut won't be displayed.
        Defaults to 3.
    SNRi_cut : float, optional
        Cut that should be applied to the signal-to-noise ratio on I.
        Any SNR < SNRi_cut won't be displayed.
        Defaults to 30. This value implies an uncertainty in P of 4.7%
    step_vec : int, optional
        Number of steps between each displayed polarization vector.
        If step_vec = 2, every other vector will be displayed.
        Defaults to 1
    savename : str, optional
        Name of the figure the map should be saved to. If None, the map won't
        be saved (only displayed).
        Defaults to None.
    plots_folder : str, optional
        Relative (or absolute) filepath to the folder in wich the map will
        be saved. Not used if savename is None.
        Defaults to current folder.
    display : str, optional
        Choose the map to display between intensity (default), polarization
        degree ('p','pol','pol_deg') or polarization degree error ('s_p',
        'pol_err','pol_deg_err').
        Defaults to None (intensity).
    ----------
    Returns:
    fig, ax : matplotlib.pyplot object
        The figure and ax created for interactive contour maps.
    """
    #Get data
    stkI = Stokes[np.argmax([Stokes[i].header['datatype']=='I_stokes' for i in range(len(Stokes))])]
    stkQ = Stokes[np.argmax([Stokes[i].header['datatype']=='Q_stokes' for i in range(len(Stokes))])]
    stkU = Stokes[np.argmax([Stokes[i].header['datatype']=='U_stokes' for i in range(len(Stokes))])]
    stk_cov = Stokes[np.argmax([Stokes[i].header['datatype']=='IQU_cov_matrix' for i in range(len(Stokes))])]
    #pol = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg' for i in range(len(Stokes))])]
    pol = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_debiased' for i in range(len(Stokes))])]
    pol_err = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_err' for i in range(len(Stokes))])]
    #pol_err_Poisson = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_deg_err_Poisson_noise' for i in range(len(Stokes))])]
    pang = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_ang' for i in range(len(Stokes))])]
    pang_err = Stokes[np.argmax([Stokes[i].header['datatype']=='Pol_ang_err' for i in range(len(Stokes))])]

    pivot_wav = Stokes[0].header['photplam']
    convert_flux = Stokes[0].header['photflam']
    wcs = WCS(Stokes[0]).deepcopy()

    #Get image mask
    if data_mask is None:
        data_mask = np.ones(stkI.shape).astype(bool)

    #Plot Stokes parameters map
    if display is None:
        plot_Stokes(Stokes, savename=savename, plots_folder=plots_folder)

    #Compute SNR and apply cuts
    pol.data[pol.data == 0.] = np.nan
    SNRp = pol.data/pol_err.data
    SNRp[np.isnan(SNRp)] = 0.
    pol.data[SNRp < SNRp_cut] = np.nan

    maskI = stk_cov.data[0,0] > 0
    SNRi = np.zeros(stkI.data.shape)
    SNRi[maskI] = stkI.data[maskI]/np.sqrt(stk_cov.data[0,0][maskI])
    pol.data[SNRi < SNRi_cut] = np.nan

    data_mask = (1.-data_mask).astype(bool)
    mask = (SNRp > SNRp_cut) * (SNRi > SNRi_cut)

    # Look for pixel of max polarization
    if np.isfinite(pol.data).any():
        p_max = np.max(pol.data[np.isfinite(pol.data)])
        x_max, y_max = np.unravel_index(np.argmax(pol.data==p_max),pol.data.shape)
    else:
        print("No pixel with polarization information above requested SNR.")

    #Plot the map
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_facecolor('k')
    fig.subplots_adjust(hspace=0, wspace=0, right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.12, 0.01, 0.75])

    if display is None:
        # If no display selected, show intensity map
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = ax.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
        levelsI = np.linspace(SNRi_cut, np.max(SNRi[SNRi > 0.]), 10)
        cont = ax.contour(SNRi, levels=levelsI, colors='grey', linewidths=0.5)
    elif display.lower() in ['pol_flux']:
        # Display polarisation flux
        pf_mask = (stkI.data > 0.) * (pol.data > 0.)
        vmin, vmax = 0., np.max(stkI.data[pf_mask]*convert_flux*pol.data[pf_mask])
        im = ax.imshow(stkI.data*convert_flux*pol.data, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda} \cdot P$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
    elif display.lower() in ['p','pol','pol_deg']:
        # Display polarization degree map
        vmin, vmax = 0., 100.
        im = ax.imshow(pol.data*100., vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$P$ [%]")
    elif display.lower() in ['s_p','pol_err','pol_deg_err']:
        # Display polarization degree error map
        vmin, vmax = 0., 10.
        p_err = pol_err.data.copy()
        p_err[p_err > vmax/100.] = np.nan
        im = ax.imshow(p_err*100., vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$\sigma_P$ [%]")
    elif display.lower() in ['s_i','i_err']:
        # Display intensity error map
        vmin, vmax = 0., np.max(np.sqrt(stk_cov.data[0,0][stk_cov.data[0,0] > 0.])*convert_flux)
        im = ax.imshow(np.sqrt(stk_cov.data[0,0])*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$\sigma_I$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")
    elif display.lower() in ['snr','snri']:
        # Display I_stokes signal-to-noise map
        vmin, vmax = 0., np.max(SNRi[SNRi > 0.])
        im = ax.imshow(SNRi, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$I_{Stokes}/\sigma_{I}$")
        levelsI = np.linspace(SNRi_cut, np.max(SNRi[SNRi > 0.]), 10)
        cont = ax.contour(SNRi, levels=levelsI, colors='grey', linewidths=0.5)
    elif display.lower() in ['snrp']:
        # Display polarization degree signal-to-noise map
        vmin, vmax = SNRp_cut, np.max(SNRp[SNRp > 0.])
        im = ax.imshow(SNRp, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$P/\sigma_{P}$")
        levelsP = np.linspace(SNRp_cut, np.max(SNRp[SNRp > 0.]), 10)
        cont = ax.contour(SNRp, levels=levelsP, colors='grey', linewidths=0.5)
    else:
        # Defaults to intensity map
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = ax.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA$]")
        levelsI = np.linspace(SNRi_cut, SNRi.max(), 10)
        cont = ax.contour(SNRi, levels=levelsI, colors='grey', linewidths=0.5)

    fontprops = fm.FontProperties(size=16)
    px_size = wcs.wcs.get_cdelt()[0]*3600.
    px_sc = AnchoredSizeBar(ax.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
    ax.add_artist(px_sc)

    #pol.data[np.isfinite(pol.data)] = 1./2.
    X, Y = np.meshgrid(np.linspace(0,stkI.data.shape[0],stkI.data.shape[0]), np.linspace(0,stkI.data.shape[1],stkI.data.shape[1]))
    U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
    Q = ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')
    pol_sc = AnchoredSizeBar(ax.transData, 2., r"$P$= 100 %", 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
    ax.add_artist(pol_sc)

    north_dir = AnchoredDirectionArrows(ax.transAxes, "E", "N", length=-0.08, fontsize=0.03, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, angle=-Stokes[0].header['orientat'], color='w', arrow_props={'ec': 'w', 'fc': 'w', 'alpha': 1,'lw': 2})
    ax.add_artist(north_dir)

    # Display instrument FOV
    if not(rectangle is None):
        x, y, width, height, angle, color = rectangle
        x, y = np.array([x, y])- np.array(stkI.data.shape)/2.
        ax.add_patch(Rectangle((x, y), width, height, angle=angle,
            edgecolor=color, fill=False))

    # Compute integrated parameters and associated errors for pixels in the cut
    n_pix = mask.size
    I_int = stkI.data[mask].sum()
    Q_int = stkQ.data[mask].sum()
    U_int = stkU.data[mask].sum()
    I_int_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[0,0][mask]))
    Q_int_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[1,1][mask]))
    U_int_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[2,2][mask]))
    IQ_int_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[0,1][mask]**2))
    IU_int_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[0,2][mask]**2))
    QU_int_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[1,2][mask]**2))

    P_int = np.sqrt(Q_int**2+U_int**2)/I_int
    P_int_err = (1./I_int)*np.sqrt((Q_int**2*Q_int_err**2 + U_int**2*U_int_err**2 + 2.*Q_int*U_int*QU_int_err)/(Q_int**2 + U_int**2) + ((Q_int/I_int)**2 + (U_int/I_int)**2)*I_int_err**2 - 2.*(Q_int/I_int)*IQ_int_err - 2.*(U_int/I_int)*IU_int_err)

    PA_int = princ_angle((90./np.pi)*np.arctan2(U_int,Q_int))
    PA_int_err = (90./(np.pi*(Q_int**2 + U_int**2)))*np.sqrt(U_int**2*Q_int_err**2 + Q_int**2*U_int_err**2 - 2.*Q_int*U_int*QU_int_err)

    # Compute integrated parameters and associated errors for all pixels
    n_pix = stkI.data[data_mask].size
    I_diluted = stkI.data[data_mask].sum()
    Q_diluted = stkQ.data[data_mask].sum()
    U_diluted = stkU.data[data_mask].sum()
    I_diluted_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[0,0][data_mask]))
    Q_diluted_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[1,1][data_mask]))
    U_diluted_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[2,2][data_mask]))
    IQ_diluted_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[0,1][data_mask]**2))
    IU_diluted_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[0,2][data_mask]**2))
    QU_diluted_err = np.sqrt(n_pix)*np.sqrt(np.sum(stk_cov.data[1,2][data_mask]**2))

    P_diluted = np.sqrt(Q_diluted**2+U_diluted**2)/I_diluted
    P_diluted_err = (1./I_diluted)*np.sqrt((Q_diluted**2*Q_diluted_err**2 + U_diluted**2*U_diluted_err**2 + 2.*Q_diluted*U_diluted*QU_diluted_err)/(Q_diluted**2 + U_diluted**2) + ((Q_diluted/I_diluted)**2 + (U_diluted/I_diluted)**2)*I_diluted_err**2 - 2.*(Q_diluted/I_diluted)*IQ_diluted_err - 2.*(U_diluted/I_diluted)*IU_diluted_err)
    #P_diluted_err = np.sqrt(2/n_pix)*100.

    PA_diluted = princ_angle((90./np.pi)*np.arctan2(U_diluted,Q_diluted))
    PA_diluted_err = (90./(np.pi*(Q_diluted**2 + U_diluted**2)))*np.sqrt(U_diluted**2*Q_diluted_err**2 + Q_diluted**2*U_diluted_err**2 - 2.*Q_diluted*U_diluted*QU_diluted_err)
    #PA_diluted_err = P_diluted_err/(2.*P_diluted)*180./np.pi

    ax.annotate(r"$F_{{\lambda}}^{{int}}$({0:.0f} $\AA$) = {1} $ergs \cdot cm^{{-2}} \cdot s^{{-1}} \cdot \AA^{{-1}}$".format(pivot_wav,sci_not(I_diluted*convert_flux,I_diluted_err*convert_flux,2))+"\n"+r"$P^{{int}}$ = {0:.1f} $\pm$ {1:.1f} %".format(P_diluted*100.,P_diluted_err*100.)+"\n"+r"$\theta_{{P}}^{{int}}$ = {0:.1f} $\pm$ {1:.1f} °".format(PA_diluted,PA_diluted_err), color='white', fontsize=16, xy=(0.01, 0.92), xycoords='axes fraction')

    ax.coords.grid(True, color='white', ls='dotted', alpha=0.5)
    ax.coords[0].set_axislabel('Right Ascension (J2000)')
    ax.coords[0].set_axislabel_position('t')
    ax.coords[0].set_ticklabel_position('t')
    ax.coords[1].set_axislabel('Declination (J2000)')
    ax.coords[1].set_axislabel_position('l')
    ax.coords[1].set_ticklabel_position('l')
    ax.axis('equal')

    if not savename is None:
        #fig.suptitle(savename)
        fig.savefig(plots_folder+savename+".png",bbox_inches='tight',dpi=200)

    plt.show()
    return fig, ax

class align_maps(object):
    """
    Class to interactively align maps with different WCS.
    """
    def __init__(self, Stokes, other_map):
        self.aligned = False
        self.Stokes_UV = Stokes
        self.other_map = other_map

        self.wcs_UV = WCS(self.Stokes_UV[0]).deepcopy()
        convert_flux = self.Stokes_UV[0].header['photflam']
        self.wcs_other = WCS(self.other_map[0]).deepcopy()
        if self.wcs_other.naxis > 2:
            self.wcs_other = WCS(self.other_map[0],naxis=[1,2]).deepcopy()
            self.other_map[0].data = self.other_map[0].data[0,0]
        #Get data
        stkI = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='I_stokes' for i in range(len(self.Stokes_UV))])]
        other_data = self.other_map[0].data

        plt.rcParams.update({'font.size': 16})
        self.fig = plt.figure(figsize=(25,15))
        #Plot the UV map
        self.ax1 = self.fig.add_subplot(121, projection=self.wcs_UV)
        self.ax1.set_facecolor('k')

        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im1 = self.ax1.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)

        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_UV.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax1.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax1.add_artist(px_sc)
        
        north_dir1 = AnchoredDirectionArrows(self.ax1.transAxes, "E", "N", length=-0.08, fontsize=0.03, loc=1, aspect_ratio=-1, sep_y=0.01, sep_x=0.01, angle=-Stokes[0].header['orientat'], color='w', arrow_props={'ec': 'w', 'fc': 'w', 'alpha': 1,'lw': 2})
        self.ax1.add_artist(north_dir1)

        self.cr_UV, = self.ax1.plot(*self.wcs_UV.wcs.crpix, 'r+')

        self.ax1.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="Click on selected point of reference.")

        #Plot the other map
        self.ax2 = self.fig.add_subplot(122, projection=self.wcs_other)
        self.ax2.set_facecolor('k')

        vmin, vmax = 0., np.max(other_data[other_data > 0.])
        im2 = self.ax2.imshow(other_data, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)

        fontprops = fm.FontProperties(size=16)
        px_size = self.wcs_other.wcs.get_cdelt()[0]*3600.
        px_sc = AnchoredSizeBar(self.ax2.transData, 1./px_size, '1 arcsec', 3, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.005, color='w', fontproperties=fontprops)
        self.ax2.add_artist(px_sc)
        
        self.cr_other, = self.ax2.plot(*self.wcs_other.wcs.crpix, 'r+')

        self.ax2.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="Click on selected point of reference.")

        #Selection button
        self.axapply = self.fig.add_axes([0.80, 0.01, 0.1, 0.04])
        self.bapply = Button(self.axapply, 'Apply reference.')
    
    def get_aligned_wcs(self):
        return self.wcs_UV, self.wcs_other

    def onclick_ref(self, event) -> None:
        if self.fig.canvas.manager.toolbar.mode == '':
            if (event.inaxes is not None) and (event.inaxes == self.ax1):
                x = event.xdata
                y = event.ydata

                self.cr_UV.set(data=[x,y])
                self.fig.canvas.draw_idle()
            
            if (event.inaxes is not None) and (event.inaxes == self.ax2):
                x = event.xdata
                y = event.ydata

                self.cr_other.set(data=[x,y])
                self.fig.canvas.draw_idle()
    
    def apply_align(self, event):
        self.wcs_UV.wcs.crpix = np.array(self.cr_UV.get_data())
        self.wcs_other.wcs.crpix = np.array(self.cr_other.get_data())
        self.wcs_other.wcs.crval = self.wcs_UV.wcs.crval
        self.fig.canvas.draw_idle()

        if self.aligned:
            plt.close(self.fig)
    
    def on_close_align(self, event):
        self.aligned = True
        #print(self.get_aligned_wcs())
    
    def align(self):
        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick_ref)
        self.bapply.on_clicked(self.apply_align)
        self.fig.canvas.mpl_connect('close_event', self.on_close_align)
        plt.show(block=True)
        return self.get_aligned_wcs()

class overplot_maps(align_maps):
    """
    Class to overplot maps from different observations.
    Inherit from class align_maps in order to get the same WCS on both maps.
    """
    def overplot(self, other_levels, SNRp_cut=3., SNRi_cut=30., savename=None):
        #Get Data
        obj = self.Stokes_UV[0].header['targname']
        stkI = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='I_stokes' for i in range(len(self.Stokes_UV))])]
        stk_cov = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='IQU_cov_matrix' for i in range(len(self.Stokes_UV))])]
        pol = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_debiased' for i in range(len(self.Stokes_UV))])]
        pol_err = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_deg_err' for i in range(len(self.Stokes_UV))])]
        pang = self.Stokes_UV[np.argmax([self.Stokes_UV[i].header['datatype']=='Pol_ang' for i in range(len(self.Stokes_UV))])]
        
        other_data = self.other_map[0].data
        other_unit = self.other_map[0].header['bunit']
        other_convert = 1.
        if other_unit.lower() == 'jy/beam':
            other_unit = r"mJy/Beam"
            other_convert = 1e3
        other_freq = self.other_map[0].header['crval3']
        
        convert_flux = self.Stokes_UV[0].header['photflam']

        #Compute SNR and apply cuts
        pol.data[pol.data == 0.] = np.nan
        SNRp = pol.data/pol_err.data
        SNRp[np.isnan(SNRp)] = 0.
        pol.data[SNRp < SNRp_cut] = np.nan
        SNRi = stkI.data/np.sqrt(stk_cov.data[0,0])
        SNRi[np.isnan(SNRi)] = 0.
        pol.data[SNRi < SNRi_cut] = np.nan

        plt.rcParams.update({'font.size': 16})
        self.fig2 = plt.figure(figsize=(15,15))
        self.ax = self.fig2.add_subplot(111, projection=self.wcs_UV)
        self.ax.set_facecolor('k')
        self.fig2.subplots_adjust(hspace=0, wspace=0, right=0.9)

        #Display UV intensity map with polarization vectors
        vmin, vmax = 0., np.max(stkI.data[stkI.data > 0.]*convert_flux)
        im = self.ax.imshow(stkI.data*convert_flux, vmin=vmin, vmax=vmax, aspect='auto', cmap='inferno', alpha=1.)
        cbar_ax = self.fig2.add_axes([0.95, 0.12, 0.01, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax, label=r"$F_{\lambda}$ [$ergs \cdot cm^{-2} \cdot s^{-1} \cdot \AA^{-1}$]")

        pol.data[np.isfinite(pol.data)] = 1./2.
        step_vec = 1
        X, Y = np.meshgrid(np.linspace(0,stkI.data.shape[0],stkI.data.shape[0]), np.linspace(0,stkI.data.shape[1],stkI.data.shape[1]))
        U, V = pol.data*np.cos(np.pi/2.+pang.data*np.pi/180.), pol.data*np.sin(np.pi/2.+pang.data*np.pi/180.)
        Q = self.ax.quiver(X[::step_vec,::step_vec],Y[::step_vec,::step_vec],U[::step_vec,::step_vec],V[::step_vec,::step_vec],units='xy',angles='uv',scale=0.5,scale_units='xy',pivot='mid',headwidth=0.,headlength=0.,headaxislength=0.,width=0.1,color='w')
        self.ax.autoscale(False)

        #Display other map as contours
        other_cont = self.ax.contour(other_data*other_convert, transform=self.ax.get_transform(self.wcs_other), levels=other_levels*other_convert, colors='grey')
        self.ax.clabel(other_cont, inline=True, fontsize=8)

        self.ax.set(xlabel="Right Ascension (J2000)", ylabel="Declination (J2000)", title="HST/FOC UV polarization map of {0:s} overplotted with {1:.2f}GHz map in {2:s}.".format(obj, other_freq*1e-9, other_unit))

        if not(savename is None):
            self.fig2.savefig(savename,bbox_inches='tight',dpi=200)

        self.fig2.canvas.draw()
    
    def plot(self, levels, SNRp_cut=3., SNRi_cut=30., savename=None) -> None:
        self.align()
        if self.aligned:
            self.overplot(other_levels=levels, SNRp_cut=SNRp_cut, SNRi_cut=SNRi_cut, savename=savename)
            plt.show(block=True)