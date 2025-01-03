from itertools import cycle
import pmagpy.pmag as pmag
import pmagpy.ipmag as ipmag
import pandas as pd
import matplotlib
import numpy as np
import numpy.ma as ma
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import random
import copy
from scipy.constants import Julian_year
import scipy.special as sc
import scipy.stats as st
import pymc3 as pm

from pymc3 import distributions
from pymc3.distributions import Continuous, draw_values, generate_samples
import theano.tensor as T
from theano.compile.ops import as_op
from theano import shared
import theano
from pymc3.theanof import floatX
import arviz as az
theano.config.floatX = 'float64'


d2r = np.pi/180
r2d = 180/np.pi
eps = 1.e-6

cmap_blue = plt.get_cmap('Blues')
cmap_red = plt.get_cmap('Reds')
cmap_green = plt.get_cmap('Greens')
cmap_orange = plt.get_cmap('Oranges')
cmap_purple = plt.get_cmap('Purples')
cmap_blue.set_bad('w', alpha=0.0)
cmap_red.set_bad('w', alpha=0.0)
cmap_green.set_bad('w', alpha=0.0)
cmap_orange.set_bad('w', alpha=0.0)
cmap_purple.set_bad('w', alpha=0.0)

cmap_list = [cmap_blue, cmap_red, cmap_green, cmap_orange, cmap_purple]
cmaps = cycle(cmap_list)

def generate_APWP_pole(start_pole, start_age, end_age, euler_pole, euler_rate):
    # this function generates synthetic APWP poles by calculating end pole positions given a start pole position and age, 
    # desired end age, euler rotation pole position and rate
    # calculates end pole position by rotating the start pole with angle = age difference * rotation rate
    
    # all poles should be given in directions, they will be calculated into cartesian coordinates
        
    start_pole_cart = pmag.dir2cart(start_pole)[0]
    euler_pole_cart = pmag.dir2cart(euler_pole)[0]
    
    age_diff = np.abs(end_age - start_age)
    
    end_pole = rotate(start_pole_cart,euler_pole_cart,age_diff*euler_rate)
    
    return pmag.cart2dir(end_pole)[0][:2]

def generate_APWP_poles(number_of_poles, start_pole, start_age, end_age, euler_pole, euler_rate, pole_a95):
    age_step = (start_age-end_age)/(number_of_poles-1)
    ages = np.arange(start_age,end_age-age_step,-age_step)

    pole_lons = []
    pole_lats = []
    pole_a95s = []

    pole_lons.append(start_pole[0])
    pole_lats.append(start_pole[1])
    pole_a95s.append(pole_a95)

    for n in range(1,number_of_poles):
        pole = generate_APWP_pole(start_pole, ages[0], ages[n], euler_pole, euler_rate)
        pole_lons.append(pole[0])
        pole_lats.append(pole[1])
        pole_a95s.append(pole_a95)

    Euler_df = pd.DataFrame(data = np.array([pole_lons, pole_lats, ages, pole_a95s]).T, columns = ['pole_lon', 'pole_lat', 'pole_age', 'pole_a95'])

    return Euler_df

def plot_paleomagnetic_poles(dataframe, pole_lon = 'pole_lon', pole_lat = 'pole_lat', pole_a95 = 'pole_a95', 
                             pole_age = 'pole_age', central_longitude=0, central_latitude=0, cmap = 'viridis_r', **kwargs):
    ax = ipmag.make_orthographic_map(central_longitude, central_latitude, **kwargs)

    ax.set_global()
    ax.gridlines()
    
    cNorm  = matplotlib.colors.Normalize(vmin=min(dataframe[pole_age]), vmax=max(dataframe[pole_age]))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    dataframe['color'] = [colors.rgb2hex(scalarMap.to_rgba(dataframe[pole_age].tolist()[i])) for i in range(dataframe.shape[0])]

    for i in range(dataframe .shape[0]):
        this_pole = Pole(dataframe[pole_lon][i], dataframe[pole_lat][i], A95 = dataframe[pole_a95][i])
        this_pole.plot(ax, color = dataframe['color'][i])
        
    cbar = plt.colorbar(scalarMap, shrink=0.85)
    cbar.ax.set_xlabel('Age (Ma)', fontsize=12) 
    return ax


# @as_op(itypes=[T.dvector, T.dvector, T.dscalar], otypes=[T.dvector])
def rotate(pole, rotation_pole, angle):
    # The idea is to rotate the pole so that the Euler pole is
    # at the pole of the coordinate system, then perform the
    # requested rotation, then restore things to the original
    # orientation
    carttodir = cartesian_to_spherical(rotation_pole)
    lon, lat, intensity = carttodir[0], carttodir[1], carttodir[2]
    
    colat = 90. - lat
    m1 = construct_euler_rotation_matrix(-lon
                                         * d2r, -colat * d2r, angle * d2r)
    m2 = construct_euler_rotation_matrix(0., colat * d2r, lon * d2r)
    return np.dot(m2, np.dot(m1, pole))


def construct_euler_rotation_matrix(alpha, beta, gamma):
    """
    Make a 3x3 matrix which represents a rigid body rotation,
    with alpha being the first rotation about the z axis,
    beta being the second rotation about the y axis, and
    gamma being the third rotation about the z axis.

    All angles are assumed to be in radians
    """
    rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                          [np.sin(alpha), np.cos(alpha), 0.],
                          [0., 0., 1.]])
    rot_beta = np.array([[np.cos(beta), 0., np.sin(beta)],
                         [0., 1., 0.],
                         [-np.sin(beta), 0., np.cos(beta)]])
    rot_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                          [np.sin(gamma), np.cos(gamma), 0.],
                          [0., 0., 1.]])
    rot = np.dot(rot_gamma, np.dot(rot_beta, rot_alpha))
    return rot

def spherical_to_cartesian(longitude, latitude, norm = 1):
    colatitude = 90. - latitude
    return np.array([norm * np.sin(colatitude * d2r) * np.cos(longitude * d2r),
                     norm * np.sin(colatitude * d2r) * np.sin(longitude * d2r),
                     norm * np.cos(colatitude * d2r)])

def atan2(y, x):
    
    if x > 0:
        return np.arctan(y/x)
    if x < 0 and y >= 0:
        return np.arctan(y/x) + np.pi
    if x < 0 and y < 0:
        return np.arctan(y/x) - np.pi
    
    if x == 0 and y > 0:
        return np.pi/2
    if x == 0 and y < 0:
        return -np.pi/2
    if x == y == 0:
        return


def cartesian_to_spherical(vecs):
    v = np.reshape(vecs, (3, -1))
    norm = np.sqrt(v[0, :] * v[0, :] + v[1, :] * v[1, :] + v[2, :] * v[2, :])
    latitude = 90. - np.arccos(v[2, :] / norm) * r2d
    longitude = np.arctan2(v[1, :],v[0, :]) * r2d
    
    return longitude, latitude, norm

# now the longitude will be constrained to be within 0 to 360 degrees
def clamp_longitude( lons ):
    lons = np.asarray(lons)
    lons = np.fmod(lons, 360.)
    lons[np.where(lons < 0)] += 360.
    return lons    

def two_sigma_from_kappa(kappa):
    return 140. / np.sqrt(kappa)


def kappa_from_two_sigma(two_sigma):
    return 140. * 140. / two_sigma / two_sigma

class APWP(object):

    def __init__(self, name, paleomagnetic_pole_list, n_euler_poles, sample_size = 2000):
        for p in paleomagnetic_pole_list:
            assert (isinstance(p, PaleomagneticPole))

        self._name = name
        self._poles = paleomagnetic_pole_list
        self.n_euler_rotations = n_euler_poles

        self._age_list = [p._age for p in self._poles]
        self._start_age = max(self._age_list)
        self._start_pole = self._poles[np.argmax(self._age_list)]
        self._sample_size = sample_size 
    
    def create_model(self, site_lon_lat=[1., 0.], k=1., kw = -1., euler_rate=1., tpw_rate_scale = None):
        assert euler_rate > 0.0, "rate_scale must be a positive number."
        assert tpw_rate_scale == None or tpw_rate_scale > 0.0
        assert kw <= 0.0, "Nonnegative Watson concentration parameters are not supported."
        if tpw_rate_scale is None:
            self.include_tpw = False
        else:
            self.include_tpw = True
            
        with pm.Model() as APWP_model:   
            start = VMF('start', 
                        lon_lat=[self._start_pole.longitude, self._start_pole.latitude], 
                        k=kappa_from_two_sigma(self._start_pole._A95), 
                        testval = np.array([1., 0.]), shape = 2)
                
            euler_1 = VMF('euler_1', lon_lat = site_lon_lat, k = k, testval = np.array([1., 0.]), shape = 2)
            rate_1 = pm.Exponential('rate_1', euler_rate) 
            euler_2 = VMF('euler_2', lon_lat = site_lon_lat, k = k, testval = np.array([1., 0.]), shape = 2)
            rate_2 = pm.Exponential('rate_2', euler_rate) 
            switchpoint = pm.Uniform('switchpoint', min(self._age_list), max(self._age_list))
                
            for i in range(len(self._poles)):
                p = self._poles[i]

                if p._age_type == 'gaussian':
                    pole_age = pm.Normal('t'+str(i), mu=self._age_list[i], tau=1/(p._sigma_age**-2))
                elif p._age_type == 'uniform':
                    pole_age = pm.Uniform('t'+str(i), lower=p._sigma_age[0], upper=p._sigma_age[1])

                lon_lat = pole_position(start, euler_1, rate_1, euler_2, rate_2, switchpoint, pole_age )
                observed_pole = VMF('p'+str(i), lon_lat, k = kappa_from_two_sigma(p._A95), observed=[p.longitude, p.latitude])
                    
            trace = pm.sample(self._sample_size, step = pm.Metropolis())
                   
        return trace
    
# based on poles.py by I. Rose

class Pole(object):
    """
    Class representing a pole on the globe:
    essentially a 3-vector with some additional
    properties and operations.
    """

    def __init__(self, longitude, latitude, magnitude=1.0, A95=None):
        """
        Initialize the pole with lon, lat, and A95 uncertainty. Removed norm from Rose version, here we assume everything is unit vector. 
        longitude, latitude, and A95 are all taken in degrees.
        """

        self._pole = np.ndarray.flatten(spherical_to_cartesian(longitude, latitude, magnitude)) # pole position in cartesian coordinates, easier for addition operations
        self._A95 = A95

    @property
    def longitude(self):
        return np.arctan2(self._pole[1], self._pole[0]) * r2d

    @property
    def latitude(self):
        return 90. - np.arccos(self._pole[2] / self.magnitude) * r2d

    @property
    def colatitude(self):
        return np.arccos(self._pole[2] / self.magnitude) * r2d

    @property
    def magnitude(self):
        return np.sqrt(self._pole[0] * self._pole[0] + self._pole[1] * self._pole[1] + self._pole[2] * self._pole[2])

#     @property
#     def angular_error(self):
#         return self._angular_error

    def copy(self):
        return copy.deepcopy(self)

    def rotate(self, pole, angle):
        # The idea is to rotate the pole about a given pole
        # at the pole of the coordinate system, then perform the
        # requested rotation, then restore things to the original
        # orientation
        
        
        p = pole._pole
        
        lon, lat, mag = cartesian_to_spherical(p)
        colat = 90. - lat
        m1 = construct_euler_rotation_matrix(
            -lon[0] * d2r, -colat[0] * d2r, angle * d2r)
        
        m2 = construct_euler_rotation_matrix(
            0., colat[0] * d2r, lon[0] * d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))
    

    def _rotate(self, pole, angle):
        print(self.longitude, self.latitude)
        p = pole._pole
        
        lon, lat, _ = cartesian_to_spherical(p)
        lon = T.as_tensor_variable(lon[0])
        lat = T.as_tensor_variable(lat[0])
        
        colat = 90. - lat
        m1 = construct_euler_rotation_matrix(-lon * d2r, -colat * d2r, angle * d2r)
        m2 = construct_euler_rotation_matrix(0., colat * d2r, lon * d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))
        self.longitude = cartesian_to_spherical(self._pole.tolist())[0].tolist()[0]
        self.latitude = cartesian_to_spherical(self._pole.tolist())[1].tolist()[0]
        self._pole = spherical_to_cartesian(self.longitude, self.latitude, self.magnitude)

        self.colatitude = 90 - self.latitude
        
    def add(self, pole):
        self._pole = self._pole + pole._pole

    def plot(self, axes, south_pole=False, **kwargs):
        artists = []
        if self._A95 is not None:
            lons = np.linspace(0, 360, 360)
            lats = np.ones_like(lons) * (90. - self._A95)
            magnitudes = np.ones_like(lons)
            
            vecs = spherical_to_cartesian(lons, lats, magnitudes)
            rotation_matrix = construct_euler_rotation_matrix(
                0., (self.colatitude) * d2r, self.longitude * d2r)
            rotated_vecs = np.dot(rotation_matrix, vecs)
            lons, lats, magnitudes = cartesian_to_spherical(rotated_vecs.tolist())
            if south_pole is True:
                lons = lons-180.
                lats = -lats
            path = matplotlib.path.Path(np.array([lons, lats]).T)
            circ_patch = matplotlib.patches.PathPatch(
                path, transform=ccrs.Geodetic(), alpha=0.5, **kwargs)
            circ_artist = axes.add_patch(circ_patch)
            artists.append(circ_artist)
        if south_pole is False:
            artist = axes.scatter(self.longitude, self.latitude,
                                  transform=ccrs.PlateCarree(), **kwargs)
        else:
            artist = axes.scatter(self.longitude-180., -self.latitude,
                                  transform=ccrs.PlateCarree(), **kwargs)
        artists.append(artist)
        return artists

class PaleomagneticPole(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """

    def __init__(self, longitude, latitude, age=0., sigma_age=0.0, **kwargs):

        if np.iterable(sigma_age) == 1:
            assert len(sigma_age) == 2  # upper and lower bounds
            self._age_type = 'uniform'
        else:
            self._age_type = 'gaussian'

        self._age = age
        self._sigma_age = sigma_age

        super(PaleomagneticPole, self).__init__(
            longitude, latitude, 1.0, **kwargs)

#     @property
#     def age_type(self):
#         return self._age_type

#     @property
#     def age(self):
#         return self._age

#     @property
#     def sigma_age(self):
#         return self._sigma_age

class EulerPole(Pole):
    """
    Subclass of Pole which represents an Euler pole.
    The rate is given in deg/Myr
    
    Here we send the rotation rate in radian/sec to the father class as the magnitude. 
    """

    def __init__(self, longitude, latitude, rate, **kwargs):
        r = rate * d2r / Julian_year / 1.e6
        super(EulerPole, self).__init__(longitude, latitude, magnitude = r, **kwargs)

    @property
    def rate(self):
        # returns the angular velocity of the object that is rotating about a given Euler pole
        return self.magnitude * r2d * Julian_year * 1.e6

    def angle(self, time):
        return self.rate * time

    def speed_at_point(self, pole):
        """
        Given a point, calculate the speed that point
        rotates around the Euler pole. This assumes that
        the test pole has a radius equal to the radius of Earth,
        6371.e3 meters. It returns the speed in cm/yr.
        """
        # Give the point the radius of the earth
        point = pole._pole
        point = point / np.sqrt(np.dot(point, point)) * 6371.e3
#         print(np.array([point[0], point[1], point[2]]))
        # calculate the speed
        vel = np.cross(self._pole, np.array([point[0], point[1], point[2]]))    
        speed = np.sqrt(np.dot(vel, vel))

        return speed * Julian_year * 100.


class PlateCentroid(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """

    def __init__(self, longitude, latitude, **kwargs):
        super(PlateCentroid, self).__init__(
            longitude, latitude, 6371.e3, **kwargs)

def two_sigma_from_kappa(kappa):
    return 140. / np.sqrt(kappa)


def kappa_from_two_sigma(two_sigma):
    return 140. * 140. / two_sigma / two_sigma

@as_op(itypes=[T.dvector, T.dscalar, T.dvector], otypes=[T.dscalar])
def vmf_logp(lon_lat, k, x):

    if x[1] < -90. or x[1] > 90.:
#         raise ZeroProbability
        return x[1]%90

    if k < eps:
        return np.log(1. / 4. / np.pi)

    mu = np.array([np.cos(lon_lat[1] * d2r) * np.cos(lon_lat[0] * d2r),
                   np.cos(lon_lat[1] * d2r) * np.sin(lon_lat[0] * d2r),
                   np.sin(lon_lat[1] * d2r)])
    test_point = np.transpose(np.array([np.cos(x[1] * d2r) * np.cos(x[0] * d2r),
                                        np.cos(x[1] * d2r) *
                                        np.sin(x[0] * d2r),
                                        np.sin(x[1] * d2r)]))

    logp_elem = np.log( -k / ( 2. * np.pi * np.expm1(-2. * k)) ) + \
        k * (np.dot(test_point, mu) - 1.)

    logp = logp_elem.sum()
    return np.array(logp)

def fisher_logp(lon_lat, k, x):

    if x[1] < -90. or x[1] > 90.:
#         raise ZeroProbability
        return x[1]%90

    if k < eps:
        return np.log(1. / 4. / np.pi)

    mu = np.array([np.cos(lon_lat[1] * d2r) * np.cos(lon_lat[0] * d2r),
                   np.cos(lon_lat[1] * d2r) * np.sin(lon_lat[0] * d2r),
                   np.sin(lon_lat[1] * d2r)])
    test_point = np.transpose(np.array([np.cos(x[1] * d2r) * np.cos(x[0] * d2r),
                                        np.cos(x[1] * d2r) *
                                        np.sin(x[0] * d2r),
                                        np.sin(x[1] * d2r)]))

    logp_elem = np.log( -k / ( 2. * np.pi * np.expm1(-2. * k)) ) + \
        k * (np.dot(test_point, mu) - 1.)

    logp = logp_elem.sum()
    return np.array(logp)

class VMF(Continuous):
    def __init__(self, lon_lat=[0,0], k=None, dtype = np.float64,
                 *args, **kwargs):
        super(VMF, self).__init__(*args, **kwargs)
        if k < eps:
            k = np.log(1. / 4. / np.pi)
        
        self._k = T.as_tensor_variable(floatX(k))
        self._lon_lat = T.as_tensor_variable(lon_lat)
    
    def logp(self, value):
        lon_lat = self._lon_lat
        k = self._k
        value = T.as_tensor(value)   

        return vmf_logp(lon_lat, k, value)
    
    
    def _random(self, lon_lat, k, size = None):

        alpha = 0.
        beta = np.pi / 2. - lon_lat[1] * d2r
        gamma = lon_lat[0] * d2r

        rotation_matrix = construct_euler_rotation_matrix(alpha, beta, gamma)
        
        zeta = st.uniform.rvs(loc=0., scale=1.)
        if k < eps:
            z = np.array([2. * zeta - 1.])
        else:
            z = 1. + 1. / k * \
                np.log(zeta + (1. - zeta) * np.exp(-2. * k))

        # x and y coordinates can be determined by a
        # uniform distribution in longitude.
        phi = st.uniform.rvs(loc=0., scale=2. * np.pi)
        x = np.sqrt(1. - z * z) * np.cos(phi)
        y = np.sqrt(1. - z * z) * np.sin(phi)
        
        # Rotate the samples to have the correct mean direction
        unrotated_samples = np.array((x, y, z))
        rotated = np.transpose(np.dot(rotation_matrix, unrotated_samples))
        rotated_dir = pmag.cart2dir(rotated)[0]
        return np.array([rotated_dir[0], rotated_dir[1]])
    
    def random(self, point=None, size=None):
        
        lon_lat, k = draw_values([self._lon_lat, self._k], point=point, size=size)
        return generate_samples(self._random, lon_lat, k,
                                dist_shape=self.shape,
                                size=size)
    

@as_op(itypes=[T.dvector, T.dscalar, T.dvector], otypes=[T.dscalar])
def watson_girdle_logp(lon_lat, k, x):
    
    if k > 0:
        raise ValueError('k has to be negative!')
        return 
    if k == 0:
        return np.log(1. / 4. / np.pi)
    
    theta = pmag.angle(x, lon_lat)[0]
    pw = 1/sc.hyp1f1(1/2, 3/2, k)/4/np.pi*np.exp(k*np.cos(theta*d2r)**2)
    log_pw = np.log(pw)
    
    return np.array(log_pw)

class Watson_Girdle(Continuous):
    def __init__(self, lon_lat=[0,0], k=None, dtype = np.float64, 
                 *args, **kwargs):
        super(Watson_Girdle, self).__init__(*args, **kwargs)
        if k == 0:
            k = np.log(1. / 4. / np.pi)
            
        self._lon_lat = T.as_tensor(lon_lat)
        self._k = T.as_tensor_variable(floatX(k))
    
    def logp(self, value):
        lon_lat = self._lon_lat
        k = self._k
        value = T.as_tensor(value)
        
        return watson_girdle_logp(lon_lat, k, value)
    
    def _random(self, lon_lat, k, size = None):
        if np.abs(k) < 0:
            k = np.log(1. / 4. / np.pi)
        beta = np.pi / 2. - lon_lat[1] * d2r
        gamma = lon_lat[0] * d2r
        rotation_matrix = construct_euler_rotation_matrix(0, beta, gamma)

        C1 = np.sqrt(abs(k))
        C2 = np.arctan(C1)

        this_lon = 0
        this_lat = 0

        i = 0
        while i < 1:
            U = np.random.random()
            V = np.random.random()
            S = 1/C1*np.tan(C2*U)
            r0 = np.random.random()

            if V < (1-k*S**2)*np.exp(k*S**2):
                this_lon = 0
                this_lat = 0
                pos_neg = 0
                colat = np.arccos(S)
                this_lon = 2*np.pi*r0
                pos_neg = random.choice([-1, 1])
                this_lon = this_lon*r2d
                this_lat = pos_neg*(90-colat*r2d)
                i = i + 1

        x = np.cos(this_lon*d2r)*np.cos(this_lat*d2r)
        y = np.sin(this_lon*d2r)*np.cos(this_lat*d2r)
        z = np.sin(this_lat*d2r)
        
        unrotated = pmag.dir2cart([this_lon, this_lat])[0]
        if np.abs(k) < eps:
            return np.array([this_lon, this_lat])
        else:
            rotated = np.transpose(np.dot(rotation_matrix, unrotated))
            rotated_dir = pmag.cart2dir(rotated)
        
            return np.array([rotated_dir[0], rotated_dir[1]])

    
    def random(self, point=None, size=None):
        lon_lat = self._lon_lat
        k = self._k
        
        lon_lat, k = draw_values([self._lon_lat, self._k], point=point, size=size)
        return generate_samples(self._random, lon_lat, k,
                                dist_shape=self.shape,
                                size=size)

@as_op(itypes=[T.dvector, T.dscalar, T.dmatrix], otypes=[T.dscalar])
def watsongirdlelogp(lon_lat, k, x):
    
    if k > 0:
        raise ValueError('k has to be negative!')
        return 
    if k == 0:
        return np.log(1. / 4. / np.pi)
    
    theta = np.array([pmag.angle(i, lon_lat) for i in x])
    pw = 1/sc.hyp1f1(1/2, 3/2, k)/4/np.pi*np.exp(k*np.cos(theta*d2r)**2)
    log_pw = np.sum(np.log(pw))
    
    return np.array(log_pw)

class WatsonGirdle(Continuous):
    def __init__(self, lon_lat=[0,0], k=None, dtype = np.float64, 
                 *args, **kwargs):
        super(WatsonGirdle, self).__init__(*args, **kwargs)
        if k == 0:
            k = np.log(1. / 4. / np.pi)
            
        self._lon_lat = T.as_tensor(lon_lat)
        self._k = T.as_tensor_variable(floatX(k))
    
    def logp(self, value):
        lon_lat = self._lon_lat
        k = self._k
        value = T.as_tensor(value)
        
        return watsongirdlelogp(lon_lat, k, value)
    
    def _random(self, lon_lat, k, size = None):
        if np.abs(k) < 0:
            k = np.log(1. / 4. / np.pi)
        beta = np.pi / 2. - lon_lat[1] * d2r
        gamma = lon_lat[0] * d2r
        rotation_matrix = construct_euler_rotation_matrix(0, beta, gamma)

        C1 = np.sqrt(abs(k))
        C2 = np.arctan(C1)

        this_lon = 0
        this_lat = 0

        i = 0
        while i < 1:
            U = np.random.random()
            V = np.random.random()
            S = 1/C1*np.tan(C2*U)
            r0 = np.random.random()

            if V < (1-k*S**2)*np.exp(k*S**2):
                this_lon = 0
                this_lat = 0
                pos_neg = 0
                colat = np.arccos(S)
                this_lon = 2*np.pi*r0
                pos_neg = random.choice([-1, 1])
                this_lon = this_lon*r2d
                this_lat = pos_neg*(90-colat*r2d)
                i = i + 1

        x = np.cos(this_lon*d2r)*np.cos(this_lat*d2r)
        y = np.sin(this_lon*d2r)*np.cos(this_lat*d2r)
        z = np.sin(this_lat*d2r)
        
        unrotated = pmag.dir2cart([this_lon, this_lat])[0]
        if np.abs(k) < eps:
            return np.array([this_lon, this_lat])
        else:
            rotated = np.transpose(np.dot(rotation_matrix, unrotated))
            rotated_dir = pmag.cart2dir(rotated)
        
            return np.array([rotated_dir[0], rotated_dir[1]])

    
    def random(self, point=None, size=None):
        lon_lat = self._lon_lat
        k = self._k
        
        lon_lat, k = draw_values([self._lon_lat, self._k], point=point, size=size)
        return generate_samples(self._random, lon_lat, k,
                                dist_shape=self.shape,
                                size=size)
    
    
@as_op(itypes=[T.dvector, T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dvector])
def pole_position_tpw(start, tpw_angle, tpw_rate, start_age, age):
    
    start_pole = PaleomagneticPole(start[0], start[1], age=start_age)
        
    # make a TPW pole
    test_1 = np.array([0.,0.,1.])
    test_2 = np.array([1.,0.,0.])
    if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
        great_circle_pole = np.cross(start_pole._pole, test_2)
    else:
        great_circle_pole = np.cross(start_pole._pole, test_1)
    lon, lat, _ = cartesian_to_spherical(great_circle_pole)

    TPW = EulerPole(lon[0], lat[0], tpw_rate)
    TPW.rotate(start_pole, tpw_angle)

    start_pole.rotate(TPW, TPW.rate*(start_age-age))
    lon_lat = np.ndarray.flatten(np.array([start_pole.longitude, start_pole.latitude]))

    return lon_lat


def plot_trace_tpw(trace, lon_lats, A95s, ages, central_lon = 30., central_lat = 30., num_paths_to_plot = 200, 
                  savefig = False, figname = 'code_output/tpw_inversion_.pdf',
                   path_resolution=16, scatter=0, posterior_n=100, calc_posterior_likelihood=True, calc_pole_path=False, arbitrary_pole_ages=[], **kwargs):
    def pole_position(start, tpw_angle, tpw_rate, start_age, age):

        start_pole = PaleomagneticPole(start[0], start[1], age=start_age)

        # make a TPW pole
        test_1 = np.array([0.,0.,1.])
        test_2 = np.array([1.,0.,0.])
        if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
            great_circle_pole = np.cross(start_pole._pole, test_2)
        else:
            great_circle_pole = np.cross(start_pole._pole, test_1)
        lon, lat, _ = cartesian_to_spherical(great_circle_pole)

        TPW = EulerPole(lon[0], lat[0], tpw_rate)
        TPW.rotate(start_pole, tpw_angle)
#         print(start_age-age)
        start_pole.rotate(TPW, TPW.rate*(start_age-age))
        lon_lat = np.ndarray.flatten(np.array([start_pole.longitude, start_pole.latitude]))

        return lon_lat
    
    tpw_angle = trace.tpw_angle
    tpw_rate = trace.tpw_rate
    
    start_age = trace.start_pole_age
    start_directions = trace.start_pole

    interval = max(1, int(len(trace.start_pole_age[:]) / num_paths_to_plot))

    ax = ipmag.make_orthographic_map(central_lon, central_lat, add_land=0, grid_lines = 1)
                
    age_list = np.linspace(max(ages), min(ages), path_resolution)
    pathlats = np.zeros(path_resolution)
    pathlons = np.zeros(path_resolution)
    
    tpw_directions = np.empty_like(trace.start_pole[:])
    index=0
    for start, tpw_a , tpw_r in zip(start_directions, tpw_angle, tpw_rate):
        test_1 = np.array([0.,0.,1.])
        test_2 = np.array([1.,0.,0.])
        start_pole = PaleomagneticPole(start[0], start[1], 1.0)
        if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
            great_circle_pole = np.cross(start_pole._pole, test_2)
        else:
            great_circle_pole = np.cross(start_pole._pole, test_1)
        lon, lat, _ = cartesian_to_spherical(great_circle_pole)
        TPW = EulerPole(lon[0], lat[0], tpw_r)
        TPW.rotate(start_pole, tpw_a)
        tpw_directions[index, :] = np.ndarray.flatten(np.array([TPW.longitude, TPW.latitude]))
        index += 1

    plot_distributions(ax, tpw_directions[:,0], tpw_directions[:,1], cmap=kwargs.get('cmap', 'Reds'), resolution=kwargs.get('resolution', 100))
    
    if scatter == False:
        for start, tpw_a, tpw_r, start_a in zip(start_directions[::interval],  
                                                tpw_angle[::interval], tpw_rate[::interval], 
                                                start_age[::interval]):

            for i,a in enumerate(age_list):
                lon_lat = pole_position( start, tpw_a, tpw_r, start_a, a)
                pathlons[i] = lon_lat[0]
                pathlats[i] = lon_lat[1]

            ax.plot(pathlons,pathlats,color='b', transform=ccrs.Geodetic(), alpha=0.05)
    
    cNorm  = matplotlib.colors.Normalize(vmin=min(ages), vmax=max(ages))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='viridis_r')
    arbitrary_pole_position=[]
    if scatter == True:
        posterior_interval= max([1,int(len(tpw_rate)/posterior_n)])
        for start, tpw_a, tpw_r, start_a in zip(start_directions[::posterior_interval],  
                                                tpw_angle[::posterior_interval], tpw_rate[::posterior_interval], 
                                                start_age[::posterior_interval]):
            if calc_pole_path:
                for i in range(len(arbitrary_pole_ages)):
                    this_age = arbitrary_pole_ages[i]
                    lon_lat = pole_position( start, tpw_a, tpw_r, start_a, this_age)
                    arbitrary_pole_position.append(lon_lat)
            else: 
                for i in range(len(ages)):
                    this_age = trace['t'+str(i)][::posterior_interval][i]
                    lon_lat = pole_position( start, tpw_a, tpw_r, start_a, this_age)
                    ax.scatter(lon_lat[0],lon_lat[1],color=colors.rgb2hex(scalarMap.to_rgba(this_age)), transform=ccrs.PlateCarree(), alpha=0.05)
    
    posterior_likelihood_list = []
    if calc_posterior_likelihood:
        for start, tpw_a, tpw_r, start_a in zip(start_directions,  
                                                tpw_angle, tpw_rate, 
                                                start_age):
            posterior_likelihood = 0
            for i in range(len(ages)):
                this_age = trace['t'+str(i)][::posterior_interval][i]
                lon_lat = pole_position( start, tpw_a, tpw_r, start_a, this_age)
                posterior_likelihood += fisher_logp(lon_lat=lon_lats[i], k=kappa_from_two_sigma(A95s[i]), x=lon_lat)
            posterior_likelihood_list.append(posterior_likelihood)
            
    # plot paleomagnetic observation poles here

    pole_colors = [colors.rgb2hex(scalarMap.to_rgba(ages[i])) for i in range(len(ages))]
    
    if kwargs.get('colorbar', True):
        cbar = plt.colorbar(scalarMap, shrink=0.85, location='bottom', pad=0.01)
        cbar.ax.set_xlabel('Age (Ma)', fontsize=12) 
    for i in range(len(lon_lats)):
        this_pole = Pole(lon_lats[i][0], lon_lats[i][1], A95=A95s[i])
        this_pole.plot(ax, color=pole_colors[i])
    if savefig == True:
        plt.savefig(figname, dpi=600, bbox_inches='tight')

    if calc_pole_path:
        return ax, posterior_likelihood_list, np.vstack([np.array(arbitrary_pole_position)[i::len(arbitrary_pole_ages)] for i in range(len(arbitrary_pole_ages))]).reshape(len(arbitrary_pole_ages), posterior_n, 2)
    return ax, posterior_likelihood_list  
    

@as_op(itypes=[T.dvector, T.dvector, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dvector])
def pole_position_1e( start, euler_1, rate_1, start_age, age ):
    
    start_pole = PaleomagneticPole(start[0], start[1], age=start_age)
    
    euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)

    start_pole.rotate(euler_pole_1, euler_pole_1.rate*(start_age-age))

    lon_lat = np.array([start_pole.longitude, start_pole.latitude])
    return lon_lat


def plot_trace_1e(trace, lon_lats, A95s,  ages, central_lon = 30., central_lat = 30., num_paths_to_plot = 200, 
                  savefig = False, figname = 'code_output/1_Euler_inversion_.pdf',
                  path_resolution=100, estimate_pole_age = 0, scatter=0, posterior_n=100, calc_posterior_likelihood=True, calc_pole_path=False, arbitrary_pole_ages=[], **kwargs):
    def pole_position( start, euler_1, rate_1, start_age, age ):

        start_pole = PaleomagneticPole(start[0], start[1], age=start_age)

        euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)

        start_pole.rotate(euler_pole_1, euler_pole_1.rate*(start_age-age))

        lon_lat = np.array([start_pole.longitude, start_pole.latitude])
        return lon_lat
    
    euler_1_directions = trace.euler_1
    rates_1 = trace.rate_1

    start_directions = trace.start_pole
    start_ages = trace.start_pole_age
    
    interval = max([1,int(len(rates_1)/num_paths_to_plot)])

    ax = ipmag.make_orthographic_map(central_lon, central_lat, add_land=0, grid_lines = 1)
    
    plot_distributions(ax, euler_1_directions[:,0], euler_1_directions[:,1], cmap=kwargs.get('cmap', 'Blues'), resolution=kwargs.get('resolution', 100))
    
    age_list = np.linspace(min(ages), max(ages), path_resolution)
    pathlons = np.empty_like(age_list)
    pathlats = np.empty_like(age_list)
    
    
    if scatter == False:
        for start, e1, r1, start_a in zip(start_directions[::interval], 
                            euler_1_directions[::interval], rates_1[::interval], 
                                 start_ages[:interval]):
            for i,a in enumerate(age_list):
                lon_lat = pole_position( start, e1, r1, start_a, a)
                pathlons[i] = lon_lat[0]
                pathlats[i] = lon_lat[1]

            ax.plot(pathlons,pathlats,color='b', transform=ccrs.Geodetic(), alpha=0.05)
            
    cNorm  = matplotlib.colors.Normalize(vmin=min(ages), vmax=max(ages))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='viridis_r')
    posterior_interval= max([1,int(len(rates_1)/posterior_n)])
    
    arbitrary_pole_position = []
    if scatter == True:
        
        for start, e1, r1, start_a in zip(start_directions[::posterior_interval], 
                            euler_1_directions[::posterior_interval], rates_1[::posterior_interval], 
                                 start_ages[::posterior_interval]):
            if calc_pole_path:
                for i in range(len(arbitrary_pole_ages)):
                    this_age = arbitrary_pole_ages[i]
                    lon_lat = pole_position( start, e1, r1, start_a, this_age)
                    arbitrary_pole_position.append(lon_lat)
            else: 
                for i in range(len(ages)):
                    this_age = trace['t'+str(i)][::posterior_interval][i]
                    lon_lat = pole_position( start, e1, r1, start_a, this_age)
                    ax.scatter(lon_lat[0],lon_lat[1],color=colors.rgb2hex(scalarMap.to_rgba(this_age)), transform=ccrs.PlateCarree(), alpha=0.05)
                
    posterior_likelihood_list = []
    if calc_posterior_likelihood:
        for start, e1, r1, start_a in zip(start_directions, 
                            euler_1_directions, rates_1, 
                                 start_ages):
            posterior_likelihood = 0
            for i in range(len(ages)):
                this_age = trace['t'+str(i)][::posterior_interval][i]           
                lon_lat = pole_position( start, e1, r1, start_a, this_age)
                posterior_likelihood += fisher_logp(lon_lat=lon_lats[i], k=kappa_from_two_sigma(A95s[i]), x=lon_lat)
            posterior_likelihood_list.append(posterior_likelihood)
            
    # plot paleomagnetic observation poles here
    
    if estimate_pole_age:
        pole_colors = [colors.rgb2hex(scalarMap.to_rgba(np.median(trace['t'+str(i)]))) for i in range(len(ages))]
    
    else:
        pole_colors = [colors.rgb2hex(scalarMap.to_rgba(ages[i])) for i in range(len(ages))]
        
    if kwargs.get('colorbar', True):
        cbar = plt.colorbar(scalarMap, shrink=0.85, location='bottom', pad=0.01)
        cbar.ax.set_xlabel('Age (Ma)', fontsize=12) 
    for i in range(len(lon_lats)):
        this_pole = Pole(lon_lats[i][0], lon_lats[i][1], A95=A95s[i])
        this_pole.plot(ax, color=pole_colors[i])
    if savefig == True:
        plt.savefig(figname,dpi=600,bbox_inches='tight')
        
    if calc_pole_path:
        return ax, posterior_likelihood_list, np.vstack([np.array(arbitrary_pole_position)[i::len(arbitrary_pole_ages)] for i in range(len(arbitrary_pole_ages))]).reshape(len(arbitrary_pole_ages), posterior_n, 2)
    return ax, posterior_likelihood_list
    
    
@as_op(itypes=[T.dvector, T.dvector, T.dscalar, T.dvector, T.dscalar,  T.dscalar,  T.dscalar, T.dscalar], otypes=[T.dvector])
def pole_position_2e( start, euler_1, rate_1, euler_2, rate_2, switchpoint, start_age, age ):

    euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)
    euler_pole_2 = EulerPole( euler_2[0], euler_2[1], rate_2)
    start_pole = PaleomagneticPole(start[0], start[1], age=start_age)

    if age > switchpoint:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*(start_age-age))
    else:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*(start_age-switchpoint))
        start_pole.rotate( euler_pole_2, euler_pole_2.rate*(switchpoint-age))

    lon_lat = np.ndarray.flatten(np.array([start_pole.longitude, start_pole.latitude]))

    return lon_lat


def plot_trace_2e( trace, lon_lats, A95s, ages, central_lon = 30., central_lat = 30., num_paths_to_plot = 500, 
                  savefig = True, figname = '2_Euler_inversion_test.pdf', 
                  path_resolution=100, scatter=0, posterior_n=100, calc_posterior_likelihood=True, calc_pole_path=False, arbitrary_pole_ages=[], **kwargs):
    def pole_position( start, euler_1, rate_1, euler_2, rate_2, switchpoint, start_age, age ):

        euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)
        euler_pole_2 = EulerPole( euler_2[0], euler_2[1], rate_2)
        start_pole = PaleomagneticPole(start[0], start[1], age=start_age)

        if age > switchpoint:
            start_pole.rotate( euler_pole_1, euler_pole_1.rate*(start_age-age))
        else:
            start_pole.rotate( euler_pole_1, euler_pole_1.rate*(start_age-switchpoint))
            start_pole.rotate( euler_pole_2, euler_pole_2.rate*(switchpoint-age))

        lon_lat = np.ndarray.flatten(np.array([start_pole.longitude, start_pole.latitude]))

        return lon_lat
    
    euler_1_directions = trace.euler_1
    rates_1 = trace.rate_1

    euler_2_directions = trace.euler_2
    rates_2 = trace.rate_2

    start_directions = trace.start_pole
    start_ages = trace.start_pole_age
    switchpoints = trace.switchpoint
    
    interval = max([1,int(len(rates_1)/num_paths_to_plot)])

    #ax = plt.axes(projection = ccrs.Orthographic(0.,30.))
    ax = ipmag.make_orthographic_map(central_lon, central_lat, add_land=kwargs.get('add_land',0), grid_lines = 1)
    
    plot_distributions(ax, euler_1_directions[:,0], euler_1_directions[:,1], cmap = kwargs.get('cmap_1', 'Blues'), resolution=kwargs.get('resolution', 100))
    plot_distributions(ax, euler_2_directions[:,0], euler_2_directions[:,1], cmap = kwargs.get('cmap_2', 'Greens'), resolution=kwargs.get('resolution', 100))
    
    age_list = np.linspace(min(ages), max(ages), path_resolution)
    pathlons = np.empty_like(age_list)
    pathlats = np.empty_like(age_list)
    
    if scatter == False:
        for start, e1, r1, e2, r2, switch, start_age \
                     in zip(start_directions[::interval], 
                            euler_1_directions[::interval], rates_1[::interval],
                            euler_2_directions[::interval], rates_2[::interval],
                            switchpoints[::interval], start_ages[::interval]):
            for i,a in enumerate(age_list):
                lon_lat = pole_position( start, e1, r1, e2, r2, switch, start_age, a)
                pathlons[i] = lon_lat[0]
                pathlats[i] = lon_lat[1]

            old_lons = [pathlons[i] if age_list[i] > switch else None for i in range(len(age_list))]
            old_lats = [pathlats[i] if age_list[i] > switch else None for i in range(len(age_list))]
            young_lons = [pathlons[i] if age_list[i] <= switch else None for i in range(len(age_list))]
            young_lats = [pathlats[i] if age_list[i] <= switch else None for i in range(len(age_list))]

            ax.plot(young_lons,young_lats,color='r', transform=ccrs.Geodetic(), alpha=0.05)
            ax.plot(old_lons,old_lats,color='b', transform=ccrs.Geodetic(), alpha=0.05)
    
    cNorm  = matplotlib.colors.Normalize(vmin=min(ages), vmax=max(ages))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='viridis_r')
    arbitrary_pole_position=[]
    if scatter == True:
        posterior_interval= max([1,int(len(rates_1)/posterior_n)])
        for start, e1, r1, e2, r2, switch, start_age \
                     in zip(start_directions[::posterior_interval], 
                            euler_1_directions[::posterior_interval], rates_1[::posterior_interval],
                            euler_2_directions[::posterior_interval], rates_2[::posterior_interval],
                            switchpoints[::posterior_interval], start_ages[::posterior_interval]):
            if calc_pole_path:
                for i in range(len(arbitrary_pole_ages)):
                    this_age = arbitrary_pole_ages[i]
                    lon_lat = pole_position( start, e1, r1, e2, r2, switch, start_age, this_age)
                    arbitrary_pole_position.append(lon_lat)
            else: 
                for i in range(len(ages)):
                    this_age = trace['t'+str(i)][::posterior_interval][i]
                    lon_lat = pole_position( start, e1, r1, e2, r2, switch, start_age, this_age)
                    ax.scatter(lon_lat[0],lon_lat[1],color=colors.rgb2hex(scalarMap.to_rgba(this_age)), transform=ccrs.PlateCarree(), alpha=0.05)
    
    posterior_likelihood_list = []
    if calc_posterior_likelihood:
        for start, e1, r1, e2, r2, switch, start_age \
                     in zip(start_directions, 
                            euler_1_directions, rates_1,
                            euler_2_directions, rates_2,
                            switchpoints, start_ages):
            posterior_likelihood = 0  
            for i in range(len(ages)):
                this_age = trace['t'+str(i)][::posterior_interval][i]
                lon_lat = pole_position( start, e1, r1, e2, r2, switch, start_age, this_age)
                posterior_likelihood += fisher_logp(lon_lat=lon_lats[i], k=kappa_from_two_sigma(A95s[i]), x=lon_lat)
            posterior_likelihood_list.append(posterior_likelihood)
            
    # plot paleomagnetic observation poles here

    pole_colors = [colors.rgb2hex(scalarMap.to_rgba(ages[i])) for i in range(len(ages))]
        
    if kwargs.get('colorbar', True):
        cbar = plt.colorbar(scalarMap, shrink=0.85, location='bottom', pad=0.01)
        cbar.ax.set_xlabel('Age (Ma)', fontsize=12)  
    for i in range(len(lon_lats)):
        this_pole = Pole(lon_lats[i][0], lon_lats[i][1], A95=A95s[i])
        this_pole.plot(ax, color=pole_colors[i])
    if savefig == True:
        plt.savefig(figname,dpi=600,bbox_inches='tight')
    if calc_pole_path:
        return ax, posterior_likelihood_list, np.vstack([np.array(arbitrary_pole_position)[i::len(arbitrary_pole_ages)] for i in range(len(arbitrary_pole_ages))]).reshape(len(arbitrary_pole_ages), posterior_n, 2)
    return ax, posterior_likelihood_list
    
@as_op(itypes=[T.dvector, T.dvector, T.dscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dvector])
def pole_position_1e_tpw(start, euler_1, rate_1, tpw_angle, tpw_rate, start_age, age):
    
    start_pole = PaleomagneticPole(start[0], start[1], age=start_age)
    
    euler_pole_1 = EulerPole(euler_1[0], euler_1[1], rate_1)
    
    # make a TPW pole
    test_1 = np.array([0.,0.,1.])
    test_2 = np.array([1.,0.,0.])
    if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
        great_circle_pole = np.cross(start_pole._pole, test_2)
    else:
        great_circle_pole = np.cross(start_pole._pole, test_1)
    lon, lat, _ = cartesian_to_spherical(great_circle_pole)

    TPW = EulerPole(lon[0], lat[0], tpw_rate)
    TPW.rotate(start_pole, tpw_angle)
    
    this_euler = euler_pole_1.copy()
    start_pole.rotate(this_euler, this_euler.rate*(start_age-age))
    
#     this_euler.add(TPW)
    
    start_pole.rotate(TPW, TPW.rate*(start_age-age))

    lon_lat = np.ndarray.flatten(np.array([start_pole.longitude, start_pole.latitude]))

    return lon_lat


def plot_trace_1e_tpw(trace, lon_lats, A95s, ages, central_lon = 30., central_lat = 30., num_paths_to_plot = 200, 
                  savefig = False, figname = 'code_output/1_Euler_inversion_.pdf', 
                      path_resolution=100, scatter=0, posterior_n=100, calc_posterior_likelihood=True, calc_pole_path=False, arbitrary_pole_ages=[], **kwargs):
    def pole_position(start, euler_1, rate_1, tpw_angle, tpw_rate, start_age, age):

        start_pole = PaleomagneticPole(start[0], start[1], age=start_age)

        euler_pole_1 = EulerPole(euler_1[0], euler_1[1], rate_1)

        # make a TPW pole
        test_1 = np.array([0.,0.,1.])
        test_2 = np.array([1.,0.,0.])
        if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
            great_circle_pole = np.cross(start_pole._pole, test_2)
        else:
            great_circle_pole = np.cross(start_pole._pole, test_1)
        lon, lat, _ = cartesian_to_spherical(great_circle_pole)

        TPW = EulerPole(lon[0], lat[0], tpw_rate)
        TPW.rotate(start_pole, tpw_angle)

        this_euler = euler_pole_1.copy()
        start_pole.rotate(this_euler, this_euler.rate*(start_age-age))
    
#     this_euler.add(TPW)
    
        start_pole.rotate(TPW, TPW.rate*(start_age-age))

        lon_lat = np.ndarray.flatten(np.array([start_pole.longitude, start_pole.latitude]))

        return lon_lat
    
    euler_1_directions = trace.euler_1
    euler_rates_1 = trace.rate_1
    
    tpw_angle = trace.tpw_angle
    tpw_rate = trace.tpw_rate
    
    start_age = trace.start_pole_age
    start_directions = trace.start_pole

    interval = max([1,int(len(euler_rates_1)/num_paths_to_plot)])
    
    ax = ipmag.make_orthographic_map(central_lon, central_lat, add_land=0, grid_lines = 1)
    
    plot_distributions(ax, euler_1_directions[:,0], euler_1_directions[:,1], cmap=kwargs.get('cmap', 'Blues'), resolution=kwargs.get('resolution', 100))
            
    age_list = np.linspace(ages[0], ages[-1], path_resolution)
    pathlons = np.empty_like(age_list)
    pathlats = np.empty_like(age_list)
    
    tpw_directions = np.empty_like(trace.start_pole[:])
    index=0
    for start, tpw_a in zip(start_directions, tpw_angle):
        test_1 = np.array([0.,0.,1.])
        test_2 = np.array([1.,0.,0.])
        start_pole = Pole(start[0], start[1], 1.0)
        if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
            great_circle_pole = np.cross(start_pole._pole, test_2)
        else:
            great_circle_pole = np.cross(start_pole._pole, test_1)
        lon, lat, _ = cartesian_to_spherical(great_circle_pole)
        TPW = Pole(lon[0], lat[0], 1.0)
        TPW.rotate(start_pole, tpw_a)
        tpw_directions[index, :] = np.ndarray.flatten(np.array([TPW.longitude, TPW.latitude]))
        index += 1

    plot_distributions(ax, tpw_directions[:,0], tpw_directions[:,1], cmap=kwargs.get('cmap', 'Reds'), resolution=kwargs.get('resolution', 100))
   
    
    if scatter ==  False:
        for start, e1, r1, tpw_a, tpw_r, start_a in zip(start_directions[::interval], 
                            euler_1_directions[::interval], euler_rates_1[::interval], 
                            tpw_angle[::interval], tpw_rate[::interval], start_age[::interval]):

                for i,a in enumerate(age_list):
                    lon_lat = pole_position( start, e1, r1, tpw_a, tpw_r, start_a, a)
                    pathlons[i] = lon_lat[0]
                    pathlats[i] = lon_lat[1]

                ax.plot(pathlons,pathlats,color='b', transform=ccrs.Geodetic(), alpha=0.05)
                
    cNorm  = matplotlib.colors.Normalize(vmin=min(ages), vmax=max(ages))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='viridis_r')
    arbitrary_pole_position=[]
    if scatter == True:
        posterior_interval= max([1,int(len(euler_rates_1)/posterior_n)])
        for start, e1, r1, tpw_a, tpw_r, start_a in zip(start_directions[::posterior_interval], 
                        euler_1_directions[::posterior_interval], euler_rates_1[::posterior_interval], 
                        tpw_angle[::posterior_interval], tpw_rate[::posterior_interval], start_age[::posterior_interval]):
            if calc_pole_path:
                for i in range(len(arbitrary_pole_ages)):
                    this_age = arbitrary_pole_ages[i]
                    lon_lat = pole_position( start, e1, r1, tpw_a, tpw_r, start_a, this_age)
                    arbitrary_pole_position.append(lon_lat)
            else:                                         
                for i in range(len(ages)):
                    this_age = trace['t'+str(i)][::posterior_interval][i]
                    lon_lat = pole_position( start, e1, r1, tpw_a, tpw_r, start_a, this_age)
                    ax.scatter(lon_lat[0],lon_lat[1],color=colors.rgb2hex(scalarMap.to_rgba(this_age)), transform=ccrs.PlateCarree(), alpha=0.05)
    
    posterior_likelihood_list = []
    if calc_posterior_likelihood:
        for start, e1, r1, tpw_a, tpw_r, start_a in zip(start_directions, 
                            euler_1_directions, euler_rates_1, 
                            tpw_angle, tpw_rate, start_age):
            posterior_likelihood = 0                                               
            for i in range(len(ages)):
                this_age = trace['t'+str(i)][::posterior_interval][i]
                lon_lat = pole_position( start, e1, r1, tpw_a, tpw_r, start_a, this_age)
                posterior_likelihood += fisher_logp(lon_lat=lon_lats[i], k=kappa_from_two_sigma(A95s[i]), x=lon_lat)
            posterior_likelihood_list.append(posterior_likelihood)
                                                          
    # plot paleomagnetic observation poles here

    pole_colors = [colors.rgb2hex(scalarMap.to_rgba(ages[i])) for i in range(len(ages))]
        
    if kwargs.get('colorbar', True):
        cbar = plt.colorbar(scalarMap, shrink=0.85, location='bottom', pad=0.01)
        cbar.ax.set_xlabel('Age (Ma)', fontsize=12)  
    for i in range(len(lon_lats)):
        this_pole = Pole(lon_lats[i][0], lon_lats[i][1], A95=A95s[i])
        this_pole.plot(ax, color=pole_colors[i])
    if savefig == True:
        plt.savefig(figname,dpi=600, bbox_inches='tight')
    if calc_pole_path:
        return ax, posterior_likelihood_list, np.vstack([np.array(arbitrary_pole_position)[i::len(arbitrary_pole_ages)] for i in range(len(arbitrary_pole_ages))]).reshape(len(arbitrary_pole_ages), posterior_n, 2)
    return ax, posterior_likelihood_list


    
@as_op(itypes=[T.dvector, T.dvector, T.dscalar, T.dvector, T.dscalar, T.dscalar, T.dscalar, T.dscalar,  T.dscalar, T.dscalar], otypes=[T.dvector])
def pole_position_2e_tpw(start, euler_1, rate_1, euler_2, rate_2, tpw_angle, tpw_rate, switchpoint, start_age, age):
    
    start_pole = PaleomagneticPole(start[0], start[1], age=start_age)
    
    euler_pole_1 = EulerPole(euler_1[0], euler_1[1], rate_1)
    euler_pole_2 = EulerPole(euler_2[0], euler_2[1], rate_2)
    
    # make a TPW pole
    test_1 = np.array([0.,0.,1.])
    test_2 = np.array([1.,0.,0.])
    if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
        great_circle_pole = np.cross(start_pole._pole, test_2)
    else:
        great_circle_pole = np.cross(start_pole._pole, test_1)
    lon, lat, _ = cartesian_to_spherical(great_circle_pole)

    TPW = EulerPole(lon[0], lat[0], tpw_rate)
    TPW.rotate(start_pole, tpw_angle)
    
    if age >= switchpoint:
        this_euler_1 = euler_pole_1.copy()
        start_pole.rotate(this_euler_1, euler_pole_1.rate*(start_age-age))
        start_pole.rotate(TPW, TPW.rate*(start_age-age))
    else:
        this_euler_1 = euler_pole_1.copy()
        this_euler_2 = euler_pole_2.copy()

        start_pole.rotate(euler_pole_1, euler_pole_1.rate*(start_age-switchpoint))
        start_pole.rotate(TPW, TPW.rate*(start_age-switchpoint))
        start_pole.rotate(euler_pole_2, euler_pole_2.rate*(switchpoint-age))
        start_pole.rotate(TPW, TPW.rate*(switchpoint-age))
    lon_lat = np.ndarray.flatten(np.array([start_pole.longitude, start_pole.latitude]))

    return lon_lat


def plot_trace_2e_tpw(trace, lon_lats, A95s, ages, central_lon = 30., central_lat = 30., num_points_to_plot = 200, num_paths_to_plot = 200, 
                  savefig = False, figname = 'code_output/1_Euler_inversion_.pdf',
                      path_resolution = 100, scatter=0, posterior_n=1000, calc_posterior_likelihood=True, calc_pole_path=False, arbitrary_pole_ages=[], **kwargs):
    def pole_position(start, euler_1, rate_1, euler_2, rate_2, tpw_angle, tpw_rate, switchpoint, start_age, age):

        start_pole = PaleomagneticPole(start[0], start[1], age=start_age)

        euler_pole_1 = EulerPole(euler_1[0], euler_1[1], rate_1)
        euler_pole_2 = EulerPole(euler_2[0], euler_2[1], rate_2)

        # make a TPW pole
        test_1 = np.array([0.,0.,1.])
        test_2 = np.array([1.,0.,0.])
        if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
            great_circle_pole = np.cross(start_pole._pole, test_2)
        else:
            great_circle_pole = np.cross(start_pole._pole, test_1)
        lon, lat, _ = cartesian_to_spherical(great_circle_pole)

        TPW = EulerPole(lon[0], lat[0], tpw_rate)
        TPW.rotate(start_pole, tpw_angle)

        if age >= switchpoint:
            this_euler_1 = euler_pole_1.copy()
            start_pole.rotate( this_euler_1, euler_pole_1.rate*(start_age-age))
            start_pole.rotate(TPW, TPW.rate*(start_age-age))
        else:
            this_euler_1 = euler_pole_1.copy()
            this_euler_2 = euler_pole_1.copy()

            start_pole.rotate( euler_pole_1, euler_pole_1.rate*(start_age-switchpoint))
            start_pole.rotate(TPW, TPW.rate*(start_age-switchpoint))
            start_pole.rotate( euler_pole_2, euler_pole_2.rate*(switchpoint-age))
            start_pole.rotate(TPW, TPW.rate*(switchpoint-age))
        lon_lat = np.ndarray.flatten(np.array([start_pole.longitude, start_pole.latitude]))

        return lon_lat
    
    euler_1_directions = trace.euler_1
    euler_rates_1 = trace.rate_1
    euler_2_directions = trace.euler_2
    euler_rates_2 = trace.rate_2
    
    tpw_angle = trace.tpw_angle
    tpw_rate = trace.tpw_rate
    
    start_age = trace.start_pole_age
    start_directions = trace.start_pole
    
    switchpoints = trace.switchpoint

    interval = max([1,int(len(euler_rates_1)/num_paths_to_plot)])

    ax = ipmag.make_orthographic_map(central_lon, central_lat, add_land=0, grid_lines = 1)
    
    plot_distributions(ax, euler_1_directions[:,0], euler_1_directions[:,1], cmap=kwargs.get('cmap', 'Blues'), resolution=kwargs.get('resolution', 100))
    plot_distributions(ax, euler_2_directions[:,0], euler_2_directions[:,1], cmap=kwargs.get('cmap', 'Greens'), resolution=kwargs.get('resolution', 100))
    
    age_list = np.linspace(ages[0], ages[-1], path_resolution)
    pathlons = np.empty_like(age_list)
    pathlats = np.empty_like(age_list)
    
    tpw_directions = np.empty_like(trace.start_pole[:])
    index=0
    for start, tpw_a in zip(start_directions, tpw_angle):
        test_1 = np.array([0.,0.,1.])
        test_2 = np.array([1.,0.,0.])
        start_pole = Pole(start[0], start[1], 1.0)
        if np.dot(start_pole._pole, test_1) > np.dot(start_pole._pole, test_2):
            great_circle_pole = np.cross(start_pole._pole, test_2)
        else:
            great_circle_pole = np.cross(start_pole._pole, test_1)
        lon, lat, _ = cartesian_to_spherical(great_circle_pole)
        TPW = Pole(lon[0], lat[0], 1.0)
        TPW.rotate(start_pole, tpw_a)
        tpw_directions[index, :] = np.ndarray.flatten(np.array([TPW.longitude, TPW.latitude]))
        index += 1

    plot_distributions(ax, tpw_directions[:,0], tpw_directions[:,1], cmap=kwargs.get('cmap', 'Reds'), resolution=kwargs.get('resolution', 100))
    
    
    if scatter == False:
        for start, e1, r1, e2, r2, tpw_a, tpw_r, switchpoint, start_a in zip(start_directions[::interval], 
                            euler_1_directions[::interval], euler_rates_1[::interval],
                            euler_2_directions[::interval], euler_rates_2[::interval],
                            tpw_angle[::interval], tpw_rate[::interval], switchpoints[::interval], start_age[::interval]):

                for i,a in enumerate(age_list):
                    lon_lat = pole_position( start, e1, r1, e2, r2, tpw_a, tpw_r, switchpoint, start_a, a)
                    pathlons[i] = lon_lat[0]
                    pathlats[i] = lon_lat[1]
                old_lons = [pathlons[i] if age_list[i] > switchpoint else None for i in range(len(age_list))]
                old_lats = [pathlats[i] if age_list[i] > switchpoint else None for i in range(len(age_list))]
                young_lons = [pathlons[i] if age_list[i] <= switchpoint else None for i in range(len(age_list))]
                young_lats = [pathlats[i] if age_list[i] <= switchpoint else None for i in range(len(age_list))]

                ax.plot(young_lons,young_lats,color='r', transform=ccrs.Geodetic(), alpha=0.05)
                ax.plot(old_lons,old_lats,color='b', transform=ccrs.Geodetic(), alpha=0.05)
                
    cNorm  = matplotlib.colors.Normalize(vmin=min(ages), vmax=max(ages))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='viridis_r')
    arbitrary_pole_position=[]
    if scatter == True:
        posterior_interval= max([1,int(len(euler_rates_1)/posterior_n)])
        for start, e1, r1, e2, r2, tpw_a, tpw_r, switchpoint, start_a in zip(start_directions[::posterior_interval], 
                    euler_1_directions[::posterior_interval], euler_rates_1[::posterior_interval],
                    euler_2_directions[::posterior_interval], euler_rates_2[::posterior_interval],
                    tpw_angle[::posterior_interval], tpw_rate[::posterior_interval], switchpoints[::posterior_interval], start_age[::posterior_interval]):
            
            if calc_pole_path:
                for i in range(len(arbitrary_pole_ages)):
                    this_age = arbitrary_pole_ages[i]
                    lon_lat = pole_position( start, e1, r1, e2, r2, tpw_a, tpw_r, switchpoint, start_a, this_age)
                    arbitrary_pole_position.append(lon_lat)
            else: 
                for i in range(len(ages)):
                    this_age = trace['t'+str(i)][::posterior_interval][i]
                    lon_lat = pole_position( start, e1, r1, e2, r2, tpw_a, tpw_r, switchpoint, start_a, this_age)
                    ax.scatter(lon_lat[0],lon_lat[1],color=colors.rgb2hex(scalarMap.to_rgba(this_age)), transform=ccrs.PlateCarree(), alpha=0.05)
   
    posterior_likelihood_list = []        
    if calc_posterior_likelihood:        
        for start, e1, r1, e2, r2, tpw_a, tpw_r, switchpoint, start_a in zip(start_directions, 
                        euler_1_directions, euler_rates_1,
                        euler_2_directions, euler_rates_2,
                        tpw_angle, tpw_rate, switchpoints, start_age):
            posterior_likelihood=0
            for i in range(len(ages)):
                this_age = trace['t'+str(i)][::posterior_interval][i]
                lon_lat = pole_position( start, e1, r1, e2, r2, tpw_a, tpw_r, switchpoint, start_a, this_age)
                posterior_likelihood += fisher_logp(lon_lat=lon_lats[i], k=kappa_from_two_sigma(A95s[i]), x=lon_lat)
            posterior_likelihood_list.append(posterior_likelihood)
                
    # plot paleomagnetic observation poles here


    pole_colors = [colors.rgb2hex(scalarMap.to_rgba(ages[i])) for i in range(len(ages))]
        
    if kwargs.get('colorbar', True):
        cbar = plt.colorbar(scalarMap, shrink=0.85, location='bottom', pad=0.01)
        cbar.ax.set_xlabel('Age (Ma)', fontsize=12) 
    for i in range(len(lon_lats)):
        this_pole = Pole(lon_lats[i][0], lon_lats[i][1], A95=A95s[i])
        this_pole.plot(ax, color=pole_colors[i])
    if savefig == True:
        plt.savefig(figname, dpi=600,bbox_inches='tight')
    if calc_pole_path:
        return ax, posterior_likelihood_list, np.vstack([np.array(arbitrary_pole_position)[i::len(arbitrary_pole_ages)] for i in range(len(arbitrary_pole_ages))]).reshape(len(arbitrary_pole_ages), posterior_n, 2)
    return ax, posterior_likelihood_list
    
def bin_trace(lon_samples, lat_samples, resolution):
    """
    Given a trace of samples in longitude and latitude, bin them
    in latitude and longitude, and normalize the bins so that
    the integral of probability density over the sphere is one.

    The resolution keyword gives the number of divisions in latitude.
    The divisions in longitude is twice that.
    """
    lats = np.linspace(-90., 90., resolution, endpoint=True)
    lons = np.linspace(-180., 180., 2 * resolution, endpoint=True)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    hist = np.zeros_like(lon_grid)

    dlon = 360. / (2. * resolution)
    dlat = 180. / resolution

    for lon, lat in zip(lon_samples, lat_samples):

        lon = np.mod(lon, 360.)
        if lon > 180.:
            lon = lon - 360.
        if lat < -90. or lat > 90.:
            # Just skip invalid latitudes if they happen to arise
            continue

        lon_index = int(np.floor((lon + 180.) / dlon))
        lat_index = int(np.floor((lat + 90.) / dlat))
        hist[lat_index, lon_index] += 1

    lat_grid += dlat / 2.
    lon_grid += dlon / 2.
    return lon_grid, lat_grid, hist


def density_distribution(lon_samples, lat_samples, resolution=30):
    count = len(lon_samples)
    lon_grid, lat_grid, hist = bin_trace(lon_samples, lat_samples, resolution)
    return lon_grid, lat_grid, hist / count


def cumulative_density_distribution(lon_samples, lat_samples, resolution=30):

    lon_grid, lat_grid, hist = bin_trace(lon_samples, lat_samples, resolution)

    # Compute the cumulative density
    hist = hist.ravel()
    i_sort = np.argsort(hist)[::-1]
    i_unsort = np.argsort(i_sort)
    hist_cumsum = hist[i_sort].cumsum()
    hist_cumsum /= hist_cumsum[-1]

    return lon_grid, lat_grid, hist_cumsum[i_unsort].reshape(lat_grid.shape)


def plot_distributions(ax, lon_samples, lat_samples, to_plot='d', resolution=100, **kwargs):

    cmap=kwargs.get('cmap', 'Blues')

    artists = []

    if 'd' in to_plot:
        lon_grid, lat_grid, density = density_distribution(
            lon_samples, lat_samples, resolution)
        density = ma.masked_where(density <= 0.05*density.max(), density)
        a = ax.pcolormesh(lon_grid, lat_grid, density, cmap=cmap,
                          transform=ccrs.PlateCarree())
        artists.append(a)

    if 'e' in to_plot:
        lon_grid, lat_grid, cumulative_density = cumulative_density_distribution(
            lon_samples, lat_samples, resolution)
        a = ax.contour(lon_grid, lat_grid, cumulative_density, levels=[
                       0.683, 0.955], cmap=cmap, transform=ccrs.PlateCarree())
        artists.append(a)

    if 's' in to_plot:
        a = ax.scatter(lon_samples, lat_samples, color=cmap(
            [0., 0.5, 1.])[-1], alpha=0.1, transform=ccrs.PlateCarree(), edgecolors=None)
        artists.append(a)

    return artists