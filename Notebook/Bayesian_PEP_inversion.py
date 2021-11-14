import pmagpy.pmag as pmag
import pmagpy.ipmag as ipmag
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import random
import copy
from scipy.constants import Julian_year

import pymc3 as pm

from pymc3 import distributions
from pymc3.distributions import Continuous, draw_values, generate_samples
import theano.tensor as T
from theano.compile.ops import as_op
from theano import shared
import theano
from pymc3.theanof import floatX

d2r = np.pi/180
r2d = 180/np.pi
eps = 1.e-6


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
    #    assert(np.all(longitude >= 0.) and np.all(longitude <= 360.))
    #    assert(np.all(latitude >= -90.) and np.all(latitude <= 90.))
    #    assert(np.all(norm >= 0.))
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
    
# def atan2(y, x):
    
#     if T.gt(x,0):
#         return np.arctan(y/x)
#     if T.gt(0,x) and T.ge(y,0):
#         return np.arctan(y/x) + np.pi
#     if T.gt(0,x) and T.gt(0,y):
#         return np.arctan(y/x) - np.pi
    
#     if T.eq(x,0) and T.gt(y,0):
#         return np.pi/2
#     if x == 0 and T.gt(0,y):
#         return -np.pi/2
#     if T.eq(x,0) and T.eq(y,0):
#         return

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
            
#             if self.n_euler_rotations == 0 and tpw_rate_scale!= None:
#                 tpw_pole_angle = pm.Uniform('tpw_pole_angle', 0., 360., value=0., observed=False)
#                 tpw_rate = pm.Exponential('tpw_rate', tpw_rate_scale)
                
#             elif self.n_euler_rotations == 1:
#                 euler_1 = VMF('euler_1', lon_lat = [1., 0.], k = k, testval = np.array([1., 0.]), shape = 2)
#                 rate_1 = pm.Exponential('rate_1', euler_rate) 

#                 for i in range(len(self._poles)):
#                     p = self._poles[i]

#                     if p._age_type == 'gaussian':
#                         pole_age = pm.Normal('t'+str(i), mu=self._age_list[i], tau=1/(p._sigma_age**-2))
#                     elif p._age_type == 'uniform':
#                         pole_age = pm.Uniform('t'+str(i), lower=p._sigma_age[0], upper=p._sigma_age[1])

#                     lon_lat = pole_position_1e(start, euler_1, rate_1, pole_age )
#                     observed_pole = VMF('p'+str(i), lon_lat, k = kappa_from_two_sigma(p._A95), observed=[p.longitude, p.latitude])
                    
#             elif self.n_euler_rotations == 2:
                
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
#             for i in range(len(self._poles)):
#                 p = self._poles[i]

# #                 time = self._start_age
#                 if p._age_type == 'gaussian':
#                     pole_age = pm.Normal('t'+str(i), mu=self._age_list[i], tau=1/(p._sigma_age**-2))
#                 elif p._age_type == 'uniform':
#                     pole_age = pm.Uniform('t'+str(i), lower=p._sigma_age[0], upper=p._sigma_age[1])

#                 if self.n_euler_rotations == 2:
#                     lon_lat = pole_position(start, eulers[0], rates[0], eulers[1], rates[1], changes[0], pole_age)
#                 lon_lat = pole_position(start, euler_1, rate_1, euler_2, rate_2, switchpoint, pole_age )
#                 observed_pole = VMF('p'+str(i), lon_lat, k=kappa_from_two_sigma(p._A95), observed=[p.longitude, p.latitude])
#                 observed_pole = VMF('p'+str(i), lon_lat, k = kappa_from_two_sigma(p._A95), observed=[p.longitude, p.latitude])
#             if self.include_tpw:
#                 tpw_pole_angle = pm.Uniform('tpw_pole_angle', lower = 0., upper = 360.)
#                 tpw_rate = pm.Exponential('tpw_rate', tpw_rate_scale)
                   
        return trace
    
# based on poles.py by I. Rose

class Pole(object):
    """
    Class representing a pole on the globe:
    essentially a 3-vector with some additional
    properties and operations.
    """

    def __init__(self, longitude, latitude, magnitude = 1, A95=None):
        """
        Initialize the pole with lon, lat, and A95 uncertainty. Removed norm from Rose version, here we assume everything is unit vector. 
        longitude, latitude, and A95 are all taken in degrees.
        """
        self.longitude = longitude
        self.latitude = latitude
        self.colatitude = 90 - latitude
        self.magnitude = magnitude
        self._pole = spherical_to_cartesian(self.longitude, self.latitude, self.magnitude) # pole position in cartesian coordinates, easier for addition operations
        self._A95 = A95

#     @property
#     def longitude(self):
#         return np.arctan2(self._pole[1], self._pole[0]) * rot.r2d

#     @property
#     def latitude(self):
#         return 90. - np.arccos(self._pole[2] / self.norm) * rot.r2d

#     @property
#     def colatitude(self):
#         return np.arccos(self._pole[2] / self.norm) * rot.r2d

#     @property
#     def norm(self):
#         return np.sqrt(self._pole[0] * self._pole[0] + self._pole[1] * self._pole[1] + self._pole[2] * self._pole[2])

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
        
        lon, lat, intensity = cartesian_to_spherical(p)
        colat = 90. - lat
        m1 = construct_euler_rotation_matrix(-lon * d2r, -colat * d2r, angle * d2r)
        m2 = construct_euler_rotation_matrix(0., colat * d2r, lon * d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))
        self.longitude = cartesian_to_spherical(self._pole.tolist())[0].tolist()[0]
        self.latitude = cartesian_to_spherical(self._pole.tolist())[1].tolist()[0]
        self._pole = spherical_to_cartesian(self.longitude, self.latitude, self.magnitude)
        self.colatitude = 90 - self.latitude

    def _rotate(self, pole, angle):
        print(self.longitude, self.latitude)
        p = pole._pole
        
        lon, lat, intensity = cartesian_to_spherical(p)
        lon = T.as_tensor_variable(lon[0])
        lat = T.as_tensor_variable(lat[0])
        
        colat = 90. - lat
        m1 = construct_euler_rotation_matrix(-lon * d2r, -colat * d2r, angle * d2r)
        m2 = construct_euler_rotation_matrix(0., colat * d2r, lon * d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))
#         return np.array(cartesian_to_spherical(self._pole.tolist())[0], 
#                         cartesian_to_spherical(self._pole.tolist())[1])
        self.longitude = cartesian_to_spherical(self._pole.tolist())[0].tolist()[0]
        self.latitude = cartesian_to_spherical(self._pole.tolist())[1].tolist()[0]
        self._pole = spherical_to_cartesian(self.longitude, self.latitude, self.magnitude)
#         print(self.longitude)
#         print(self.latitude)
#         print(self._pole)
#         print(self.longitude, self.latitude)
        self.colatitude = 90 - self.latitude
        
    def add(self, pole):
        self._pole = self._pole + pole._pole

    # 4/9/2020 YZ corrected the transform for the circle path from PlateCarree to Geodetic so that the poles and circles won't look odd at high latitudes. 
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
            path = matplotlib.path.Path(np.transpose(np.array([lons, lats])))
            circ_patch = matplotlib.patches.PathPatch(
                path, transform=ccrs.PlateCarree(), alpha=0.5, **kwargs)
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

#     if x[1] < -90. or x[1] > 90.:
# #         raise ValueError('input latitude must be within (-90, 90)')
#         return -np.inf
    if k < eps:
        return np.log(1. / 4. / np.pi)
    
    theta = pmag.angle(x, lon_lat)[0]
    PdA = k*np.exp(k*np.cos(theta*d2r))/(2*np.pi*(np.exp(k)-np.exp(-k)))
    logp = np.log(PdA)

    return np.array(logp)


class VMF(Continuous):
    def __init__(self, lon_lat=[0,0], k=None, dtype = np.float64,
                 *args, **kwargs):
        super(VMF, self).__init__(*args, **kwargs)
        
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

        lamda = np.exp(-2*k)

        r1 = np.random.random()
        r2 = np.random.random()
        colat = 2*np.arcsin(np.sqrt(-np.log(r1*(1-lamda)+lamda)/2/k))
        this_lon = 2*np.pi*r2
        lat = 90-colat*r2d
        lon = this_lon*r2d

        unrotated = pmag.dir2cart([lon, lat])[0]
        rotated = np.transpose(np.dot(rotation_matrix, unrotated))
        rotated_dir = pmag.cart2dir(rotated)
        return np.array([rotated_dir[0], rotated_dir[1]])
    
    def random(self, point=None, size=None):
        
        lon_lat, k = draw_values([self._lon_lat, self._k], point=point, size=size)
        return generate_samples(self._random, lon_lat, k,
                                dist_shape=self.shape,
                                size=size)
    



class Watson_Girdle(Continuous):
    def __init__(self, lon_lat=[0,0], k=None, dtype = np.float64, 
                 *args, **kwargs):
        super(Watson_Girdle, self).__init__(*args, **kwargs)
        
        self._lon_lat = T.as_tensor(lon_lat)
        self._k = T.as_tensor_variable(floatX(k))
    
    def logp(self, value):
        lon_lat = self._lon_lat
        k = self._k
        value = T.as_tensor(value)
        
        return watson_girdle_logp(lon_lat, k, value)
    
    def _random(self, lon_lat, k, size = None):
        
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
        rotated = np.transpose(np.dot(rotation_matrix, unrotated))
        rotated_dir = pmag.cart2dir(rotated)
        return np.array([rotated_dir[0], rotated_dir[1]])
    
#         unrotated_samples = np.array([x, y, z])
#         s = np.transpose(np.dot(rotation_matrix, unrotated_samples))

#         lon_lat = np.array([np.arctan2(s[1], s[0]), np.pi /
#                             2. - np.arccos(s[2] / np.sqrt(np.dot(s, s)))]) * r2d
# #         print(lon_lat)
#         return lon_lat
    
    def random(self, point=None, size=None):
        lon_lat = self._lon_lat
        k = self._k
        
        lon_lat, k = draw_values([self._lon_lat, self._k], point=point, size=size)
        return generate_samples(self._random, lon_lat, k,
                                dist_shape=self.shape,
                                size=size)
    
    
@as_op(itypes=[T.dvector, T.dvector, T.dscalar, T.dvector, T.dscalar,  T.dscalar,  T.dscalar], otypes=[T.dvector])
def pole_position_2e( start, euler_1, rate_1, euler_2, rate_2, switchpoint, time ):

    euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)
    euler_pole_2 = EulerPole( euler_2[0], euler_2[1], rate_2)
    start_pole = PaleomagneticPole(start[0], start[1], age=time)

    if time <= switchpoint:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*time)
    else:
        start_pole.rotate( euler_pole_1, euler_pole_1.rate*switchpoint)
        start_pole.rotate( euler_pole_2, euler_pole_2.rate*(time-switchpoint))

    lon_lat = np.array([start_pole.longitude, start_pole.latitude])

    return lon_lat

@as_op(itypes=[T.dvector, T.dvector, T.dscalar, T.dscalar], otypes=[T.dvector])
def pole_position_1e( start, euler_1, rate_1, age ):

    euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)
    start_pole = PaleomagneticPole(start[0], start[1], age=age)

    start_pole.rotate(euler_pole_1, euler_pole_1.rate*age)

    lon_lat = np.array([start_pole.longitude, start_pole.latitude])

    return lon_lat

def plot_trace( trace, lon_lats, age, central_lon = 30., central_lat = 30., num_points_to_plot = 500, num_paths_to_plot = 500, savefig = False, figname = '2_Euler_inversion_test.pdf'):
    def pole_position( start, euler_1, rate_1, euler_2, rate_2, switchpoint, time ):

        euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)
        euler_pole_2 = EulerPole( euler_2[0], euler_2[1], rate_2)
        start_pole = PaleomagneticPole(start[0], start[1], age=time)

        if time <= switchpoint:
            start_pole.rotate( euler_pole_1, euler_pole_1.rate*time)
        else:
            start_pole.rotate( euler_pole_1, euler_pole_1.rate*switchpoint)
            start_pole.rotate( euler_pole_2, euler_pole_2.rate*(time-switchpoint))

        lon_lat = np.array([start_pole.longitude, start_pole.latitude])

        return lon_lat
    
    euler_1_directions = trace.euler_1
    rates_1 = trace.rate_1

    euler_2_directions = trace.euler_2
    rates_2 = trace.rate_2

    start_directions = trace.start
    switchpoints = trace.switchpoint

    interval = max([1,int(len(rates_1)/num_paths_to_plot)])

    #ax = plt.axes(projection = ccrs.Orthographic(0.,30.))
    ax = ipmag.make_orthographic_map(central_lon, central_lat)
    
    ax.scatter(euler_1_directions[:,0][:num_points_to_plot], euler_1_directions[:,1][:num_points_to_plot], transform=ccrs.PlateCarree(), marker = 's', color='b', alpha=0.1)
    ax.scatter(euler_2_directions[:,0][:num_points_to_plot], euler_2_directions[:,1][:num_points_to_plot], transform=ccrs.PlateCarree(), marker = 's', color='r', alpha=0.1)
    
#     ipmag.plot_vgp( ax, euler_1_directions[:,0][:num_points_to_plot], euler_1_directions[:,1][:num_points_to_plot])
#     ipmag.plot_vgp( ax, euler_2_directions[:,0][:num_points_to_plot], euler_2_directions[:,1][:num_points_to_plot])

    age_list = np.linspace(ages[0], ages[-1], num_paths_to_plot)
    pathlons = np.empty_like(age_list)
    pathlats = np.empty_like(age_list)
    for start, e1, r1, e2, r2, switch \
                 in zip(start_directions[::interval], 
                        euler_1_directions[::interval], rates_1[::interval],
                        euler_2_directions[::interval], rates_2[::interval],
                        switchpoints[::interval]):
        for i,a in enumerate(age_list):
            lon_lat = pole_position( start, e1, r1, e2, r2, switch, a)
            pathlons[i] = lon_lat[0]
            pathlats[i] = lon_lat[1]

        ax.plot(pathlons,pathlats,color='b', transform=ccrs.Geodetic(), alpha=0.05)
    for i in range(len(lon_lats)):
        this_pole = Pole(lon_lats[i][0], lon_lats[i][1], A95=5.)
        this_pole.plot(ax, color='C'+str(i))
    if savefig == True:
        plt.savefig(figname)
    plt.show()
    
def plot_trace_1e( trace, lon_lats, ages, central_lon = 30., central_lat = 30., num_points_to_plot = 200, num_paths_to_plot = 200, savefig = False, figname = '1_Euler_inversion_test.pdf'):
    def pole_position( start, euler_1, rate_1, time ):

        euler_pole_1 = EulerPole( euler_1[0], euler_1[1], rate_1)
        start_pole = PaleomagneticPole(start[0], start[1], age=time)

        start_pole.rotate( euler_pole_1, euler_pole_1.rate*time)

        lon_lat = np.array([start_pole.longitude, start_pole.latitude])

        return lon_lat
    
    euler_1_directions = trace.euler_1
    rates_1 = trace.rate_1

    start_directions = trace.start

    interval = max([1,int(len(rates_1)/num_paths_to_plot)])

    ax = ipmag.make_orthographic_map(central_lon, central_lat)
    
    ax.scatter(euler_1_directions[:,0][:num_points_to_plot], euler_1_directions[:,1][:num_points_to_plot], transform=ccrs.PlateCarree(), marker = 's', color='b', alpha=0.1)

    age_list = np.linspace(ages[0], ages[-1], num_paths_to_plot)
    pathlons = np.empty_like(age_list)
    pathlats = np.empty_like(age_list)
    for start, e1, r1 in zip(start_directions[::interval], 
                        euler_1_directions[::interval], rates_1[::interval]):
        for i,a in enumerate(age_list):
            lon_lat = pole_position( start, e1, r1, a)
            pathlons[i] = lon_lat[0]
            pathlats[i] = lon_lat[1]

        ax.plot(pathlons,pathlats,color='b', transform=ccrs.PlateCarree(), alpha=0.05)
    for i in range(len(lon_lats)):
        this_pole = Pole(lon_lats[i][0], lon_lats[i][1], A95=10.)
        this_pole.plot(ax, color='C'+str(i))
    if savefig == True:
        plt.savefig(figname)
    plt.show()