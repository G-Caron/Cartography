# ~ # -*- coding: utf-8 -*-

# ~ pip3 uninstall shapely
# ~ pip3 uninstall shapely
# ~ pip3 install shapely --no-binary shapely


# ~ #***********************************************************************
# ~ #                      BASIC EXEMPLE - SAVE IMAGE
# ~ #***********************************************************************
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

# Save the plot by calling plt.savefig() BEFORE plt.show()
plt.savefig('coastlines.pdf')
plt.savefig('coastlines.png')

plt.show()


# ~ #***********************************************************************
# ~ #                    CARTE + INDICATEUR DE POSITION
# ~ #***********************************************************************
import matplotlib
# ~ matplotlib.use("TkAgg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.Robinson())
ax.stock_img()

plt.show()


# ~ #***********************************************************************
# ~ #                    CARTE + VILLE + TRAJET
# ~ #***********************************************************************
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()

ny_lon, ny_lat = -75, 43
delhi_lon, delhi_lat = 77.23, 28.61

plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
         color='blue', linewidth=2, marker='o',
         transform=ccrs.Geodetic(),
         )

plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
         color='gray', linestyle='--',
         transform=ccrs.PlateCarree(),
         )

plt.text(ny_lon - 3, ny_lat - 12, 'New York',
         horizontalalignment='right',
         transform=ccrs.Geodetic())

plt.text(delhi_lon + 3, delhi_lat - 12, 'Delhi',
         horizontalalignment='left',
         transform=ccrs.Geodetic())

plt.show()


# ~ #***********************************************************************
# ~ #         CHANGEMENT DE COORDONNEES - PROJECTION - TRANSFORM
# ~ #***********************************************************************
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


lon = np.linspace(-80, 80, 25)
lat = np.linspace(30, 70, 25)
lon2d, lat2d = np.meshgrid(lon, lat)

data = np.cos(np.deg2rad(lat2d) * 4) + np.sin(np.deg2rad(lon2d) * 4)

# The data are defined in lat/lon coordinate system, so PlateCarree()
# is the appropriate choice:
data_crs = ccrs.PlateCarree()


# The projection keyword determines how the plot will look
plt.figure(figsize=(6, 3))
ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine())
ax.set_global()
ax.coastlines()

ax.contourf(lon, lat, data, transform=data_crs) 
plt.show()


# ~ #***********************************************************************
# ~ #                       LISTE DES FEATURES
# ~ #***********************************************************************
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

plt.figure(figsize=(14,14))
ax = plt.axes(projection=ccrs.Miller())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.STATES)

plt.show()


# ~ #***********************************************************************
# ~ #                    AFFICHAGE AEROPORTS (VEGA_DATASETS)
# ~ #***********************************************************************
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from vega_datasets import data as vds

airports = vds.airports()
airports = airports.iloc[:10]

plt.figure(figsize=(14, 14))
ax = plt.axes(projection=ccrs.PlateCarree())

# (x0, x1, y0, y1)
ax.set_extent([-130, -60, 20, 55], ccrs.PlateCarree())         

ax.add_feature(cfeature.STATES)
ax.coastlines()

for i in airports.itertuples():
    ax.scatter(i.longitude, i.latitude, color='blue', transform=ccrs.PlateCarree())
    plt.text(i.longitude, i.latitude, i.name)

plt.show()


# ~ #***********************************************************************
# ~ #                  LISTE DES PROJECTIONS
# ~ #***********************************************************************
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,18))
fig.suptitle('Projections', fontsize=20, y=0.92)

projections = {'PlateCarree': ccrs.PlateCarree(), 'AlbersEqualArea': ccrs.AlbersEqualArea(), 
               'AzimuthalEquidistant': ccrs.AzimuthalEquidistant(), 'EquidistantConic': ccrs.EquidistantConic(), 
               'LambertConformal': ccrs.LambertConformal(), 'LambertCylindrical': ccrs.LambertCylindrical(), 
               'Mercator': ccrs.Mercator(), 'Miller': ccrs.Miller(), 'Mollweide': ccrs.Mollweide(), 
               'Orthographic': ccrs.Orthographic(), 'Robinson': ccrs.Robinson(), 'Sinusoidal': ccrs.Sinusoidal(), 
               'Stereographic': ccrs.Stereographic(), 'TransverseMercator': ccrs.TransverseMercator(), 
               'InterruptedGoodeHomolosine': ccrs.InterruptedGoodeHomolosine(),
               'RotatedPole': ccrs.RotatedPole(), 'OSGB': ccrs.OSGB(), 'EuroPP': ccrs.EuroPP(), 
               'Geostationary': ccrs.Geostationary(), 'NearsidePerspective': ccrs.NearsidePerspective(), 
               'EckertI': ccrs.EckertI(), 'EckertII': ccrs.EckertII(), 'EckertIII': ccrs.EckertIII(), 
               'EckertIV': ccrs.EckertIV(), 'EckertV': ccrs.EckertV(), 'EckertVI': ccrs.EckertVI(), 
               'Gnomonic': ccrs.Gnomonic(),
               'LambertAzimuthalEqualArea': ccrs.LambertAzimuthalEqualArea(), 
               'NorthPolarStereo': ccrs.NorthPolarStereo(), 'OSNI': ccrs.OSNI(), 
               'SouthPolarStereo': ccrs.SouthPolarStereo()}

for index, projection in enumerate(projections.items()):
    ax = fig.add_subplot(7, 5, index+1, projection=projection[1])
    ax.coastlines()
    ax.set_title(projection[0])

plt.show()


# ~ #***********************************************************************
# ~ #                   CARTE + GRILLE LATITUDE LONGITUDE
# ~ #***********************************************************************
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt

# latitude and longitude with east and west, etc.
plt.figure(figsize=(18, 12))
m8 = plt.axes(projection=ccrs.PlateCarree())
m8.set_extent([-12,30,30,72],ccrs.PlateCarree())

grid_lines = m8.gridlines(draw_labels=True)

# peut etre commenter si on veut les nombre negatif
grid_lines.xformatter = LONGITUDE_FORMATTER
grid_lines.yformatter = LATITUDE_FORMATTER

m8.coastlines(resolution='10m')

plt.show()


# ~ #***********************************************************************
# ~ #                   CARTE + GRILLE LATITUDE LONGITUDE
# ~ #***********************************************************************
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
stamen_terrain = cimgt.Stamen('terrain-background')
ax = plt.axes(projection=stamen_terrain.crs)
# (x0, x1, y0, y1)
ax.set_extent([-130, -60, 20, 55], ccrs.PlateCarree())         
# add map, zoom level
ax.add_image(stamen_terrain, 8)

plt.show()


#***********************************************************************
#         CARTOPY & GEOPANDAS - EXEMPLE CONTOUR ETAT ALLEMAGNE
#***********************************************************************
from cartopy.io import shapereader
import numpy as np
import geopandas
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

ax = plt.axes(projection=ccrs.PlateCarree())

# get country borders
resolution = '10m'
category = 'cultural'
name = 'admin_0_countries'

# get natural earth data (http://www.naturalearthdata.com/)
shpfilename = shapereader.natural_earth(resolution, category, name)

ax.set_extent([5, 16, 47, 56], ccrs.PlateCarree())

# ~ # read the shapefile using geopandas
df = geopandas.read_file(shpfilename)
print(df.iloc[0:10])

# read the german borders
poly = df.loc[df['ADMIN'] == 'Germany']['geometry'].values[0]
print(poly)
ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='blue', edgecolor='0')

plt.show()


# ~ #***********************************************************************
# ~ #                NATURAL EARTH FEATURE - ADD_FEATURE EXAMPLES
# ~ #***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# get country borders
ax = plt.axes(projection=ccrs.PlateCarree())
feat = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '10m',facecolor='g',edgecolor='k')
#ax.add_feature(feat)
#feat = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m',edgecolor='b',linestyle=':')
#feat = cfeature.NaturalEarthFeature('cultural', 'populated_places', '10m')
#feat = cfeature.NaturalEarthFeature('physical', 'lakes', '50m',facecolor='none')
#feat = cfeature.NaturalEarthFeature('physical', 'lakes', '10m')
                                      

ax.add_feature(feat)

ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
#ax.add_feature(cfeature.STATES)
#ax.add_feature(cfeature.LAND)
ax.set_extent([-5, 8, 41, 52], ccrs.PlateCarree())
plt.show()


# ~ #***********************************************************************
# ~ #              NATURAL EARTH FEATURE - RECUPERATION DES SHAPES
# ~ #***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import itertools

shp = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
reader = shpreader.Reader(shp)
countries = reader.records()
country = next(countries)

# ~ # tri par population
population = lambda country: country.attributes['POP_EST'] 
countries_by_pop = sorted(countries, key=population)
plop = ', '.join([ country.attributes['NAME_LONG'] for country in countries_by_pop if (country.attributes['CONTINENT'] == "Africa")]) 
print(plop) 

# ~ # tri par population des pays africain
population = lambda country: country.attributes['POP_EST'] 
countries_by_pop = sorted(countries, key=population)
plop = ', '.join([ country.attributes['NAME_FR'] for country in countries_by_pop if (country.attributes['CONTINENT'] == "Africa")]) 
print(plop) 

# ~ # tri par population et par premieres lettres
first_letter = lambda country: country.attributes['NAME_LONG'][0]
# define a function which returns the population of a country
population = lambda country: country.attributes['POP_EST']

# sort the countries so that the groups come out alphabetically
countries = sorted(reader.records(), key=first_letter)

# group the countries by first letter
for letter, group in itertools.groupby(countries, key=first_letter):
    # print the letter and least populated country
    print(letter, [ c.attributes['NAME_LONG'] for c in  sorted(group, key=population)])


# ~ #***********************************************************************
# ~ #              NATURAL EARTH FEATURE - RECUPERATION DES SHAPES
# ~ #***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd

shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_1_states_provinces')
df = gpd.read_file(shp, encoding='utf-8')

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-5, 8.3, 41.2, 52], ccrs.PlateCarree())

for index, row in df.iterrows():
    print(row['name_fr'])
    if row['admin'] == 'France':
        # Erreur : 'Polygon' object is not iterable, Solution le convertir en liste
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='blue', edgecolor='0')


feat = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '10m',facecolor='none',edgecolor='k')
ax.add_feature(feat)

plt.show()


# ~ #***********************************************************************
# ~ # NATURAL EARTH - RECUPERATION DES SHAPES - LES PROJECTIONS VALABLES
# ~ #***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd

projections = {'PlateCarree': ccrs.PlateCarree(), 'AlbersEqualArea': ccrs.AlbersEqualArea(), 
               'AzimuthalEquidistant': ccrs.AzimuthalEquidistant(), 'EquidistantConic': ccrs.EquidistantConic(), 
               'LambertConformal': ccrs.LambertConformal(), 'LambertCylindrical': ccrs.LambertCylindrical(), 
               'Mercator': ccrs.Mercator(), 'Miller': ccrs.Miller(), 'Mollweide': ccrs.Mollweide(), 
               'Orthographic': ccrs.Orthographic(), 'Robinson': ccrs.Robinson(), 'Sinusoidal': ccrs.Sinusoidal(), 
               'TransverseMercator': ccrs.TransverseMercator(), 
               'Geostationary': ccrs.Geostationary(), 'NearsidePerspective': ccrs.NearsidePerspective(), 
               'EckertI': ccrs.EckertI(),  'EckertIII': ccrs.EckertIII(), 
               'EckertIV': ccrs.EckertIV(), 'EckertV': ccrs.EckertV(), 'EckertVI': ccrs.EckertVI(), 
               'LambertAzimuthalEqualArea': ccrs.LambertAzimuthalEqualArea(), 
               'NorthPolarStereo': ccrs.NorthPolarStereo(), 'SouthPolarStereo': ccrs.SouthPolarStereo()}


shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_1_states_provinces')
df = gpd.read_file(shp)

fig = plt.figure(figsize=(16,18))



feat = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '10m',facecolor='none',edgecolor='k')

for ind, proj in enumerate(projections.items()):
    print("ind",ind, ", projection:",proj[0] )
    ax = fig.add_subplot(4, 6, ind+1, projection=proj[1])
    ax.set_extent([-5, 8, 41, 52], ccrs.PlateCarree())    
    ax.set_title(proj[0])    
    for index, row in df.iterrows():
        if row['admin'] == 'France':
            # Erreur : 'Polygon' object is not iterable, Solution le convertir en liste
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='blue', edgecolor='0')

    ax.add_feature(feat)

plt.savefig('France_projections.png')
plt.show()


# ~ #***********************************************************************
# ~ #    NATURAL EARTH - RECUPERATION DES SHAPES - MEILLEURES PROJECTIONS
# ~ #***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd

projections = {'Miller': ccrs.Miller(),  'Robinson': ccrs.Robinson(), 
               'EckertV': ccrs.EckertV(),
               'LambertAzimuthalEqualArea': ccrs.LambertAzimuthalEqualArea()}


shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_1_states_provinces')
df = gpd.read_file(shp)

fig = plt.figure(figsize=(16,18))



feat = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '10m',facecolor='none',edgecolor='k')

for ind, proj in enumerate(projections.items()):
    print("ind",ind, ", projection:",proj[0] )
    ax = fig.add_subplot(2, 2, ind+1, projection=proj[1])
    ax.set_extent([-5, 8, 41, 52], ccrs.PlateCarree())    
    ax.set_title(proj[0])    
    for index, row in df.iterrows():
        if row['admin'] == 'France':
            # Erreur : 'Polygon' object is not iterable, Solution le convertir en liste
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='blue', edgecolor='0')

    ax.add_feature(feat)

plt.savefig('France_meilleures_projections.png')
plt.show()






