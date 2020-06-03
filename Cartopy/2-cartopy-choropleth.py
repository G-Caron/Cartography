# ~ # -*- coding: utf-8 -*-

pip3 uninstall shapely
pip3 uninstall shapely
pip3 install shapely --no-binary shapely


# ~ #***********************************************************************
# ~ #                       CHOROPLETH SUR LATITUDE
# ~ #***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd
import numpy as np

def dms2dd( data ) :
    part = data['Latitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LatitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)
    
    part = data['Longitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LongitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)

data = pd.read_csv("BaseDeDonnee/pays.csv") 
print(data.head())

shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
df = gpd.read_file(shp,encoding='UTF-8')

# Recherche des pays de data qui ne sont pas dans df
for index, row in data.iterrows():
    if row['Pays'].casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    elif str(row['Nom autre']).casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    else:
        print(index, row['Pays'])


fig, ax = plt.subplots(figsize=(18, 12), subplot_kw=dict(projection=ccrs.EckertIII()))
ax.set_position([0.05, 0.05, 0.85, 0.85])

# Conversion latitude/longitude DMS en DD
dms2dd(data)

# Creation de la colormap
col='LatitudeDD'
mn = min(data[col])
mx = max(data[col])
norm = plt.Normalize(vmin=mn, vmax=mx)
cmapName = plt.cm.ScalarMappable(norm=norm, cmap='Blues')


# Boucle sur les pays
for index, row in df.iterrows():
    
    if row['NAME_FR'] != None:
            
        if row['NAME_FR'].casefold() in (data['Pays'].str.casefold()).tolist():
            c = cmapName.to_rgba( ( data[data['Pays'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
            # lon = data[data['Pays'] == row['NAME_FR']]['LongitudeDD']
            # lat = data[data['Pays'] == row['NAME_FR']]['LatitudeDD']
            # plt.text(lon, lat, row['NAME_FR'], horizontalalignment='right', transform=ccrs.Geodetic())

        elif row['NAME_FR'].casefold() in (data['Nom autre'].str.casefold()).tolist():
            c = cmapName.to_rgba( ( data[data['Nom autre'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
            # lon = data[data['Nom autre'] == row['NAME_FR']]['LongitudeDD']
            # lat = data[data['Nom autre'] == row['NAME_FR']]['LatitudeDD']            
            # plt.text(lon, lat, row['NAME_FR'], horizontalalignment='right', transform=ccrs.Geodetic())

        else:
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='r', edgecolor='0')
            print(index, "Le pays ", row['NAME_FR'], "n a pas de correspondant")


cbaxes = fig.add_axes([0.91, 0.062, 0.03, 0.826]) 
fig.colorbar(cmapName, cax=cbaxes, orientation='vertical')
ax.set_title('Latitude', fontdict=dict(fontsize=24))
plt.savefig('choropleth-Latitude.png')
plt.show()


#***********************************************************************
#                       CHOROPLETH SUR LONGITUDE
#***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd
import numpy as np

def dms2dd( data ) :
    part = data['Latitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LatitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)
    
    part = data['Longitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LongitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)

data = pd.read_csv("BaseDeDonnee/pays.csv") 
print(data.head())

shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
df = gpd.read_file(shp,encoding='UTF-8')

# Recherche des pays de data qui ne sont pas dans df
for index, row in data.iterrows():
    if row['Pays'].casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    elif str(row['Nom autre']).casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    else:
        print(index, row['Pays'])

fig, ax = plt.subplots(figsize=(18, 12), subplot_kw=dict(projection=ccrs.EckertIII()))
ax.set_position([0.05, 0.05, 0.85, 0.85])


# Conversion latitude/longitude DMS en DD
dms2dd(data)

# Creation de la colormap
col='LongitudeDD'
mn = min(data[col])
mx = max(data[col])
norm = plt.Normalize(vmin=mn, vmax=mx)
cmapName = plt.cm.ScalarMappable(norm=norm, cmap='Blues')


# Boucle sur les pays
for index, row in df.iterrows():
    
    if row['NAME_FR'] != None:
            
        if row['NAME_FR'].casefold() in (data['Pays'].str.casefold()).tolist():
            c = cmapName.to_rgba( ( data[data['Pays'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
            # lon = data[data['Pays'] == row['NAME_FR']]['LongitudeDD']
            # lat = data[data['Pays'] == row['NAME_FR']]['LatitudeDD']
            # plt.text(lon, lat, row['NAME_FR'], horizontalalignment='right', transform=ccrs.Geodetic())

        elif row['NAME_FR'].casefold() in (data['Nom autre'].str.casefold()).tolist():
            c = cmapName.to_rgba( ( data[data['Nom autre'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
            # lon = data[data['Nom autre'] == row['NAME_FR']]['LongitudeDD']
            # lat = data[data['Nom autre'] == row['NAME_FR']]['LatitudeDD']            
            # plt.text(lon, lat, row['NAME_FR'], horizontalalignment='right', transform=ccrs.Geodetic())

        else:
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='r', edgecolor='0')
            print(index, "Le pays ", row['NAME_FR'], "n a pas de correspondant")

cbaxes = fig.add_axes([0.91, 0.062, 0.03, 0.826]) 
fig.colorbar(cmapName, cax=cbaxes, orientation='vertical')
ax.set_title('Longitude', fontdict=dict(fontsize=24))
plt.savefig('choropleth-Longitude.png')
plt.show()



#***********************************************************************
#                       CHOROPLETH SUR IDH
#***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd
import numpy as np

def dms2dd( data ) :
    part = data['Latitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LatitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)
    
    part = data['Longitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LongitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)

data = pd.read_csv("BaseDeDonnee/pays.csv") 
print(data.head())

shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
df = gpd.read_file(shp,encoding='UTF-8')

# Recherche des pays de data qui ne sont pas dans df
for index, row in data.iterrows():
    if row['Pays'].casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    elif str(row['Nom autre']).casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    else:
        print(index, row['Pays'])

fig, ax = plt.subplots(figsize=(18, 12), subplot_kw=dict(projection=ccrs.EckertIII()))
ax.set_position([0.05, 0.05, 0.85, 0.85])


# Conversion latitude/longitude DMS en DD
dms2dd(data)

# Creation de la colormap
col='IDH'
mn = min(data[col])
mx = max(data[col])
norm = plt.Normalize(vmin=mn, vmax=mx)
cmapName = plt.cm.ScalarMappable(norm=norm, cmap='Blues')


# Boucle sur les pays
for index, row in df.iterrows():
    
    if row['NAME_FR'] != None:
            
        if row['NAME_FR'].casefold() in (data['Pays'].str.casefold()).tolist():
            c = cmapName.to_rgba( ( data[data['Pays'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
            # lon = data[data['Pays'] == row['NAME_FR']]['LongitudeDD']
            # lat = data[data['Pays'] == row['NAME_FR']]['LatitudeDD']
            # plt.text(lon, lat, row['NAME_FR'], horizontalalignment='right', transform=ccrs.Geodetic())

        elif row['NAME_FR'].casefold() in (data['Nom autre'].str.casefold()).tolist():
            c = cmapName.to_rgba( ( data[data['Nom autre'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
            # lon = data[data['Nom autre'] == row['NAME_FR']]['LongitudeDD']
            # lat = data[data['Nom autre'] == row['NAME_FR']]['LatitudeDD']            
            # plt.text(lon, lat, row['NAME_FR'], horizontalalignment='right', transform=ccrs.Geodetic())

        else:
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='r', edgecolor='0')
            print(index, "Le pays ", row['NAME_FR'], "n a pas de correspondant")

cbaxes = fig.add_axes([0.91, 0.062, 0.03, 0.826]) 
fig.colorbar(cmapName, cax=cbaxes, orientation='vertical')
ax.set_title('IDH', fontdict=dict(fontsize=24))
plt.savefig('choropleth-IDH.png')
plt.show()



#***********************************************************************
#                       CHOROPLETH SUR POP2018
#***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd
import numpy as np
import math

def dms2dd( data ) :
    part = data['Latitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LatitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)
    
    part = data['Longitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LongitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)

data = pd.read_csv("BaseDeDonnee/pays.csv") 
print(data.head())

shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
df = gpd.read_file(shp,encoding='UTF-8')

# Recherche des pays de data qui ne sont pas dans df
for index, row in data.iterrows():
    if row['Pays'].casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    elif str(row['Nom autre']).casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    else:
        print(index, row['Pays'])
        
fig, ax = plt.subplots(figsize=(18, 12), subplot_kw=dict(projection=ccrs.EckertIII()))
ax.set_position([0.05, 0.05, 0.85, 0.85])


# Conversion latitude/longitude DMS en DD
dms2dd(data)

# Creation de la colormap
col='Pop2018'
mn = math.log(min(data[col]))
mx = math.log(max(data[col]))
print(mn,mx)
norm = plt.Normalize(vmin=mn, vmax=mx)
cmapName = plt.cm.ScalarMappable(norm=norm, cmap='Blues')


# Boucle sur les pays
for index, row in df.iterrows():
    
    if row['NAME_FR'] != None:
            
        if row['NAME_FR'].casefold() in (data['Pays'].str.casefold()).tolist():
            c = cmapName.to_rgba( math.log( data[data['Pays'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
            # ~ # lon = data[data['Pays'] == row['NAME_FR']]['LongitudeDD']
            # ~ # lat = data[data['Pays'] == row['NAME_FR']]['LatitudeDD']
            # ~ # plt.text(lon, lat, row['NAME_FR'], horizontalalignment='right', transform=ccrs.Geodetic())

        elif row['NAME_FR'].casefold() in (data['Nom autre'].str.casefold()).tolist():
            c = cmapName.to_rgba( math.log( data[data['Nom autre'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
            # ~ # lon = data[data['Nom autre'] == row['NAME_FR']]['LongitudeDD']
            # ~ # lat = data[data['Nom autre'] == row['NAME_FR']]['LatitudeDD']            
            # ~ # plt.text(lon, lat, row['NAME_FR'], horizontalalignment='right', transform=ccrs.Geodetic())

        else:
            ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='r', edgecolor='0')
            print(index, "Le pays ", row['NAME_FR'], "n a pas de correspondant")

cbaxes = fig.add_axes([0.91, 0.062, 0.03, 0.826]) 
fig.colorbar(cmapName, cax=cbaxes, orientation='vertical')
ax.set_title('Population en 2018', fontdict=dict(fontsize=24))
plt.savefig('choropleth-Pop2018.png')
plt.show()


# ~ #***********************************************************************
# ~ #                    ANIMATION CHOROPLETH SUR POP
# ~ #***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd
import numpy as np
import math
import matplotlib.animation as animation

def dms2dd( data ) :
    part = data['Latitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LatitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)
    
    part = data['Longitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LongitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)

data = pd.read_csv("BaseDeDonnee/pays.csv") 
print(data.head())

shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
df = gpd.read_file(shp,encoding='UTF-8')

# Recherche des pays de data qui ne sont pas dans df
for index, row in data.iterrows():
    if row['Pays'].casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    elif str(row['Nom autre']).casefold() in (df['NAME_FR'].str.casefold()).tolist():
        pass
    else:
        print(index, row['Pays'])
        
fig, ax = plt.subplots(figsize=(18, 12), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.set_position([0.05, 0.05, 0.85, 0.85])


# Conversion latitude/longitude DMS en DD
dms2dd(data)

# Creation de la colormap
mn = 1e7
mx = 0
for i in range(59):
    col='Pop'+str(1960+i)
    mnt = math.log(min(data[col]))
    mxt = math.log(max(data[col]))
    mn = mnt if mnt < mn else mn
    mx = mxt if mxt > mx else mx
mn = 300000
mx = 200000000
norm = plt.Normalize(vmin=mn, vmax=mx)
cmapName = plt.cm.ScalarMappable(norm=norm, cmap='Blues')

def init():
    # Boucle sur les pays
    for index, row in df.iterrows():
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='0')
    return ax,

def animate(i):
    col='Pop'+str(1960+5*i)
    print(col)
    ax.set_title('Population en ' + str(1960+5*i), fontdict=dict(fontsize=24))
    for index, row in df.iterrows():
        
        if row['NAME_FR'] != None:
                
            if row['NAME_FR'].casefold() in (data['Pays'].str.casefold()).tolist():
                c = cmapName.to_rgba( ( data[data['Pays'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
                ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')


            elif row['NAME_FR'].casefold() in (data['Nom autre'].str.casefold()).tolist():
                c = cmapName.to_rgba( ( data[data['Nom autre'].str.casefold() == row['NAME_FR'].casefold()][col] ) )
                ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor=list(c), edgecolor='0')
    return ax,

cbaxes = fig.add_axes([0.91, 0.062, 0.03, 0.826]) 
fig.colorbar(cmapName, cax=cbaxes, orientation='vertical')

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=10, blit=True, interval=1000, repeat=False)
plt.show()



#***********************************************************************
#                    ANIMATION CHOROPLETH SUR POP
#***********************************************************************
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd
import numpy as np
import math
import matplotlib.animation as animation

def dms2dd( data ) :
    part = data['Latitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LatitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)
    
    part = data['Longitude'].str.extract('(\d+)°(\d+)\'([^"]+)"([N|S|E|W])', expand=True)
    data['LongitudeDD'] = (part[0].astype(int) + part[1].astype(float)/60  + part[2].astype(float)/3600) * np.where(part[3].isin(['S','W']), -1,1)

data = pd.read_csv("BaseDeDonnee/pays.csv") 
print(data.head())

shp = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_0_countries')
df = gpd.read_file(shp,encoding='UTF-8')

df.boundary.plot()
plt.show()
