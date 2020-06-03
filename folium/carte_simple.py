import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import os



tiles = ['cartodbdark_matter'
            ,'cartodbpositron'
            ,'cartodbpositronnolabels'
            ,'cartodbpositrononlylabels'
            ,'mapboxbright'
            ,'mapboxcontrolroom'
            ,'openstreetmap'
            ,'stamenterrain'
            ,'stamentoner'
            ,'stamentonerbackground'
            ,'stamentonerlabels'
            ,'stamenwatercolor']
            
for t in tiles:
    print(t)
    # Create a map
    m_1 = folium.Map(location=[42.32,-71.0589], tiles=t, zoom_start=10)

    # Display the map
    m_1.save(t+'.html')    
    os.system("chromium-browser " + t + ".html")

# TILES WITH API
# ~ tiles = ['Cloudmade','Mapbox']
            
# ~ for t in tiles:
    # ~ print(t)
    # ~ # Create a map
    # ~ m_1 = folium.Map(location=[42.32,-71.0589], tiles=t, API_key='YourKey', zoom_start=10)

    # ~ # Display the map
    # ~ m_1.save(t+'.html')


