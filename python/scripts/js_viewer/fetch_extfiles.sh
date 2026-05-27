#!/bin/bash
#
# Fetch the third-party JS/CSS used by index-local.html into ./extlibs/,
# plus a few image assets into ./imgs/. The default index.html pulls these
# straight from CDNs and does not need this script.
#
# Usage:  ./fetch_extfiles.sh

set -e

mkdir -p extlibs imgs

wget -P extlibs http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js
wget -P extlibs http://ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/themes/smoothness/jquery-ui.css
wget -P extlibs http://ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js
wget -P extlibs http://d3js.org/d3.v3.min.js
wget -P extlibs https://cdn.jsdelivr.net/npm/jquery-numeric@1.5.0/jquery.numeric.min.js
wget -P extlibs https://cdn.jsdelivr.net/npm/tinycolor2@1.6.0/cjs/tinycolor.min.js
wget -P extlibs https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.1/plotly-basic.min.js
wget -P extlibs https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.1/plotly-cartesian.min.js
wget -P extlibs https://cdnjs.cloudflare.com/ajax/libs/underscore.js/1.13.6/underscore-min.js
wget -P extlibs https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack.min.css
wget -P extlibs https://cdn.jsdelivr.net/npm/gridstack@10.3.1/dist/gridstack-all.js

wget -P imgs https://cdn.eso.org/images/screen/eso1339e.jpg
