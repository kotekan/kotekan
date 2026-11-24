function imgPlotter() {
	this.min=0;
	this.max=1;
	this.palette=[];

	this.colormaps={
                    'viridis':["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"],
                    'inferno':["#fcffa4", "#f98e09", "#bc3754", "#57106e", "#000004"],
                    'magma':["#fcfdbf", "#fc8961", "#b73779", "#51127c", "#000004"],
                    'plasma':["#f0f921", "#f89540", "#cc4778", "#7e03a8", "#0d0887"],
					'jet':["#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"],
					'heat':['black','red','yellow','white'],
					'red':['red','white'],
					'blue':['blue','white'],
                    'gray':['black','white'],
				   }
	this.gradientScale(this.colormaps['viridis']);
}

imgPlotter.prototype.knownColormaps =
	function() {return Object.keys(this.colormaps);}

imgPlotter.prototype.initPalette =
	function () {
		for (i=0; i<1000; i++){
			this.palette[i]=this.colormap(i,0,1000).toRgb();
		}
	}

// Neutral "no data" colour painted into the waterfall wherever frames are
// missing (kotekan upstream drop, WS reconnect, etc). A mid grey reads as
// "blank" against any of the standard colormaps.
imgPlotter.GAP_R = 0x40;
imgPlotter.GAP_G = 0x40;
imgPlotter.GAP_B = 0x40;

imgPlotter.prototype.setPixel =
	function(imageData, x, y, d) {
	    var index = (x + y * imageData.width) * 4;
	    if (d !== d) {  // NaN sentinel for missing-data rows
	        imageData.data[index+0] = imgPlotter.GAP_R;
	        imageData.data[index+1] = imgPlotter.GAP_G;
	        imageData.data[index+2] = imgPlotter.GAP_B;
	        imageData.data[index+3] = 255;
	        return;
	    }
	    var c=this.color(d)
	    imageData.data[index+0] = c.r;
	    imageData.data[index+1] = c.g;
	    imageData.data[index+2] = c.b;
	    imageData.data[index+3] = 255;
	}

imgPlotter.prototype.gradientScale =
	function(colors){
		this.colormap=function(n,min,max){
			var f = (n-min)/(max-min);
			var l = colors.length-1;
			var ci=f*l;
			return tinycolor.mix(tinycolor(colors[Math.floor(ci)]),
								 tinycolor(colors[Math.floor(ci)+1]),
								 (ci-Math.floor(ci))*100 
								 );
		}
		this.initPalette();

		this.cb_grad="0-"+this.colormap(0,0,1)+"-";
		for (i=1; i<colors.length-1; i++) {
			this.cb_grad+=this.colormap(i,0,(colors.length-1)).toRgbString()
							+":"
							+Math.floor(i*100/(colors.length-1)).toString()
							+"-";
		}
		this.cb_grad+=this.colormap(1,0,1).toRgbString();
		return this
	}

// CSS ``linear-gradient(...)`` string for a colormap -- used by the
// viewer (app/) for the colorbar fill and select-menu previews.
imgPlotter.prototype.cssGradString =
	function(colors){
		var stops=[];
		for (var i=0; i<colors.length; i++) {
			var pct=Math.floor(i*100/(colors.length-1));
			stops.push(tinycolor(colors[i]).toRgbString()+" "+pct+"%");
		}
		return "linear-gradient(to right, "+stops.join(", ")+")";
	}

imgPlotter.prototype.color = 
	function(n){
		if (n<this.min) {return this.palette[0];}
		if (n>this.max) {return this.palette[this.palette.length-1];}
		if (isNaN(n)) {return {r:0, g:0, b:0};}
		return this.palette[Math.floor((n-this.min)/(this.max-this.min)*this.palette.length)];
	}
