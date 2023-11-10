function waterfall(container){
    var self=this;
    this.container=container


	this.num_freqs=1024;
	this.waterfall_buffer_length=1000;
	this.waterfall_buffer_display_length=300;
	this.plot_width=512;
	this.margin=[100,100];
	this.waterfall_plot_height=500;

	this.scroll_data=[];
	this.timearr=[];
	this.ms_per_datum=25.;
	this.freq_list=[];

	this.cb = new imgPlotter();
	this.cb_rect;

	this.cb.min=-10;
	this.cb.max=10;

	this.time=new Date().getTime();

    this.jqcontainer=$("#"+this.container)
    	.css('position','relative')
    	.attr('width',this.plot_width+this.margin[0])
    	.attr('height',this.waterfall_plot_height+this.margin[1])
		.width(this.plot_width+this.margin[0])
    	.height(this.waterfall_plot_height+this.margin[1])

	var waterfall_plot_div=$( "<div/>").uniqueId()
				.css({
       				'position':'relative',
    				'font-size':'8pt',
    				'height':this.waterfall_plot_height,'width':this.margin[0],
    			})
    			.attr('class','axis')
				.appendTo(this.jqcontainer)

	this.yaxis_scale = d3.time.scale.utc()
			    .domain([new Date(this.time),
			    		 new Date(this.time+this.waterfall_buffer_display_length*this.ms_per_datum)])
			    .range([0, this.waterfall_plot_height]);
    this.yaxis = d3.svg.axis().ticks(this.waterfall_plot_height*this.ms_per_datum/1000/2)
				              .scale(this.yaxis_scale).orient("left").tickFormat(d3.time.format('%H:%M:%S'))
	this.yaxisplot=d3.select('#'+waterfall_plot_div[0].id).append("svg")
	    .style("position","absolute")
	    .attr("height", this.waterfall_plot_height)
		.append("g")
	    .attr("transform", "translate(" + this.margin[0] + "," + 0 + ")")
	    .call(this.yaxis)

	this.yaxisplot.append("text")
			.attr("text-anchor","middle")
			.attr("font-size",20)
			.attr("y",-this.margin[0]+35)
			.attr("x",-this.waterfall_plot_height/2)
			.attr("transform", "rotate(-90)")
			.text("Time");

    this.scroll_canvas=$( "<canvas/>")
    						.attr('width', this.num_freqs)
    						.attr('height', this.waterfall_plot_height)
    						.width(this.plot_width)
    						.height(this.waterfall_plot_height)
    						.css({position:'absolute',left:this.margin[0]})
    						.appendTo(waterfall_plot_div)
    this.scrollbuf_canvas=$( "<canvas/>")
    						.attr('width', this.num_freqs)
    						.attr('height', this.waterfall_buffer_display_length)
    						.css('display','none')
    						.appendTo(this.jqcontainer)


    var freq_div = $("<div/>").uniqueId().height(20)
    						.css({
		           				'position':'relative',
		        				'font-size':'8pt',
	    	    				'height':20,'width':this.plot_width,
	    	    				'left':this.margin[0]
	            			})
	            			.attr('class','axis')
							.appendTo(this.jqcontainer)
	this.freq_scale = d3.scale.linear().range([0,this.plot_width]).domain([1,2]);
	this.freq_axis = d3.svg.axis().scale(this.freq_scale).orient("bottom")
	this.freq_axisplot=d3.select('#'+freq_div[0].id).append("svg")
	    .style("position","absolute")
	    .style("left",-10)
	    .attr("width", this.plot_width+20)
		.append("g")
	    .attr("transform", "translate(" + 10 + "," + 0 + ")")
	    .call(this.freq_axis)
	
	this.freq_axisplot.append("text")
			.attr("text-anchor","middle")
			.attr("font-size",20)
			.attr("x",this.plot_width/2)
			.attr("y",this.margin[1]/2)
			.text("Frequency [MHz]");

}

waterfall.prototype.draw =
	function()  {
		var self=this
		var now = new Date().getTime();
		var dt = now - (this.time || now);
		if (dt < 50) return;
		this.time = now;
		if (this.r > 0) {return;}
		this.r=requestAnimationFrame(function(){self.dodraw(); self.r=0;});
	}


waterfall.prototype.dodraw =
	function()  {
		var scd=this.scroll_data;
		var scroll_img=[];

	    this.scroll_canvas.attr('height', this.waterfall_buffer_display_length)
		var img_mean=Array.apply(null, new Array(this.num_freqs)).map(Number.prototype.valueOf,0);
		var disp_start=Math.max(0,scd.length-this.waterfall_buffer_display_length);
		for (i=0; i<this.num_freqs; i++) {
			for (j=0; j<scd.length; j++) {
				img_mean[i]+=scd[j][i];
			}
			img_mean[i]/=scd.length;
		}
	 	var c = this.scroll_canvas[0].getContext("2d");
		c.imageSmoothingEnabled = false;
		imageData = c.createImageData(this.num_freqs,this.waterfall_buffer_display_length)
		for (j=disp_start; j<scd.length; j++){
			scroll_img[j]=[]
			for (i=0; i<this.num_freqs; i++){
					scroll_img[j][i]=10*Math.log10(scd[j][i]);///img_mean[i]);
					this.cb.setPixel(imageData,i,j-disp_start,scroll_img[j][i])
			}
		}
		c.putImageData(imageData, 0, 0);

		if (scd.length > this.waterfall_buffer_display_length)
		{
			this.yaxis_scale.domain([ new Date(this.timearr[this.timearr.length-this.waterfall_buffer_display_length]*1e3),
								new Date(this.timearr[this.timearr.length-1]*1e3) ])
			this.yaxisplot.call(this.yaxis)
		}

	}

waterfall.prototype.openSocket =
	function()
	{
		this.isopen=false;
	    this.socket = new WebSocket("ws://localhost:8539");
	    this.socket.binaryType = "arraybuffer";
	    self=this
	    this.socket.onopen = function() {
	       console.log("Connected!");
	       this.isopen = true;
	    }
	    this.socket.onmessage = function(e) {
	       if (typeof e.data == "string") {
	       	  msg=JSON.parse(e.data)
	          console.log("Text message received: " + e.data)
	          for (var key in msg)
	          {
	          	console.log(key,msg[key])
	          }
     		  self.num_freqs=msg['nfreq'];
		      self.scroll_canvas.attr('width', self.num_freqs)
    		  self.scrollbuf_canvas.attr('width', self.num_freqs)
	       } else {
	       	  var msgtype = new Int8Array(e.data.slice(0,1))[0]

	       	  switch (msgtype) {
	       	  	case 1: //freq list
	       	  	  self.freq_list = new Float32Array(e.data.slice(1))
				  self.freq_scale.domain([self.freq_list[0],self.freq_list[self.num_freqs-1]])
				  self.freq_axisplot.call(self.freq_axis)
	       	  	  break;
	       	  	case 2: //timestep
		       	  var timestamp = new Float64Array(e.data.slice(1,9))[0]
		       	  while (self.timearr.length>self.waterfall_buffer_length) {self.timearr.shift();}
		       	  self.timearr.push(timestamp);
		          var arr = new Float32Array(e.data.slice(9));
				  while (self.scroll_data.length>self.waterfall_buffer_length) {self.scroll_data.shift();}
				  self.scroll_data.push(arr);
				  break;
			  }
	       }
	       self.draw();
	    }
	    this.socket.onerror = function(error) {
			alert(`[error]`);
		};

	    this.socket.onclose = function(e) {
	       console.log("Connection closed.");
	       this.socket = null;
	       this.isopen = false;
	    }
	}

waterfall.prototype.closeSocket =
	function()
	{
		this.socket.close()
	}

waterfall.prototype.addColorSlider = 
	function(target,range)
	{
		var width=$("#"+target).width()
	    var self=this
	    var inrange=[-1000,1000]
	    var scale=(inrange[1]-inrange[0])/(range[1]-range[0])
	    var slider_height=50
		var slider_text=[]
		var marg=15
		var width=$("#"+target).width()

	    wrapper=$("<div style='margin:0px'/>").uniqueId().height(slider_height).appendTo($("#"+target))
	    cbslider=$("<div/>").uniqueId().appendTo(wrapper)
	    	.css({left:marg, width:width-2*marg})

		cbslider.slider({min:inrange[0],max:inrange[1],range:true,
						values:[(this.cb.min-range[0])*scale+inrange[0],
								(this.cb.max-range[0])*scale+inrange[0]],
						slide:function(event, ui){
							self.cb.min=(ui.values[0]-inrange[0])/scale+range[0];
							self.cb.max=(ui.values[1]-inrange[0])/scale+range[0];
							slider_text[0].attr({"text":self.cb.min.toFixed(2)});
							slider_text[0].attr({"x":(ui.values[0]-inrange[0])/(inrange[1]-inrange[0])*(width-2*marg)+marg});
							slider_text[1].attr({"text":self.cb.max.toFixed(2)});
							slider_text[1].attr({"x":(ui.values[1]-inrange[0])/(inrange[1]-inrange[0])*(width-2*marg)+marg});	

							for (i=0; i<self.cb_tags.length; i++){
								self.cb_tags[i].attr({"text":(i/(self.cb_tags.length-1)*(self.cb.max-self.cb.min)+self.cb.min).toFixed(2)});
							}
							self.draw();
						}})
//		cbslider.width(width);


		var rr=Raphael($("<div style='position:relative'/>").uniqueId().appendTo(wrapper)[0].id,width, 50);
		rr.canvas.style.position="absolute";
		rr.canvas.style.zIndex="100";
		rr.setStart();
		slider_text[0]=rr.text((this.cb.min-range[0])/(range[1]-range[0])*(width-2*marg)+marg,13,self.cb.min.toFixed(2));
		slider_text[0].attr({'font-size': 12});
		slider_text[1]=rr.text((this.cb.max-range[0])/(range[1]-range[0])*(width-2*marg)+marg,13,self.cb.max.toFixed(2));
		slider_text[1].attr({'font-size': 12});
		rr.text(width/2,30,"Color Bar Range [dB]").attr({'font-size':14});
		rr.setFinish()
	}

waterfall.prototype.addColorBar = 
	function(target)
	{
		var cb_height=50
		var marg=15
		var width=$("#"+target).width()

	    wrapper=$("<div style='margin:0px'/>").uniqueId().height(cb_height).appendTo($("#"+target))
	    cb=$( "<div/>").uniqueId().appendTo(wrapper)

		var rr = Raphael(cb[0].id, width,cb_height);
		rr.setStart()
		this.cb_rect=rr.rect(marg,0,width-2*marg,30).attr({fill:this.cb.cb_grad});

		this.cb_tags=[]
		var ntags=5
		for (i=0; i<ntags; i++)
		{
			this.cb_tags[i]=rr.text(i/(ntags-1)*(width-2*marg),40,(i/(ntags-1)*(this.cb.max-this.cb.min)+this.cb.min).toFixed(2))
							.attr({'text-anchor':'start'});	
		}

		rr.setFinish();
	}

waterfall.prototype.addColorSelect =
	function(target,cb2use)
	{
		var self=this
		var marg=15
		var width=$("#"+target).width()

		cp=$("<select/>").appendTo($("<div/>").appendTo("#"+target).css({margin:marg}))
		for (var newcm of cb2use){
			if (newcm in this.cb.colormaps){
			    cp.append("<option>"+newcm+"</option>")}
			else {console.log(newcm+" not a known colormap!")}
		}
		cp.selectmenu({
			change: function(event, data){self.change_palette(event, data);},
			width: width-2*marg}
		)
		.data("ui-selectmenu")
		._renderItem=function(ul, item) {
			self.li = $( "<li>", {text: item.label});
			var im = $( "<span/>")
						.appendTo(self.li)
						.css('float','right');

			var rr = Raphael(im[0],"100%",Math.ceil(this.button[0].clientHeight/2));
			rr.setStart();
			rr.rect(0,0,"100%","100%").attr({fill:self.cb.gradString(self.cb.colormaps[item.label])});
			rr.setFinish();

			return self.li.appendTo(ul);
		};

	}

waterfall.prototype.start = function() {
	self.openSocket();
}
waterfall.prototype.stop = function() {
	self.closeSocket();
}

waterfall.prototype.addStartStop =
	function(target)
	{
		self=this
	    wrapper=$("<div/>").uniqueId().appendTo($("#"+target)).css({margin:45})
		self.startstop_btn = $("<button/>").appendTo($("<div/>").appendTo(wrapper))
				.button({label:'Start',icons:{primary: "ui-icon-play"}})
				.css({margin:"0 auto",display:"block"})
				.click(function() {
				 	if ( $( this ).text() === "Stop" ) {
						$( this ).button( "option", {label: "Resume", icons: {primary: "ui-icon-play"}})
						self.stop();
				    } else {
						$( this ).button( "option", {label: "Stop", icons: {primary: "ui-icon-stop"}})
						self.start();
					}
				});
	}

waterfall.prototype.addRecordButton = 
	function(target)
	{
		self = this
		this.recording = false
		this.fn_idx = 0
		var marg=15
		var width=$("#"+target).width()
	    wrapper=$("<div style='margin:10px,width:100%'/>").uniqueId()
	    			.height(30).width(width-2*marg).appendTo($("#"+target))

		this.record_fn=$("<input type='text'/>")
				.css({'width':'50%','float':'left', 'font-size':'16pt', 'margin-top':5})
				.val("output"+("000" + this.fn_idx).slice(-4)+".dat")
				.appendTo(wrapper)

		this.record_btn = $("<button/>").appendTo($( "<div style='margin:10px'/>").appendTo(wrapper))
				.css({'float':'right'})
				.button({label:'Record', icons: {primary: "ui-icon-disk"}})
				.click(function() {
					if (!self.recording) {
						self.socket.send(JSON.stringify({'type': 'record', 'state': true, 'file': self.record_fn.val()}))
						$ (this ).button( "option", {label: "Recording", icons: {primary: "ui-icon-bullet"}})
						self.recording = true
					}
					else {
						self.socket.send(JSON.stringify({'type': 'record', 'state': false, 'file': "null"}))
						$ (this ).button( "option", {label: "Record", icons: {primary: "ui-icon-blank"}})
						self.recording = false
						self.fn_idx = self.fn_idx+1;
						self.record_fn.val("output"+("000" + self.fn_idx).slice(-4)+".dat")
					}
				});
	}

waterfall.prototype.addAirspyGainControl =
	function(target)
	{
		self=this
		this.adc={'mean':0, 'rms':0, 'railfrac':0};
		this.kotekan_url = "localhost"
		this.kotekan_port= 12048

		change_gain = function(type,value,o){
			fetch('http://'+o.kotekan_url+':'+o.kotekan_port+'/airspy_input/config', {
				mode: 'no-cors',
			    method: 'POST',
			    headers: {
			        'Accept': 'application/json',
			        'Content-Type': 'application/json'
			    },
			    body: JSON.stringify({[type]: value})
			})
		   .then(r =>
			   	fetch('http://'+o.kotekan_url+':'+o.kotekan_port+'/airspy_input/adcstat',{})
			   		 .then(r => r.json().then(data => console.log(data)))

		   )
		}

		var marg=15
		var width=$("#"+target).width()
	    var slider_width=50
	    var slider_height=200
	    wrapper=$("<div style='margin:10px,width:100%'/>").uniqueId()
	    			.height(slider_height).width(width-2*marg).appendTo($("#"+target))

		$("<p/>").css({'font-family':'sans-serif','text-align':'center','margin':marg})
		    		.text("Gain").appendTo(wrapper)		    		

	    {
		    lnawrap = $("<div style='float:left'/>").width(slider_width)
		    			.css({'font-family':'sans-serif','text-align':'center','margin':2}).appendTo(wrapper)
		    $("<p/>").css({'font-family':'sans-serif', 'margin':2})
		    		.text("LNA").appendTo(lnawrap)

		    gain_lna=$("<div/>").uniqueId().appendTo(lnawrap).css({'margin':'auto'})
						.slider({min:0,max:14,value:10,step:1,
							orientation: "vertical",
							slide:function(event, ui){
								lna_gain=ui.value;
								change_gain("gain_lna",lna_gain,self)
								gain_lnat.text(ui.value);
							}
						})
		    var gain_lnat=$("<p/>").css({'font-family':'sans-serif','text-align':'center','margin':2})
		    		.text(10).appendTo(lnawrap)
    	}
	    {
		    mixwrap = $("<div style='float:left'/>").width(slider_width)
		    			.css({'font-family':'sans-serif','text-align':'center','margin':2}).appendTo(wrapper)
		    $("<p/>").css({'font-family':'sans-serif','margin':2})
		    		.text("MIX").appendTo(mixwrap)

		    gain_mix=$("<div/>").uniqueId().appendTo(mixwrap).css({'margin':'auto'})
						.slider({min:0,max:15,value:10,step:1,
							orientation: "vertical",
							slide:function(event, ui){
								mix_gain=ui.value;
								change_gain("gain_mix",mix_gain,self)
								gain_mixt.text(ui.value);
							}
						})
		    var gain_mixt=$("<p/>").css({'font-family':'sans-serif', 'margin':2})
		    		.text("10").appendTo(mixwrap)
    	}
	    {
		    ifwrap = $("<div style='float:left'/>").width(slider_width)
		    			.css({'font-family':'sans-serif','text-align':'center','margin':2}).appendTo(wrapper)
		    $("<p/>").css({'font-family':'sans-serif', 'margin':2})
		    		.text("IF").appendTo(ifwrap)

		    gain_if=$("<div/>").uniqueId().appendTo(ifwrap).css({'margin':'auto'})
						.slider({min:0,max:15,value:10,step:1,
							orientation: "vertical",
							slide:function(event, ui){
								if_gain=ui.value;
								change_gain("gain_if",if_gain,self)
								gain_ift.text(ui.value);
							}
						})
		    var gain_ift=$("<p/>").css({'font-family':'sans-serif', 'margin':2})
		    		.text("10").appendTo(ifwrap)
    	}
	}


waterfall.prototype.addWaterfallControl =
	function(target)
	{
		self=this

		var width=$("#"+target).width()
		var marg=15
	    var self=this
	    var slider_height=50
	    var wfslider

	    wrapper=$("<div style='margin:10px'/>").uniqueId().height(slider_height).width(width-2*marg).appendTo($("#"+target))
//	    	.css({'border':'1px solid black'})
		var bins_text=$("<input type='number'/>")
				.attr({min:200,max:this.waterfall_buffer_length})
				.css({'width':'25%','float':'right', 'font-size':'16pt', 'margin-top':5})
				.val(this.waterfall_buffer_display_length)
				.appendTo(wrapper)
				.change(
					function(){
						if (parseInt(this.value) < ($(this).attr("min"))) {this.value=$(this).attr("min")}
						if (parseInt(this.value) > ($(this).attr("max"))) {this.value=$(this).attr("max")}
						wfslider.slider('value',this.value)
						self.waterfall_buffer_display_length=parseInt(this.value);
						self.draw()
					}
				)
		bins_text.numeric()

	    var bintext=$("<p/>").css({'font-family':'sans-serif', 'margin':2})
	    		.text("Time Samples in Waterfall:").appendTo(wrapper)

	    wfslider=$("<div style='width:70%'/>").uniqueId().appendTo(wrapper)
					.slider({min:200,max:this.waterfall_buffer_length,value:this.waterfall_buffer_display_length,
						slide:function(event, ui){
							self.waterfall_buffer_display_length=ui.value;
							self.draw()
							bins_text.val(ui.value);
						}})




	}

waterfall.prototype.addBufferControl =
	function(target)
	{
		self=this

		var width=$("#"+target).width()
		var marg=15
	    var self=this
	    var slider_height=50
	    var bufslider

	    wrapper=$("<div style='margin:10px'/>").uniqueId().height(slider_height).width(width-2*marg).appendTo($("#"+target))
//	    	.css({'border':'1px solid black'})
		var bins_text=$("<input type='number'/>")
				.attr({min:200,max:2000})
				.css({'width':'25%','float':'right', 'font-size':'16pt', 'margin-top':5})
				.val(this.waterfall_buffer_length)
				.appendTo(wrapper)
				.change(
					function(){
						if (parseInt(this.value) < ($(this).attr("min"))) {this.value=$(this).attr("min")}
						if (parseInt(this.value) > ($(this).attr("max"))) {this.value=$(this).attr("max")}
						bufslider.slider('value',this.value)
						self.waterfall_buffer_length=parseInt(this.value);
					}
				)
		bins_text.numeric()

	    var bintext=$("<p/>").css({'font-family':'sans-serif', 'margin':2})
	    		.text("Timesamples to Buffer:").appendTo(wrapper)

	    bufslider=$("<div style='width:70%'/>").uniqueId().appendTo(wrapper)
					.slider({min:200,max:2000,value:this.waterfall_buffer_length,
						slide:function(event, ui){
							self.waterfall_buffer_length=ui.value;
							self.draw()
							bins_text.val(ui.value);
						}})
	}


waterfall.prototype.change_palette=
	function(event, data)
	{
		this.cb.gradientScale(this.cb.colormaps[data.item.label]);
		this.cb_rect.attr({fill:this.cb.cb_grad})
		this.draw();
	}
