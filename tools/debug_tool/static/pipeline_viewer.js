function isIE() { return ((navigator.appName == 'Microsoft Internet Explorer') || ((navigator.appName == 'Netscape') && (new RegExp("Trident/.*rv:([0-9]{1,}[\.0-9]{0,})").exec(navigator.userAgent) != null))); }

// Time between updating kotekan metrics
const POLL_WAIT_TIME_MS = 1000;
// Poll kotekan via web server every POLL_WAIT_TIME_MS and update page with new metrics
function poll(label, index) {
    get_data().then(function (new_buffers) {
        update_buf_utl(new_buffers, label, index);
    });

    // Call poll() again to get the next message
    setTimeout(() => { poll(label, index); }, POLL_WAIT_TIME_MS);
}

// Update buffer utilization
function update_buf_utl(new_buffers, label, index){
    for (var i of index) {
        var name = d3.select(label[0][i]).select("tspan").text();
        d3.select(label[0][i]).select("#utl")
            .text(new_buffers[name].num_full_frame + "/" + new_buffers[name].num_frames);
    }
}

// Read from endpoint /buffers to get buffer stats
async function get_data() {
    let response = await fetch("/buffers");

    if (response.status == 502) {
        // Status 502 is a connection timeout error,
        // may happen when the connection was pending for too long,
        // and the remote server or a proxy closed it
        // let's reconnect
    } else if (response.status != 200) {
        // An error - let's show it
        console.log("Error: " + response.statusText);
        // Reconnect in one second
        await new Promise(resolve => setTimeout(resolve, 1000));
    } else {
        // Get buffer stats
        return await response.json();
    }
}

class PipelineViewer {
    #d3cola;
    #svg;
    #graph;
    #link;
    #buffers;
    #stages;
    #buffer_labels;
    #stage_labels;

    constructor(buffers, body) {
        this.buffers = buffers;
        this.bufNames = Object.keys(this.buffers);
        this.body = body;
    }

    start_viewer(width = 960, height = 500, margin = 6) {
        this.width = width;
        this.height = height;
        this.margin = margin;
        this.init_svg();
        this.parse_data();
        this.create_objs();
        this.start_buff_ult();
    }

    init_svg() {
        this.#d3cola = cola.d3adaptor(d3)
            .linkDistance(80)
            .size([this.width, this.height]);

        // Add a svg section and employ zooming
        var self = this;
        var outer = d3.select("body").append("svg")
            .attr("width", this.width)
            .attr("height", this.height);

        outer.append('rect')
            .attr('class', 'background')
            .attr('width', "100%")
            .attr('height', "100%")
            .call(d3.behavior.zoom().on("zoom", () => {
                self.#svg.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")");
            }));

        var vis = outer
            .append('g')
            .attr('transform', 'translate(80,80) scale(0.7)');

        this.#svg = vis.append("g");
    }

    parse_data() {
        var buffers = [], stages = [], links = [];
        this.#graph = {};

        // For every buffer, add its producers and comsumers to nodes and create links
        for (var val of this.bufNames) {
            var obj = this.buffers[val];
            buffers.push({name: val});

            for (var stage in obj.producers) {
                stages.push({name: stage});
                links.push({source: stages.find(obj => obj.name === stage), target: buffers.find(obj => obj.name === val)});
            }
            if (obj.consumers !== "None") {
                for (var stage in obj.consumers) {
                    stages.push({name: stage});
                    links.push({source: buffers.find(d => d.name === val), target: stages.find(d => d.name === stage)});
                }
            }
        }

        // Remove all duplicated nodes
        this.#graph.buffers = Array.from(new Set(buffers.map(a => a.name)))
            .map(name => {
                return buffers.find(a => a.name === name)
            });
        this.#graph.stages = Array.from(new Set(stages.map(a => a.name)))
            .map(name => {
                return stages.find(a => a.name === name)
            });
        this.#graph.links = links;
        this.#graph.nodes = this.#graph.buffers.concat(this.#graph.stages);
    }

    create_objs() {
        var self = this;
        // Set cola parameters and enable non-overlapping
        this.#d3cola.nodes(this.#graph.nodes)
            .links(this.#graph.links)
            .flowLayout("y", 60)
            .defaultNodeSize(100)
            .handleDisconnected(true)
            .avoidOverlaps(true)
            .start(30,20,20);

        // define arrow markers for graph links
        this.#svg.append('svg:defs').append('svg:marker')
            .attr('id', 'end-arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 6)
            .attr('markerWidth', 3)
            .attr('markerHeight', 3)
            .attr('orient', 'auto')
            .append('svg:path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#000');

        // Create link objects in svg section
        this.#link = this.#svg.selectAll(".link")
            .data(this.#graph.links)
            .enter().append('line')
            .attr('class', 'link');

        // Create node objects (bounded boxes) in svg section
        var pad = 12, width = 60, height = 40;
        var node_width = width + 2 * pad + 2 * this.margin;
        var node_height = height + 2 * pad + 2 * this.margin;
        this.#buffers = this.#svg.selectAll(".buffers")
            .data(this.#graph.buffers)
            .enter().append("rect")
            .attr("class", "buffers")
            .attr("width", node_width)
            .attr("height", node_height)
            .attr("rx", 25).attr("ry", 25)
            .call(self.#d3cola.drag);

        this.#stages = this.#svg.selectAll(".stages")
            .data(this.#graph.stages)
            .enter().append("rect")
            .attr("class", "stages")
            .attr("width", node_width)
            .attr("height", node_height)
            .attr("rx", 5).attr("ry", 5)
            .call(self.#d3cola.drag);

        // Create label objects (stage and buffer names) in svg section
        this.#buffer_labels = this.#svg.selectAll(".buffer_labels")
            .data(this.#graph.buffers)
            .enter().append("text")
            .attr("class", "label")
            .call(self.#d3cola.drag);

        this.#stage_labels = this.#svg.selectAll(".stage_labels")
            .data(this.#graph.stages)
            .enter().append("text")
            .attr("class", "label")
            .call(self.#d3cola.drag);

        var insertLinebreaks = function (d) {
            var el = d3.select(this);
            var words = d.name.split('/');
            var tspan_x = self.margin/2

            // The first line will be used to check if it is a buffer
            var tspan = el.append('tspan').text(words[0])
                            .attr('x', tspan_x).attr('dy', '15')
                            .attr("font-size", "15");

            for (var i = 1; i < words.length; i++) {
                tspan = el.append('tspan').text('/' + words[i]);
                tspan.attr('x', tspan_x).attr('dy', '15')
                        .attr("font-size", "15");
            }
        }

        this.#buffer_labels.each(insertLinebreaks);
        this.#stage_labels.each(insertLinebreaks);

        // Add names to identify different nodes
        this.#buffers.append("title")
            .text(function (d) { return d.name; });
        this.#stages.append("title")
            .text(function (d) { return d.name; });

        // Calculate node, link, and label positions
        this.#d3cola.on("tick", function () {
            self.#buffers.each(function (d) {
                d.innerBounds = d.bounds.inflate(- self.margin);
            });

            self.#stages.each(function (d) {
                d.innerBounds = d.bounds.inflate(- self.margin);
            });

            self.#link.each(function (d) {
                d.route = cola.makeEdgeBetween(d.source.innerBounds, d.target.innerBounds, 5);
                if (isIE())  this.parentNode.insertBefore(this, this);
            });

            self.#link.attr("x1", function (d) { return d.route.sourceIntersection.x; })
                .attr("y1", function (d) { return d.route.sourceIntersection.y; })
                .attr("x2", function (d) { return d.route.arrowStart.x; })
                .attr("y2", function (d) { return d.route.arrowStart.y; });

            self.#buffer_labels.each(function (d) {
                var b = this.getBBox();
                d.width = b.width + 2 * self.margin + 8;
                d.height = b.height + 2 * self.margin + 8;
            });
            self.#stage_labels.each(function (d) {
                var b = this.getBBox();
                d.width = b.width + 2 * self.margin + 8;
                d.height = b.height + 2 * self.margin + 8;
            });

            self.#buffers.attr("x", function (d) { return d.innerBounds.x; })
                .attr("y", function (d) { return d.innerBounds.y; })
                .attr("width", function (d) { return Math.abs(d.innerBounds.width()); })
                .attr("height", function (d) { return Math.abs(d.innerBounds.height()); });

            self.#stages.attr("x", function (d) { return d.innerBounds.x; })
                .attr("y", function (d) { return d.innerBounds.y; })
                .attr("width", function (d) { return Math.abs(d.innerBounds.width()); })
                .attr("height", function (d) { return Math.abs(d.innerBounds.height()); });

            self.#buffer_labels.attr("transform", function (d) {
                return "translate(" + d.innerBounds.x + self.margin + "," + (d.innerBounds.y + self.margin/2) + ")";
            });
            self.#stage_labels.attr("transform", function (d) {
                return "translate(" + d.innerBounds.x + self.margin + "," + (d.innerBounds.y + self.margin/2) + ")";
            });
        });
    }

    start_buff_ult() {
        // Add utilization for all buffers and record their index
        // Index is used later to dynamically update buffer utilization
        var index = [];
        this.#buffer_labels[0].reduce((pre, cur, ind) => {
            if (this.bufNames.includes(cur.textContent)) {
                var el = d3.select(cur);
                var tspan = el.append('tspan')
                            .text(this.buffers[cur.textContent].num_full_frame + "/" + this.buffers[cur.textContent].num_frames)
                tspan.attr('x', this.margin/2).attr('dy', '15')
                        .attr("font-size", "15")
                        .attr("id", "utl");
                index.push(ind)
            }
        }, 0)

        // Start polling kotekan for metrics
        var labels = this.#buffer_labels;
        poll(labels, index);
    }

}


