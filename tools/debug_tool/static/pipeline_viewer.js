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
        var name = d3.select(label._groups[0][i]).select("tspan").text();
        d3.select(label._groups[0][i]).select("#utl")
            .text(new_buffers[name].num_full_frame + "/" + new_buffers[name].num_frames);
    }
}

// To avoid long string, break the label by '/'
var insertLinebreaks = function (d, margin = 6) {
    var el = d3.select(this);
    var words = d.name.split('/');
    var tspan_x = margin/2

    // The first line will be used to check if it is a buffer
    var tspan = el.append('tspan').text(words[0])
                    .attr('x', tspan_x).attr('dy', '15')
                    .attr("font-size", "15");

    for (var i = 1; i < words.length; i++) {
        tspan = el.append('tspan').text('/' + words[i]);
        tspan.attr('x', tspan_x).attr('dy', '15')
                .attr("font-size", "15");
    }
};

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
    #node;
    #label;

    constructor(buffers, body) {
        this.buffers = buffers;
        this.bufNames = Object.keys(this.buffers);
        this.body = body;
    }

    start_viewer() {
        this.init_svg();
        this.parse_data();
        this.create_objs();
        this.start_buff_ult();
    }

    init_svg(width = 960, height = 500) {;
        this.#d3cola = cola.d3adaptor(d3)
            .linkDistance(80)
            .size([width, height]);

        // Add a svg section and employ zooming
        this.#svg = this.body.append("svg")
            .attr("width", width)
            .attr("height", height)
            .call(d3.zoom().on("zoom", function () {
                svg.attr("transform", d3.event.transform)
            }));
    }

    parse_data() {
        var nodes = [], links = [];
        this.#graph = {};

        // For every buffer, add its producers and comsumers to nodes and create links
        for (var val of this.bufNames) {
            var obj = this.buffers[val];
            nodes.push({name: val});

            for (var stage in obj.producers) {
                nodes.push({name: stage});
                links.push({source: nodes.find(obj => obj.name === stage), target: nodes.find(obj => obj.name === val)});
            }
            if (obj.consumers !== "None") {
                for (var stage in obj.consumers) {
                    nodes.push({name: stage});
                    links.push({source: nodes.find(d => d.name === val), target: nodes.find(d => d.name === stage)});
                }
            }
        }

        // Remove all duplicated nodes
        this.#graph.nodes = Array.from(new Set(nodes.map(a => a.name)))
            .map(name => {
                return nodes.find(a => a.name === name)
            });
        this.#graph.links = links;
    }

    create_objs(margin = 6) {
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
        var node_width = width + 2 * pad + 2 * margin;
        var node_height = height + 2 * pad + 2 * margin;
        this.#node = this.#svg.selectAll(".node")
            .data(this.#graph.nodes)
            .enter().append("rect")
            .attr("class", "node")
            .attr("width", node_width)
            .attr("height", node_height)
            .attr("rx", 5).attr("ry", 5)
            .call(this.#d3cola.drag);

        // Create label objects (stage and buffer names) in svg section
        this.#label = this.#svg.selectAll(".label")
            .data(this.#graph.nodes)
            .enter().append("text")
            .attr("class", "label")
            .call(this.#d3cola.drag);
        this.#label.each(insertLinebreaks);

        // Add names to identify different nodes
        this.#node.append("title")
            .text(function (d) { return d.name; });

        // Calculate node, link, and label positions
        this.#d3cola.on("tick", function () {
            self.#node.each(function (d) {
                d.innerBounds = d.bounds.inflate(- margin);
            });

            self.#link.each(function (d) {
                d.route = cola.makeEdgeBetween(d.source.innerBounds, d.target.innerBounds, 5);
                if (isIE())  this.parentNode.insertBefore(this, this);
            });

            self.#link.attr("x1", function (d) { return d.route.sourceIntersection.x; })
                .attr("y1", function (d) { return d.route.sourceIntersection.y; })
                .attr("x2", function (d) { return d.route.arrowStart.x; })
                .attr("y2", function (d) { return d.route.arrowStart.y; });

            self.#label.each(function (d) {
                var b = this.getBBox();
                d.width = b.width + 2 * margin + 8;
                d.height = b.height + 2 * margin + 8;
            });

            self.#node.attr("x", function (d) { return d.innerBounds.x; })
                .attr("y", function (d) { return d.innerBounds.y; })
                .attr("width", function (d) { return Math.abs(d.innerBounds.width()); })
                .attr("height", function (d) { return Math.abs(d.innerBounds.height()); });

            self.#label.attr("transform", function (d) {
                return "translate(" + d.innerBounds.x + margin + "," + (d.innerBounds.y + margin/2) + ")";
            });
        });
    }

    start_buff_ult(margin = 6) {
        // Add utilization for all buffers and record their index
        // Index is used later to dynamically update buffer utilization
        var index = [];
        this.#label._groups[0].reduce((pre, cur, ind) => {
            if (this.bufNames.includes(cur.textContent)) {
                var el = d3.select(cur);
                var tspan = el.append('tspan')
                            .text(this.buffers[cur.textContent].num_full_frame + "/" + this.buffers[cur.textContent].num_frames)
                tspan.attr('x', margin/2).attr('dy', '15')
                        .attr("font-size", "15")
                        .attr("id", "utl");
                index.push(ind)
            }
        }, 0)

        // Start polling kotekan for metrics
        var labels = this.#label;
        poll(labels, index);
    }

}


