function isIE() { return ((navigator.appName == 'Microsoft Internet Explorer') || ((navigator.appName == 'Netscape') && (new RegExp("Trident/.*rv:([0-9]{1,}[\.0-9]{0,})").exec(navigator.userAgent) != null))); }

// Time between updating kotekan metrics
const POLL_WAIT_TIME_MS = 1000;
// Poll kotekan via web server every POLL_WAIT_TIME_MS and update page with new metrics
function poll(buffer_labels, stage_labels) {
    get_data("/buffers").then(function (new_buffers) {
        update_buf_utl(new_buffers, buffer_labels);
    });

    // get_data("/cpu_ult").then(function (cpu_stats) {
    //     update_cpu_utl(cpu_stats, stage_labels);
    // })

    get_data("/trackers_current").then(function (trackers) {
        show_trackers_in_label(trackers, stage_labels);
        update_trackers(trackers);
    })

    // Call poll() again to get the next message
    setTimeout(() => { poll(buffer_labels, stage_labels); }, POLL_WAIT_TIME_MS);
}

// Update buffer utilization
function update_buf_utl(new_buffers, label){
    label[0].reduce((pre, cur) => {
        var el = d3.select(cur);
        var name = el.select("tspan").text();
        el.select("#utl")
            .text(new_buffers[name].num_full_frame + "/" + new_buffers[name].num_frames);
    })
}

// Update stage utilization
function update_cpu_utl(cpu_stats, label){
    if ("error" in cpu_stats) {
        return;
    }

    label[0].reduce((pre, cur) => {
        var el = d3.select(cur);
        var name = el.select("tspan").text();
        var tot_ult = 0, usr_ult = 0, sys_ult = 0;

        d3.select(cpu_stats[name]).forEach((stage) => {
            var threads = Object.keys(stage[0]);
            for (tid of threads) {
                var usr = stage[0][tid]["usr_cpu_ult"];
                var sys = stage[0][tid]["sys_cpu_ult"];
                usr_ult += usr;
                sys_ult += sys;
                tot_ult += usr + sys;
            }
        });
        // Limit to two decimals
        usr_ult = usr_ult.toFixed(2);
        sys_ult = sys_ult.toFixed(2);
        tot_ult = tot_ult.toFixed(2);

        el.select("title").text("usr: " + usr_ult + "%; sys: " + sys_ult + "%");
        el.select("#utl").text("CPU: " + tot_ult + "%");
    }, 0)
}

function update_trackers(trackers){
    var stage_names = Object.keys(trackers);
    for (stage of stage_names) {
        d3.select(trackers[stage]).forEach((stage_obj) => {

                var tracker_name = Object.keys(stage_obj[0]);
                for (tracker of tracker_name) {
                    // Limit to two-decimal scientific expression.
                    var avg = (stage_obj[0][tracker]["avg"]).toExponential(2);
                    var max = (stage_obj[0][tracker]["max"]).toExponential(2);
                    var min = (stage_obj[0][tracker]["min"]).toExponential(2);
                    var std = (stage_obj[0][tracker]["std"]).toExponential(2);
                    var cur = (stage_obj[0][tracker]["cur"]["value"]).toExponential(2);
                    var unit = stage_obj[0][tracker]["unit"];

                    // Skip if stage is not selected.
                    var stage_btn = document.getElementById(stage + "_button");
                    if (stage_btn) {
                        var el = document.getElementById(stage + "/" + tracker);
                        // If the tracker info exists, only update it.
                        if (el) {
                            document.getElementById(stage + "/" + tracker + "_cur").innerHTML = cur;
                            document.getElementById(stage + "/" + tracker + "_min").innerHTML = min;
                            document.getElementById(stage + "/" + tracker + "_max").innerHTML = max;
                            document.getElementById(stage + "/" + tracker + "_avg").innerHTML = avg;
                            document.getElementById(stage + "/" + tracker + "_std").innerHTML = std;
                        } else {
                            var stage_div = d3.select(document.getElementById(stage + "_div"));
                            var stage_tbl = document.getElementById(stage + "_table");

                            // Create table if tracker table does not exist.
                            if (!stage_tbl) {
                                stage_tbl = stage_div.append("table").attr("id", stage + "_table");
                                var header_row = stage_tbl.append("tr");
                                header_row.append("th").text("name");
                                header_row.append("th").text("cur");
                                header_row.append("th").text("unit");
                                header_row.append("th").text("avg");
                                header_row.append("th").text("std");
                                header_row.append("th").text("min");
                                header_row.append("th").text("max");
                            }

                            // Create a new tracker row.
                            var tracker_row = stage_tbl.append("tr").attr("id", stage + "/" + tracker);
                            tracker_row.append("td").text(tracker);
                            tracker_row.append("td").text(cur).attr("id", stage + "/" + tracker + "_cur");
                            tracker_row.append("td").text(unit).attr("id", stage + "/" + tracker + "_unit");
                            tracker_row.append("td").text(avg).attr("id", stage + "/" + tracker + "_avg");
                            tracker_row.append("td").text(std).attr("id", stage + "/" + tracker + "_std");
                            tracker_row.append("td").text(min).attr("id", stage + "/" + tracker + "_min");
                            tracker_row.append("td").text(max).attr("id", stage + "/" + tracker + "_max");
                        }
                    }

                    // Update tracker shortcut in nodes.
                    var target = document.getElementById(stage + "/" + tracker + "_sc");
                    if (target) {
                        target.innerHTML = tracker + ": " + cur + " " + unit;
                    }

                }
        });
    }
}

var show_trackers_in_label = (function() {
    var done = false;
    // This function will only execute once at the first time tracker info arrives.
    return function (trackers, label) {
        if (!done) {
            done = true;

            var stage_names = Object.keys(trackers);
            for (stage of stage_names) {
                d3.select(trackers[stage]).forEach((stage_obj) => {
                    var tracker_names = Object.keys(stage_obj[0]);

                    // Only scan the first two trackers in a stage.
                    for (let i = 0; i < 2; i++) {
                        var tracker = tracker_names[i];
                        if (tracker) {
                            var cur = (stage_obj[0][tracker]["cur"]["value"]).toExponential(2);
                            var unit = stage_obj[0][tracker]["unit"];

                            // Update text and id for easier search later.
                            var target = document.getElementById(stage + "_" + (i+1));
                            target.setAttribute("id", stage + "/" + tracker + "_sc");
                            target.innerHTML = tracker + ": " + cur + " " + unit;
                        }
                    }
                });
            };

        }
    }
})();

// Read from endpoint /buffers to get buffer stats
async function get_data(endpoint = "/buffers") {
    let response = await fetch(endpoint);

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
        this.start_ult();
        this.enable_sidebar();
    }

    init_svg() {
        this.#d3cola = cola.d3adaptor(d3)
            .linkDistance(80)
            .size([this.width, this.height]);

        // Add a svg section and employ zooming
        var outer = d3.select("body").append("svg")
            .attr("width", this.width)
            .attr("height", this.height)
            .attr("class", "main");

        // Add a sidebar to display all tracker info
        d3.select("body").append("div")
            .attr("class", "sidenav")
            .attr("id", "sidebar");

        outer.append('rect')
            .attr('class', 'background')
            .attr('width', "100%")
            .attr('height', "100%")
            .call(d3.behavior.zoom().on("zoom", () => {
                this.#svg.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")");
            }));

        var vis = outer
            .append('g')
            .attr('transform', 'translate(80,80) scale(0.7)');

        this.#svg = vis.append("g");
    }

    parse_data() {
        var buffers = [], stages = [], links = [];
        this.#graph = {};

        // For every buffer, add its producers and consumers to nodes and create links
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
            .call(this.#d3cola.drag);

        this.#stages = this.#svg.selectAll(".stages")
            .data(this.#graph.stages)
            .enter().append("rect")
            .attr("class", "stages")
            .attr("width", node_width)
            .attr("height", node_height)
            .attr("rx", 5).attr("ry", 5)
            .call(this.#d3cola.drag);

        // Create label objects (stage and buffer names) in svg section
        this.#buffer_labels = this.#svg.selectAll(".buffer_labels")
            .data(this.#graph.buffers)
            .enter().append("text")
            .attr("class", "buffer_labels")
            .call(this.#d3cola.drag);

        this.#stage_labels = this.#svg.selectAll(".stage_labels")
            .data(this.#graph.stages)
            .enter().append("text")
            .attr("class", "stage_labels")
            .attr("id", function(d) { return d.name; })
            .call(this.#d3cola.drag);

        var insertLinebreaks = function (d) {
            var el = d3.select(this);
            var words = d.name.split(' ');
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

    start_ult() {
        // Add utilization for buffers.
        this.#buffer_labels[0].reduce((pre, cur) => {
            var el = d3.select(cur);

            // Add CPU usage to buffers.
            var tspan = el.append('tspan')
                        .text(this.buffers[cur.textContent].num_full_frame + "/" + this.buffers[cur.textContent].num_frames)
            tspan.attr('x', this.margin/2).attr('dy', '15')
                    .attr("font-size", "15")
                    .attr("id", "utl");
        }, 0)

        // Add utilization for stages.
        this.#stage_labels[0].reduce((pre, cur) => {
            var el = d3.select(cur);
            var stage_name = cur.getAttribute("id");

            // Add title as tooltip to show details when mouse moves over.
            el.append("title").text("usr: 0%; sys: 0%");

            // Add CPU usage to stages.
            var tspan = el.append('tspan').text("CPU: 0%");
            tspan.attr('x', this.margin/2).attr('dy', '15')
                    .attr("font-size", "15")
                    .attr("id", "utl");

            // Add spots for first two trackers.
            el.append('tspan').text("")
                .attr('x', this.margin/2).attr('dy', '15')
                .attr("font-size", "15")
                .attr("id", stage_name + "_1");
            el.append('tspan').text("")
                .attr('x', this.margin/2).attr('dy', '15')
                .attr("font-size", "15")
                .attr("id", stage_name + "_2");
        }, 0)

        // Start polling kotekan for metrics
        poll(this.#buffer_labels, this.#stage_labels);
    }

    // Toggle dropdown display on every button click.
    enable_sidebar() {
        var dropdown = document.getElementsByClassName("dropdown-btn");
        var i;
        for (i = 0; i < dropdown.length; i++) {
            dropdown[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var dropdownContent = this.nextElementSibling;
            if (dropdownContent.style.display === "block") {
                dropdownContent.style.display = "none";
            } else {
                dropdownContent.style.display = "block";
            }
            });
        };

        // Add click event for each stage.
        for (let stage of this.#graph.stages) {
            var el = document.getElementById(stage.name);
            el.addEventListener("click", function (event) {
                var stage_name = event.target.parentElement.getAttribute("id");
                var stage_btn = document.getElementById(stage_name + "_button");
                var stage_div = document.getElementById(stage_name + "_div");
            
                if (stage_btn) {
                    // Remove stage from sidebar if it exists.
                    stage_btn.parentElement.removeChild(stage_btn);
                    stage_div.parentElement.removeChild(stage_div);
                } else {
                    // Add stage to sidebar.
                    var dropdown_btn = d3.select(document.getElementById("sidebar"))
                        .append("button")
                        .attr("class", "dropdown-btn")
                        .text(stage_name)
                        .attr("id", stage_name + "_button");

                    var div = document.createElement("div");
                    d3.select(div)
                        .attr("id", stage_name + "_div")
                        .attr("class", "dropdown-container");

                    var btn = document.getElementById(stage_name + "_button");
                    btn.parentNode.insertBefore(div, btn.nextElementSibling);
                }
            });
        };
    }

}
