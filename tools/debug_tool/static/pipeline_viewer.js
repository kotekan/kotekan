function isIE() { return ((navigator.appName == 'Microsoft Internet Explorer') || ((navigator.appName == 'Netscape') && (new RegExp("Trident/.*rv:([0-9]{1,}[\.0-9]{0,})").exec(navigator.userAgent) != null))); }

// Time between updating kotekan metrics
const POLL_WAIT_TIME_MS = 1000;
// Poll kotekan via web server every POLL_WAIT_TIME_MS and update page with new metrics
function poll(buffer_labels, stage_labels) {
    get_data("/kotekan_instance/buffers").then(function (new_buffers) {
        update_buf_utl(new_buffers, buffer_labels);
    });

    get_data("/kotekan_instance/trackers_current").then(function (trackers) {
        show_trackers_in_label(trackers);
        update_trackers(trackers, true);
    });

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

// Update sidebar table content
// stats = [cur, min, max, avg, std, timestamp, unit]
function update_table(stage, tracker, stats, isDynamic) {
    // Skip if stage is not selected.
    var stage_btn = document.getElementById(stage + "_button");
    if (stage_btn) {
        var el = document.getElementById(stage + "/" + tracker);
        var time = get_time(stats[5]);
        // If the tracker info exists, only update it.
        if (el) {
            if (!isDynamic) {
                document.getElementById(stage + "/" + tracker + "_time").innerHTML = time;
            }
            document.getElementById(stage + "/" + tracker + "_cur").innerHTML = stats[0];
            document.getElementById(stage + "/" + tracker + "_min").innerHTML = stats[1];
            document.getElementById(stage + "/" + tracker + "_max").innerHTML = stats[2];
            document.getElementById(stage + "/" + tracker + "_avg").innerHTML = stats[3];
            document.getElementById(stage + "/" + tracker + "_std").innerHTML = stats[4];
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
                if (!isDynamic) {
                    header_row.append("th").text("time");
                }
                header_row.append("th").text("avg");
                header_row.append("th").text("std");
                header_row.append("th").text("min");
                header_row.append("th").text("max");
            } else {
                stage_tbl = d3.select(stage_tbl);
            }

            // Create a new tracker row.
            var tracker_row = stage_tbl.append("tr").attr("id", stage + "/" + tracker);
            tracker_row.append("td").text(tracker);
            tracker_row.append("td").text(stats[0]).attr("id", stage + "/" + tracker + "_cur");
            tracker_row.append("td").text(stats[6]).attr("id", stage + "/" + tracker + "_unit");
            if (!isDynamic) {
                tracker_row.append("td").text(time)
                    .attr("id", stage + "/" + tracker + "_time");
            }
            tracker_row.append("td").text(stats[3]).attr("id", stage + "/" + tracker + "_avg");
            tracker_row.append("td").text(stats[4]).attr("id", stage + "/" + tracker + "_std");
            tracker_row.append("td").text(stats[1]).attr("id", stage + "/" + tracker + "_min");
            tracker_row.append("td").text(stats[2]).attr("id", stage + "/" + tracker + "_max");
        }
    }
}

function update_cpu_label(cpu_usage, stage) {
    var total = cpu_usage[0] + cpu_usage[1];
    document.getElementById(stage + "_cpu").innerHTML = "CPU: " + total + "%";
    document.getElementById(stage + "_cpu_detail").innerHTML = "usr: " + cpu_usage[0] + "% sys: " + cpu_usage[1];
}

// Update stage cpu usage.
function update_cpu_utl(cpu_stats, isDynamic, time_required){
    if ("error" in cpu_stats) {
        return;
    }

    // Process cpu stats.
    var cpu_map = new Map();
    var stage_usage = new Map();
    var tracker_names = Object.keys(cpu_stats);
    for (tracker of tracker_names) {
        var words = tracker.split("|");
        var stage_name = words[0];
        var type = words[2];
        var cur, timestamp;
        if (isDynamic) {
            cur = cpu_stats[tracker]["cur"]["value"];
        } else {
            var data = binary_search(cpu_stats[tracker]["samples"], time_required);
            cur = data[0];
            timestamp = data[1];
        }

        var avg = (cpu_stats[tracker]["avg"]).toExponential(2);
        var max = (cpu_stats[tracker]["max"]).toExponential(2);
        var min = (cpu_stats[tracker]["min"]).toExponential(2);
        var std = (cpu_stats[tracker]["std"]).toExponential(2);
        var unit = cpu_stats[tracker]["unit"];

        // Sum all threads in the same stage.
        var list = cpu_map.get(stage_name + "_" + type);
        var list_stage = stage_usage.get(stage_name);
        if (list === undefined) {
            list = [0, 0, 0, 0, 0, 0, 0];  // [cur, min, max, avg, std, timestamp, unit]
            if (list_stage === undefined) {
                list_stage = [0, 0]  // [usr, sys]
            }
        }
        list[0] += cur;
        list[1] += min;
        list[2] += max;
        list[3] += avg;
        list[4] += std;
        list[5] = Math.max(list[5], timestamp);
        list[6] = unit;
        cpu_map.set(stage_name + "_" + type, list);

        if (type == "usr") {
            list_stage[0] = cur;
        } else {
            list_stage[1] = cur;
        }
        stage_usage.set(stage_name, list_stage);
    }

    // Display cpu usage.
    for ([key, value] of cpu_map) {
        var stage_name = key.substring(0, key.length - 4);
        var type = key.substring(key.length - 4);
        update_table(stage_name, "cpu" + type, value, isDynamic);
        update_cpu_label(stage_usage.get(stage_name), stage_name);
    }
}

function update_stats(tracker, stage, isDynamic, time_required) {
    d3.select(tracker).forEach((stage_obj) => {

        var tracker_name = Object.keys(stage_obj[0]);
        for (tracker of tracker_name) {
            // Limit to two-decimal scientific expression.
            var avg = (stage_obj[0][tracker]["avg"]).toExponential(2);
            var max = (stage_obj[0][tracker]["max"]).toExponential(2);
            var min = (stage_obj[0][tracker]["min"]).toExponential(2);
            var std = (stage_obj[0][tracker]["std"]).toExponential(2);
            var unit = stage_obj[0][tracker]["unit"];
            var cur;
            var time;

            if (isDynamic) {
                cur = (stage_obj[0][tracker]["cur"]["value"]).toExponential(2);
            } else {
                // Dump viewer shows the last sample before the given timestamp.
                var data = binary_search(stage_obj[0][tracker]["samples"], time_required);
                cur = (data[0]).toExponential(2);
                time = data[1];
            }

            update_table(stage, tracker, [cur, min, max, avg, std, time, unit], isDynamic);

            // Update tracker shortcut in nodes.
            var target = document.getElementById(stage + "/" + tracker + "_sc");
            if (target) {
                target.innerHTML = tracker + ": " + cur + " " + unit;
            }

        }
    });
}

// Find the largest timestamp before the given time.
var binary_search = function (arr, x) {
    let len = arr.length;
    if (len == 0) {
        return [NaN, NaN];
    }

    let start = 0;
    let end = len - 1;

    // Return NaN if no time earlier than the given time.
    if (arr[start]["timestamp"] > x) {
        return [NaN, NaN];
    }
    if (arr[end]["timestamp"] <= x) {
        return [arr[end]["value"], arr[end]["timestamp"]];
    }

    while (start < end - 1){
        let mid = Math.floor((start + end)/2);
        if (arr[mid]["timestamp"] == x) {
            return  [arr[mid]["value"], arr[mid]["timestamp"]];
        } else if (arr[mid]["timestamp"] < x) {
            start = mid;
        } else {
            end = mid;
        }
    }
    return [arr[start]["value"], arr[start]["timestamp"]];
}

function update_trackers(trackers, isDynamic, time_required){
    var stage_names = Object.keys(trackers);
    for (stage of stage_names) {
        if (stage == "cpu_monitor") {
            update_cpu_utl(trackers[stage], isDynamic, time_required);
            continue;
        } else {
            update_stats(trackers[stage], stage, isDynamic, time_required);
        }
    }
}

var show_trackers_in_label = (function() {
    var done = false;
    // This function will only execute once at the first time tracker info arrives.
    return function (trackers) {
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

// Read useful info from endpoint
async function get_data(endpoint) {
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
        // Get returned content
        if (endpoint === "/dump_dir") {
            // Parse HTML text to a list of file names
            var htmlString = await response.text();
            var doc = new DOMParser().parseFromString(htmlString, "text/html");
            var elements = doc.querySelectorAll("a");
            var files = [];
            elements.forEach(function(el) {
                var text = ((el.innerHTML).replaceAll("\n", "")).replaceAll(" ", "");
                files.push(text);
            })
            return files;
        } else if (endpoint.includes("crash_stats")) {
            // Separate buffer and tracker info from dump file
            var obj = await response.json();
            var buffers = obj["buffers"];
            var trackers = obj["trackers"];
            return [buffers, trackers];
        } else {
            return await response.json();
        }
    }
}

// Sort objects in ascending order.
function sort_by_key(array, key) {
    return array.sort(function(a, b) {
        var x = a[key];
        var y = b[key];
        return ((x < y) ? -1 : ((x > y) ? 1 : 0));
    });
}

// Get formatted time from timestamp.
function get_time(timestamp) {
    if (Number.isNaN(timestamp)) {
        return NaN;
    }
    var date = new Date(timestamp);
    var minutes = String(date.getMinutes()).padStart(2, "0");
    var seconds = String(date.getSeconds()).padStart(2, "0");
    var miliseconds = String(date.getMilliseconds()).padStart(3, "0");

    return minutes + ":" + seconds + ":" + miliseconds;
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

    constructor(buffers, body, trackers) {
        this.buffers = buffers;
        this.bufNames = Object.keys(this.buffers);
        this.body = body;
        this.trackers = trackers;

        if (trackers) {
            // Dump viewer
            this.process_timestamp(trackers);
            this.isDynamic = false;
        } else {
            this.isDynamic = true;
        }
    }

    // Sort samples and find min/max timestamp.
    process_timestamp(trackers) {
        var max = 0;
        var min = Number.MAX_VALUE;

        var stage_names = Object.keys(trackers);
        for (const stage of stage_names) {
            d3.select(trackers[stage]).forEach((stage_obj) => {

                var tracker_name = Object.keys(stage_obj[0]);
                for (const tracker of tracker_name) {
                    var samples = stage_obj[0][tracker]["samples"];
                    samples = sort_by_key(samples, "timestamp");

                    min = Math.min(min, samples[0]["timestamp"]);
                    max = Math.max(max, samples[samples.length - 1]["timestamp"]);
                }
            });
        }

        this.time_min = min;
        this.time_max = max;
    }

    start_viewer(width = 960, height = 500, margin = 6) {
        this.width = width;
        this.height = height;
        this.margin = margin;
        this.clear();
        this.init_svg();
        this.parse_data();
        this.create_objs();
        this.enable_sidebar();
        this.reserve_tracker_space();
        this.add_cpu_label();

        if (this.isDynamic) {
            this.add_buffer_ult();
            this.start_ult();
        } else {
            this.set_up_slider();
        }
    }

    init_svg() {
        this.#d3cola = cola.d3adaptor(d3)
            .linkDistance(80)
            .size([this.width, this.height]);

        // Add a svg section and employ zooming
        var outer = d3.select("body").append("svg")
            .attr("width", this.width)
            .attr("height", this.height)
            .attr("class", "main")
            .attr("id", "main");

        // Add a sidebar to display all tracker info
        d3.select("body").append("div")
            .attr("class", "sidenav")
            .attr("id", "sidebar");

        var rect = outer.append('rect')
            .attr('class', 'background')
            .attr('width', "100%")
            .attr('height', "100%");

        var vis = outer
            .append('g')
            .attr('transform', 'translate(80,80) scale(0.7)');

        this.#svg = vis.append("g").attr("id", "svg_group");
        rect.call(d3.behavior.zoom().on("zoom", () => {
            this.#svg.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")");
        }));
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

    reserve_tracker_space() {
        this.#stage_labels[0].reduce((pre, cur) => {
            var el = d3.select(cur);
            var stage_name = cur.getAttribute("id");

            // Add spots for first two trackers.
            el.append('tspan').text("")
                .attr('x', this.margin/2).attr('dy', '15')
                .attr("font-size", "15")
                .attr("id", stage_name + "_1");
            el.append('tspan').text("")
                .attr('x', this.margin/2).attr('dy', '15')
                .attr("font-size", "15")
                .attr("id", stage_name + "_2");
        }, 0);
    }

    add_cpu_label() {
        this.#stage_labels[0].reduce((pre, cur) => {
            var el = d3.select(cur);
            var stage_name = cur.getAttribute("id");

            // Add title as tooltip to show details when mouse moves over.
            el.append("title").text("usr: 0%; sys: 0%")
                .attr("id", stage_name + "_cpu_detail");

            // Add CPU usage to stages.
            var tspan = el.append('tspan').text("CPU: 0%");
            tspan.attr('x', this.margin/2).attr('dy', '15')
                    .attr("font-size", "15")
                    .attr("id", stage_name + "_cpu");
        }, 0);
    }

    add_buffer_ult() {
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
    }

    start_ult() {
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

    // Clear previous graph and sidebar
    clear() {
        var graph = document.getElementById("main");
        var sidebar = document.getElementById("sidebar");
        if (graph) {
            graph.parentElement.removeChild(graph);
        }
        if (sidebar) {
            sidebar.parentElement.removeChild(sidebar);
        }
    }

    // Connect with slider input
    set_up_slider() {
        var slider = document.getElementById("slide_input");
        var output = document.getElementById("slide_value");
        var trackers = this.trackers;

        // Reset slider value to 100%
        slider.value = 100;
        output.innerHTML = get_time(this.time_max);

        // Show first two trackers
        var stage_names = Object.keys(trackers);
        for (let stage of stage_names) {
            if (stage == "cpu_monitor") continue;
            d3.select(trackers[stage]).forEach((stage_obj) => {
                var tracker_names = Object.keys(stage_obj[0]);

                // Only scan the first two trackers in a stage.
                for (let i = 0; i < 2; i++) {
                    var tracker = tracker_names[i];
                    if (tracker) {
                        // Dump viewer shows the latest value.
                        var samples = stage_obj[0][tracker]["samples"];
                        var cur = (samples[samples.length - 1]["value"]).toExponential(2);
                        var unit = stage_obj[0][tracker]["unit"];

                        // Update text and id for easier search later.
                        var target = document.getElementById(stage + "_" + (i+1));
                        target.setAttribute("id", stage + "/" + tracker + "_sc");
                        target.innerHTML = tracker + ": " + cur + " " + unit;
                    }
                }
            });
        };

        var time_min = this.time_min;
        var time_max = this.time_max;
        update_trackers(trackers, false, time_max);

        // Slider callback function
        slider.oninput = function() {
            var percent = this.value / 100;
            var time_required = Math.floor((time_max - time_min) * percent + time_min);

            // Show required time in minute:second:milisec format
            output.innerHTML = get_time(time_required);

            update_trackers(trackers, false, time_required);
        }
    }
}
