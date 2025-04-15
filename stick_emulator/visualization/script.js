fetch('/graph_data')
    .then(response => response.json())
    .then(data => {
        // Initialize a compound graph
        var g = new dagre.graphlib.Graph({ compound: true });
        g.setGraph({ rankdir: "LR" });
        g.setDefaultEdgeLabel(function() { return {}; });

        // Add nodes with color and label
        data.nodes.forEach(node => {
            g.setNode(node.id, { label: node.label, width: node.width, height: node.height });
        });

        // Add edges with color and label
        data.edges.forEach(edge => {
            g.setEdge(edge.source, edge.target, { label: edge.label, color: edge.color });
        });

        // // Create groups
        // data.groups.forEach(group => {
        //     g.setNode(group.id, { label: group.label, width: 400, height: 200, style: "fill: #f0f0f0; stroke: #333; stroke-width: 2;" });
        //     group.nodes.forEach(nodeId => {
        //         g.setParent(nodeId, group.id); // Set parent for nodes
        //     });
        // });

        // Create the renderer
        var render = new dagreD3.render();
        var svg = d3.select("#graph").append("svg")
            .attr("width", "100%") // Full width
            .attr("height", "100%"); // Full height

        var inner = svg.append("g");

        // Create zoom behavior
        var zoom = d3.zoom()
            .scaleExtent([0.1, 3]) // Set zoom scale limits
            .on("zoom", function(event) {
                inner.attr("transform", event.transform); // Apply zoom transformation
            });

        svg.call(zoom); // Apply zoom behavior to the SVG

        // Render the graph
        render(inner, g);

        // Center the graph
        var xCenterOffset = (svg.attr("width") - g.graph().width) / 2;
        var yCenterOffset = (svg.attr("height") - g.graph().height) / 2; // Center vertically
        inner.attr("transform", "translate(" + xCenterOffset + ", " + yCenterOffset + ")");

        // Style nodes
        // inner.selectAll("g.node rect")
        //     .style("fill", d => data.nodes.find(node => node.id === d).color)
        //     .style("stroke", "#333")
        //     .style("stroke-width", 2);

        inner.selectAll("g.node")
            .select("rect")
            .remove(); // Remove the rectangle

        inner.selectAll("g.node")
            .append("circle") // Add a circle
            .attr("r", d => 70) // Set radius
            .style("fill", d => data.nodes.find(node => node.id === d).color)
            .style("stroke", "#333")
            .style("stroke-width", 2);

        inner.selectAll("g.node")
            .append("text") // Add text element
            .attr("dy", ".35em") // Vertically center the text
            .attr("text-anchor", "middle") // Center the text horizontally
            .style("fill", "#333")
            .style("font-weight", "bold")
            .style("font-size", "14px")
            .text(d => data.nodes.find(node => node.id === d).label); // Set node label text

        // Style edges
        inner.selectAll("g.edgePath path")
            .style("stroke", d => data.edges.find(edge => edge.source === d.v && edge.target === d.w).color)
            .style("stroke-width", 2)
            .style("stroke-opacity", 1)
            .style("fill", "none");

        inner.selectAll("g.edgeLabel")
            .each(function(d) {
                const edge = data.edges.find(edge => edge.source === d.v && edge.target === d.w);
                
                // Clear existing content to prevent duplication
                d3.select(this).selectAll("*").remove(); // Remove existing elements in edgeLabel
        
                // Create a new group for the label and rectangle
                const labelGroup = d3.select(this);
        
                // Add a rectangle behind the text
                const textElement = labelGroup.append("text")
                    .style("fill", "#333")
                    .style("font-size", "14px")
                    .style("font-weight", "bold")
                    .attr("dy", "-15px") // Vertically center the text
                    .attr("text-anchor", "middle") // Center the text horizontally
                    .text(edge.label); // Set edge label text
        
                const bbox = textElement.node().getBBox(); // Get bounding box of the text after it's added
        
                // Add a rectangle behind the text based on the bounding box
                labelGroup.insert("rect", "text") // Insert the rectangle before the text
                    .attr("fill", "rgba(255, 255, 255, 0.8)") // Semi-transparent background
                    .attr("rx", 3) // Rounded corners
                    .attr("ry", 3) // Rounded corners
                    .attr("x", bbox.x - 2) // Position it slightly to the left
                    .attr("y", bbox.y - 2) // Position it slightly above
                    .attr("width", bbox.width + 4) // Width based on text width
                    .attr("height", bbox.height + 4); // Height based on text height
            });

        // Add group label
        // inner.append("text")
        //     .attr("x", g.node("best_players").x - 0.31 * g.node("best_players").width) // top left corner
        //     .attr("y", g.node("best_players").y - 0.95 * g.node("best_players").height) // top left corner
        //     .attr("text-anchor", "start")
        //     .style("font-size", "14px")
        //     .style("font-weight", "bold")
        //     .text("Best Players");
    })
    .catch(error => console.error('Error fetching graph data:', error));
