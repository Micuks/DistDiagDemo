import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { makeStyles } from '@material-ui/core/styles';
import { Card, CardContent, Typography } from '@material-ui/core';

const useStyles = makeStyles((theme) => ({
  root: {
    width: '100%',
    height: '600px',
    marginTop: theme.spacing(2),
  },
  graphContainer: {
    width: '100%',
    height: '100%',
    position: 'relative',
  },
  tooltip: {
    position: 'absolute',
    padding: theme.spacing(1),
    background: 'rgba(0, 0, 0, 0.8)',
    color: 'white',
    borderRadius: theme.spacing(1),
    pointerEvents: 'none',
  },
}));

const AnomalyGraph = ({ data, onNodeClick }) => {
  const classes = useStyles();
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (!data || !data.nodes || !data.edges) return;

    // Clear previous graph
    d3.select(svgRef.current).selectAll('*').remove();

    // Set up SVG
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Create arrow marker for directed edges
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('xoverflow', 'visible')
      .append('svg:path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#999')
      .style('stroke', 'none');

    // Set up force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges)
        .id(d => d.id)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));

    // Create edges
    const edges = svg.append('g')
      .selectAll('line')
      .data(data.edges)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.weight) * 2)
      .attr('marker-end', 'url(#arrowhead)');

    // Create nodes
    const nodes = svg.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('r', 10)
      .attr('fill', d => getNodeColor(d.anomaly_prob))
      .call(drag(simulation));

    // Add node labels
    const labels = svg.append('g')
      .selectAll('text')
      .data(data.nodes)
      .enter()
      .append('text')
      .text(d => d.id)
      .attr('font-size', 12)
      .attr('dx', 15)
      .attr('dy', 4);

    // Set up tooltip
    const tooltip = d3.select(tooltipRef.current);

    nodes
      .on('mouseover', (event, d) => {
        tooltip.style('opacity', 1)
          .html(`
            <div>
              <strong>${d.id}</strong><br/>
              Anomaly Prob: ${(d.anomaly_prob * 100).toFixed(2)}%<br/>
              CPU: ${d.cpu}%<br/>
              Memory: ${d.memory}%<br/>
              I/O: ${d.io}%
            </div>
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('opacity', 0);
      })
      .on('click', (event, d) => {
        if (onNodeClick) onNodeClick(d);
      });

    // Update force simulation
    simulation.on('tick', () => {
      edges
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      nodes
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

      labels
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });

    // Clean up
    return () => {
      simulation.stop();
    };
  }, [data, onNodeClick]);

  // Helper function for node color based on anomaly probability
  const getNodeColor = (probability) => {
    if (!probability) return '#69b3a2';
    const red = Math.floor(255 * probability);
    const green = Math.floor(255 * (1 - probability));
    return `rgb(${red},${green},0)`;
  };

  // Drag handler
  const drag = (simulation) => {
    const dragstarted = (event) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    };

    const dragged = (event) => {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    };

    const dragended = (event) => {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    };

    return d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  };

  return (
    <Card className={classes.root}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Anomaly Graph Visualization
        </Typography>
        <div className={classes.graphContainer}>
          <svg ref={svgRef} />
          <div ref={tooltipRef} className={classes.tooltip} style={{ opacity: 0 }} />
        </div>
      </CardContent>
    </Card>
  );
};

export default AnomalyGraph; 