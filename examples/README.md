# Examples

This directory contains Jupyter notebooks that demonstrate the main parts of
the project:

- `01_intervals_and_circle.ipynb`: interval and circle primitives
- `02_object_zoo_visualization.ipynb`: Euclidean object zoo with plots
- `03_riemannian_objects_and_projections.ipynb`: metric objects, set
  operations, projections, and smooth images
- `04_set_operations_2d.ipynb`: 2D set-theoretic operations with plotted
  unions, intersections, differences, and symmetric differences
- `08_modern_space_object_mesh_pipeline.ipynb`: current end-to-end pipeline
  from spaces and objects to meshes, plots, and export data

It also contains runnable Python scripts for the newer end-to-end workflows:

- `05_space_object_mesh_pipeline.py`: `space -> native object -> mesh`
- `06_metric_object_zoo_pipeline.py`: broader metric-object zoo sampling/mesh
- `07_plot_and_export_pipeline.py`: `mesh -> matplotlib/plotly -> OBJ/PLY/glTF`

Recommended setup:

```sh
pip install -e ".[examples]"
jupyter lab
```

Runnable scripts:

```sh
python examples/05_space_object_mesh_pipeline.py
python examples/06_metric_object_zoo_pipeline.py
python examples/07_plot_and_export_pipeline.py
```

The notebooks also prepend the repository root to `sys.path`
automatically, so they can be started directly from `examples/`
during local development.
